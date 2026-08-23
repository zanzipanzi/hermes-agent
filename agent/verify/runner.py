"""Verification runner: execute a Recipe's phases and smoke-test the app.

Scoped port of the execution flow grok-cli's verify sub-agent performs
(install/bootstrap -> build -> test -> start in background -> curl-style
readiness loop -> teardown), reimplemented as a plain subprocess runner.

Commands come from the project's own recipe (its package.json scripts,
Makefile targets, etc.) and are executed with ``shell=True`` on purpose:
this is a developer tool running the project's own build commands in the
project's own checkout — the same trust level as the terminal tool.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.verify.recipes import Recipe
from tools.process_lifecycle import (
    TerminationStatus,
    capture_process_identity,
    terminate_process_tree,
)

DEFAULT_PHASE_TIMEOUT = 600.0
DEFAULT_READY_TIMEOUT = 60.0
_TAIL_CHARS = 2000
# Hard bound on captured phase output. ``on_output`` historically received
# the full stream; this cap keeps a runaway chatty command from growing the
# buffer indefinitely while staying far above anything a real build emits.
_OUTPUT_CAPTURE_CAP = 200_000
PHASE_ORDER = ("bootstrap", "build", "test")


@dataclass
class PhaseResult:
    phase: str
    command: str
    exit_code: int | None
    duration: float
    output_tail: str
    timed_out: bool = False
    teardown_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "command": self.command,
            "exitCode": self.exit_code,
            "duration": round(self.duration, 3),
            "ok": self.ok,
            "timedOut": self.timed_out,
            "teardownFailed": self.teardown_failed,
            "outputTail": self.output_tail,
        }


@dataclass
class ReadinessResult:
    url: str
    ready: bool
    status_code: int | None
    duration: float
    error: str | None = None
    output_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "ready": self.ready,
            "statusCode": self.status_code,
            "duration": round(self.duration, 3),
            "error": self.error,
            "outputTail": self.output_tail,
        }


@dataclass
class VerifyResult:
    recipe_name: str
    phases: list[PhaseResult] = field(default_factory=list)
    readiness: ReadinessResult | None = None

    @property
    def ok(self) -> bool:
        phases_ok = all(p.ok for p in self.phases)
        readiness_ok = self.readiness.ready if self.readiness is not None else True
        return phases_ok and readiness_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe_name,
            "ok": self.ok,
            "phases": [p.to_dict() for p in self.phases],
            "readiness": self.readiness.to_dict() if self.readiness else None,
        }


def _tail(text: str, limit: int = _TAIL_CHARS) -> str:
    return text[-limit:] if len(text) > limit else text


class _OutputDrain:
    """Continuously drain a text pipe into a bounded buffer from a thread.

    A chatty child (dev servers logging every request) can fill the OS pipe
    buffer in kilobytes; without a concurrent reader the child blocks on its
    own stdout and never becomes ready. Reading only after teardown both
    wedged cleanup and lost the output. ``output()`` is bounded to
    ``_OUTPUT_CAPTURE_CAP`` characters (front-trimmed, tail preserved); the
    daemon reader can never hang its owner — ``wait()`` is bounded and a
    broken/invalid pipe is swallowed.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        self._chunks: list[str] = []
        self._captured = 0
        self.done = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def _drain(self) -> None:
        try:
            for line in self._stream:
                self._chunks.append(line)
                self._captured += len(line)
                while self._captured > _OUTPUT_CAPTURE_CAP and len(self._chunks) > 1:
                    self._captured -= len(self._chunks[0])
                    self._chunks.pop(0)
                if self._captured > _OUTPUT_CAPTURE_CAP and self._chunks:
                    self._chunks[0] = self._chunks[0][
                        self._captured - _OUTPUT_CAPTURE_CAP :
                    ]
                    self._captured = _OUTPUT_CAPTURE_CAP
        except (OSError, ValueError):
            pass
        finally:
            self.done.set()

    def start(self) -> "_OutputDrain":
        self._thread.start()
        return self

    def wait(self, timeout: float) -> None:
        """Bounded reader shutdown; returns even if the pipe never EOFs."""
        self.done.wait(timeout=timeout)

    def output(self) -> str:
        return "".join(self._chunks)[-_OUTPUT_CAPTURE_CAP:]


def _run_phase_command(
    phase: str,
    command: str,
    root: Path,
    timeout: float,
    on_output: Callable[[str], None] | None = None,
) -> PhaseResult:
    """Run one phase command as an owned process tree.

    Unlike a bare ``subprocess.run(..., timeout=...)`` — which on Windows
    kills only the direct ``cmd.exe`` wrapper and then blocks forever in an
    unbounded ``communicate()`` while surviving grandchildren hold the stdout
    pipe — this helper:

    - captures the wrapper's PID identity (PID + kernel start time) up front;
    - continuously drains combined stdout/stderr into a bounded buffer from a
      reader thread, so a chatty child can never wedge the pipe;
    - waits against a monotonic deadline and, on timeout, reaps the *whole*
      owned tree via the shared lifecycle seam;
    - returns only after bounded teardown verification and reports a
      ``teardown_failed`` phase instead of silently claiming completion.
    """
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        shell=True,  # project-authored commands; see module docstring
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    identity = capture_process_identity(proc.pid)
    drain = _OutputDrain(proc.stdout).start()

    exit_code: int | None = None
    timed_out = False
    teardown_failed = False
    try:
        exit_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        reap = terminate_process_tree(identity)
        if reap.status is TerminationStatus.failed:
            teardown_failed = True
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            teardown_failed = True
        if not teardown_failed:
            # Bounded verification that the tree is actually gone; report a
            # failed teardown instead of claiming completion.
            teardown_failed = not _pid_gone_within(identity.pid, 5.0)

    # Bounded reader shutdown: EOF arrives once every writer in the tree is
    # dead. A wedged reader must never wedge teardown in turn.
    drain.wait(timeout=5.0)
    output = drain.output()

    duration = time.monotonic() - started
    if on_output and output:
        on_output(output)
    return PhaseResult(
        phase=phase,
        command=command,
        exit_code=exit_code,
        duration=duration,
        output_tail=_tail(output),
        timed_out=timed_out,
        teardown_failed=teardown_failed,
    )


def _pid_gone_within(pid: int, timeout: float, interval: float = 0.05) -> bool:
    import psutil

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(interval)
    return not psutil.pid_exists(pid)


def _poll_readiness(url: str, timeout: float, interval: float = 1.0) -> tuple[bool, int | None, str | None]:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return True, resp.status, None
        except urllib.error.HTTPError as exc:
            # The server answered — it is up, even if it returned 4xx/5xx.
            return True, exc.code, None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(interval)
    return False, None, last_error


def _listener_pids(port: int) -> list[int]:
    """PIDs of processes listening on 127.0.0.1/0.0.0.0 ``port`` (best effort).

    Returns ``[]`` when nothing is listening; raises nothing — platforms we
    can't inspect fall through to the caller's post-readiness ownership
    proof, which is the authoritative fail-closed check.
    """
    try:
        import psutil

        pids = set()
        for conn in psutil.net_connections(kind="inet"):
            if (
                conn.status == psutil.CONN_LISTEN
                and conn.laddr
                and conn.laddr.port == port
                and conn.pid
            ):
                pids.add(conn.pid)
        return sorted(pids)
    except Exception:
        return []


def _owned_pids(root_pid: int) -> set[int]:
    """The root PID plus all of its live descendants (snapshot)."""
    import psutil

    owned = {root_pid}
    try:
        for child in psutil.Process(root_pid).children(recursive=True):
            owned.add(child.pid)
    except Exception:
        pass
    return owned


def _run_start_phase(
    recipe: Recipe,
    root: Path,
    ready_timeout: float,
    port_override: int | None = None,
) -> ReadinessResult:
    assert recipe.start is not None
    port = port_override or recipe.port or 8000
    url = f"http://127.0.0.1:{port}{recipe.readiness_path}"
    started = time.monotonic()

    # Fail closed BEFORE spawn: a pre-existing listener must never count as
    # our readiness, and Eleventy-class dev servers silently advance to the
    # next free port when the requested one is taken — which is how a stale
    # listener manufactured false "ready" verdicts and orphaned servers.
    existing = _listener_pids(port)
    if existing:
        return ReadinessResult(
            url=url,
            ready=False,
            status_code=None,
            duration=time.monotonic() - started,
            error=(
                f"port {port} already in use (listener pid "
                f"{', '.join(str(p) for p in existing)}); refusing to start"
            ),
        )

    proc = subprocess.Popen(
        recipe.start,
        shell=True,  # project-authored command; see module docstring
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group for clean teardown
        text=True,
        errors="replace",
    )
    identity = capture_process_identity(proc.pid)
    # Drain while readiness polls: a chatty dev server fills the OS pipe
    # buffer in kilobytes and blocks on its own stdout — never becoming
    # ready — unless something reads concurrently. Reading only after
    # teardown also wedged cleanup whenever a survivor held the pipe.
    drain = _OutputDrain(proc.stdout).start()
    try:
        ready, status, error = _poll_readiness(url, ready_timeout)
        if ready:
            # The URL answering is not enough: prove the listener belongs to
            # the tree we just spawned. A foreign listener (race with an
            # unrelated process binding between our preflight and now, or a
            # server that silently relocated to another port) must fail
            # readiness, never pass it.
            listeners = _listener_pids(port)
            owned = _owned_pids(proc.pid)
            if not listeners or not set(listeners).issubset(owned):
                ready, status = False, None
                error = (
                    f"listener on port {port} not owned by started process "
                    f"(listener pids {listeners}, owned pids {sorted(owned)})"
                )
    finally:
        # Reap the whole owned tree — killing only the direct shell leaves
        # grandchildren (npm → cmd → eleventy) alive holding the stdout pipe.
        terminate_process_tree(identity)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # Bounded reader shutdown; the daemon thread can never hang us.
        drain.wait(timeout=5.0)
    return ReadinessResult(
        url=url,
        ready=ready,
        status_code=status,
        duration=time.monotonic() - started,
        error=error,
        output_tail=_tail(drain.output()),
    )


def run_verify(
    root: Path,
    recipe: Recipe,
    phases: tuple[str, ...] | list[str] | None = None,
    phase_timeout: float = DEFAULT_PHASE_TIMEOUT,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
    skip_start: bool = False,
    port_override: int | None = None,
    stop_on_failure: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> VerifyResult:
    """Run a verify pass for ``recipe`` at project ``root``.

    Executes the selected command phases sequentially, then (unless
    ``skip_start`` or a phase failed) launches ``recipe.start`` in the
    background, polls the readiness URL, and tears the process group down.
    """
    root = Path(root)
    selected = tuple(phases) if phases else PHASE_ORDER + ("start",)
    result = VerifyResult(recipe_name=recipe.name)

    failed = False
    for phase in PHASE_ORDER:
        if phase not in selected:
            continue
        for command in getattr(recipe, phase):
            phase_result = _run_phase_command(phase, command, root, phase_timeout, on_output)
            result.phases.append(phase_result)
            if not phase_result.ok:
                failed = True
                if stop_on_failure:
                    return result

    if skip_start or "start" not in selected or failed or not recipe.start:
        return result

    result.readiness = _run_start_phase(recipe, root, ready_timeout, port_override)
    return result
