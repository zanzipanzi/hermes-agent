"""Shared, PID-reuse-safe termination of process trees we spawned.

One seam for verifier phases, the terminal backend, and the process registry
so no module grows a private tree-kill copy. The behavior was extracted from
``ProcessRegistry._terminate_host_pid`` (the strongest existing
implementation) and is deliberately unchanged on either platform:

- Identity guard: callers capture ``(pid, kernel start time)`` while they own
  the process and pass it back at teardown. The kernel recycles PID numbers,
  so a stale PID can name an *unrelated* process; a start-time mismatch means
  the number was recycled and must never be signalled. When no baseline was
  captured (``start_time is None``) we degrade to best-effort signalling of
  the bare PID, preserving legacy checkpoint behaviour.
- Windows: ``taskkill /PID <pid> /T /F`` — the documented Microsoft
  tree-kill primitive, exact PID only, hidden console window, bounded wait.
  We can't reuse the psutil walk on Windows because the OS doesn't maintain a
  Unix-style process tree (PPID links go stale when intermediates exit) and
  ``Process.terminate()`` there is a single-handle ``TerminateProcess()``.
- POSIX: snapshot descendants with psutil, SIGTERM children before the
  parent (so subprocess trees aren't reparented to init and survive), then
  after a bounded grace window SIGKILL any survivor that ignored SIGTERM.
  Survivors are re-probed directly rather than trusting ``wait_procs``
  partitioning (mis-partitioned across parent/child trees in the wild).

``gateway.status.terminate_pid`` is a separate, older utility with its own
raising contract and callers; it is intentionally left untouched here.

Never use this module to kill by executable name, and never fall back from a
proven identity mismatch to an unguarded kill: leaking an orphan is strictly
preferable to killing a stranger that recycled our PID.
"""

from __future__ import annotations

import logging
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


class TerminationStatus(str, Enum):
    """Outcome of a :func:`terminate_process_tree` call."""

    terminated = "terminated"
    already_exited = "already_exited"
    refused_identity_mismatch = "refused_identity_mismatch"
    failed = "failed"


@dataclass(frozen=True)
class ProcessIdentity:
    """A PID plus the kernel start-time fingerprint captured while we owned it.

    ``(pid, start_time)`` uniquely identifies a process on a host: a recycled
    PID (same number, different process) yields a different start time and is
    never mistaken for the original.
    """

    pid: int
    start_time: Optional[int] = None


@dataclass(frozen=True)
class TerminationResult:
    status: TerminationStatus
    detail: str = ""


def capture_process_identity(pid: int) -> ProcessIdentity:
    """Snapshot a process identity while it is still ours.

    The start-time fingerprint is ``None`` when the process already exited or
    the platform can't read it; :func:`terminate_process_tree` then degrades
    to best-effort signalling of the bare PID (legacy behaviour).
    """
    from gateway.status import get_process_start_time

    try:
        start_time = get_process_start_time(pid)
    except Exception:
        start_time = None
    return ProcessIdentity(pid=pid, start_time=start_time)


def _proc_alive(proc) -> bool:
    """True if a psutil.Process is running and not a zombie.

    A zombie is already dead (just unreaped), so there's nothing to signal.
    """
    try:
        import psutil

        if not proc.is_running():
            return False
        return proc.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _pid_exists(pid: int) -> bool:
    from gateway.status import _pid_exists as _exists

    return _exists(pid)


def _current_start_time(pid: int) -> Optional[int]:
    from gateway.status import get_process_start_time

    try:
        return get_process_start_time(pid)
    except Exception:
        return None


def terminate_process_tree(
    identity: ProcessIdentity,
    *,
    grace_seconds: float = 2.0,
) -> TerminationResult:
    """Terminate an owned PID and its descendants, PID-reuse-safe.

    ``grace_seconds`` bounds the POSIX SIGTERM→SIGKILL escalation window
    (ignored on Windows where ``/F`` is already a hard kill).
    """
    pid = identity.pid

    if identity.start_time is not None:
        if not _pid_exists(pid):
            return TerminationResult(TerminationStatus.already_exited)
        if _current_start_time(pid) != identity.start_time:
            # PID was recycled (start time changed) — never signal a
            # stranger. A leaked orphan is strictly preferable to killing
            # e.g. a browser whose session leader reused this dead PID.
            logger.warning(
                "Refusing to terminate host pid %d: start-time mismatch — "
                "PID was recycled onto an unrelated process.",
                pid,
            )
            return TerminationResult(TerminationStatus.refused_identity_mismatch)

    if _IS_WINDOWS:
        return _terminate_tree_windows(pid)

    return _terminate_tree_posix(pid, grace_seconds)


def _terminate_tree_windows(pid: int) -> TerminationResult:
    from hermes_cli._subprocess_compat import windows_hide_flags

    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=windows_hide_flags(),
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        # Missing/unusable taskkill (effectively unreachable on real Windows
        # installs): fall back to a bare single-PID signal, best-effort.
        try:
            os.kill(pid, signal.SIGTERM)
            return TerminationResult(TerminationStatus.terminated)
        except (OSError, ProcessLookupError, PermissionError):
            return TerminationResult(TerminationStatus.failed, detail=str(exc))

    if result.returncode == 0:
        return TerminationResult(TerminationStatus.terminated)

    details = (result.stderr or result.stdout or "").strip()
    if not _pid_exists(pid):
        # taskkill reported failure because the PID is already gone.
        return TerminationResult(TerminationStatus.already_exited, detail=details)
    return TerminationResult(
        TerminationStatus.failed,
        detail=details or f"taskkill exited {result.returncode} for PID {pid}",
    )


def _terminate_tree_posix(pid: int, grace_seconds: float) -> TerminationResult:
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return TerminationResult(TerminationStatus.already_exited)
    except (OSError, PermissionError) as exc:
        try:
            os.kill(pid, signal.SIGTERM)
            return TerminationResult(TerminationStatus.terminated)
        except (OSError, ProcessLookupError, PermissionError):
            return TerminationResult(TerminationStatus.failed, detail=str(exc))

    # Snapshot the whole tree (children before parent) and SIGTERM each.
    try:
        targets = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        targets = []
    targets.append(parent)

    for proc in targets:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError):
            pass

    if grace_seconds <= 0:
        return TerminationResult(TerminationStatus.terminated)

    # Escalate to SIGKILL for anything that ignored SIGTERM within the grace
    # window — a daemon stalled in its signal handler would otherwise leak
    # indefinitely. We deliberately do NOT trust ``psutil.wait_procs``'s
    # gone/alive partition: it reaps via ``Process.wait()`` and can
    # mis-partition when a target transitions through a zombie state or when
    # reaping is racy across a parent/child tree, which left survivors
    # un-killed. A direct liveness re-probe is deterministic.
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not any(_proc_alive(p) for p in targets):
            break
        time.sleep(0.05)
    for proc in targets:
        try:
            if not _proc_alive(proc):
                continue
            proc.kill()  # SIGKILL on POSIX
            logger.info(
                "Escalated to SIGKILL for pid %d (ignored SIGTERM within "
                "%.1fs grace)",
                proc.pid,
                grace_seconds,
            )
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError):
            pass

    return TerminationResult(TerminationStatus.terminated)
