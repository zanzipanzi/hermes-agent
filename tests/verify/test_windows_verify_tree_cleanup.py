"""Windows real-process regression suite for verifier process ownership.

Every test builds a real ``cmd.exe → python parent → python child`` tree via
the shared ``tests/fixtures/verify_process_tree.py`` fixture, records PIDs,
and asserts the hardened contract: owned trees are reaped on normal
readiness teardown, on readiness timeout, and on phase timeout; a foreign
listener is preserved while verify fails closed before spawn; and repeated
start/stop cycles leave zero descendants and zero listeners behind.

Cleanup kills only processes whose own command line proves ownership (the
fixture-script path or the embedded ``vpt-marker`` child tag) through the
PID+start-time guarded lifecycle seam — never by executable name. If
ownership cannot be proven, the process is left alone.
"""

import http.server
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.verify.recipes import Recipe
from agent.verify.runner import run_verify
from tools.process_lifecycle import (
    capture_process_identity,
    terminate_process_tree,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows real-process tree cleanup"
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "verify_process_tree.py"
_OWNERSHIP_NEEDLES = ("verify_process_tree.py", "# vpt-marker")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _listener_pids(port: int) -> list[int]:
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


def _fixture_proc_pids() -> list[int]:
    import psutil

    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except Exception:
            continue
        if any(needle in cmdline for needle in _OWNERSHIP_NEEDLES):
            found.append(proc.info["pid"])
    return found


def _pid_gone(pid: int) -> bool:
    import psutil

    return not psutil.pid_exists(pid)


def _wait_until(predicate, timeout: float, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _read_marker(marker: Path, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            text = marker.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        time.sleep(0.05)
    pytest.fail(f"fixture parent never wrote its child PID to {marker}")


def _reap_all_fixtures(*markers) -> None:
    """Tear down only verifiably-owned fixture trees (see module docstring).

    A process is signalled only when its own command line proves ownership:
    the wrapper/parent's cmdline contains one of this test's unique marker
    paths (or the tmp_path that scopes them), or a marker-file PID is alive
    AND carries the ``# vpt-marker`` child tag — a recycled PID fails the
    tag check and is left alone. Generic fixture-path scanning stays
    detection-only (``_fixture_proc_pids``) and never kills.
    """
    import psutil

    marker_paths = {str(m) for m in markers if m is not None}
    marker_files = [m for m in markers if m is not None and m.is_file()]
    targets: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except Exception:
            continue
        if any(path in cmdline for path in marker_paths) and proc.info["pid"] != 0:
            targets.append(proc.info["pid"])
    for marker in marker_files:
        try:
            child_pid = int(marker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if child_pid in targets:
            continue
        try:
            cmdline = " ".join(psutil.Process(child_pid).cmdline() or [])
        except Exception:
            cmdline = ""
        if "# vpt-marker" in cmdline:
            targets.append(child_pid)
        elif psutil.pid_exists(child_pid):
            print(
                f"NOTE: marker pid {child_pid} is alive but not verifiably "
                "ours (cmdline mismatch) — left untouched"
            )
    for pid in targets:
        try:
            if psutil.pid_exists(pid):
                terminate_process_tree(capture_process_identity(pid))
        except Exception:
            pass


def _serve_command(marker: Path, port: int) -> str:
    return f'"{sys.executable}" "{_FIXTURE}" "{marker}" --serve {port}'


def _tree_command(marker: Path) -> str:
    return f'"{sys.executable}" "{_FIXTURE}" "{marker}"'


class TestWindowsTreeCleanup:
    def test_normal_readiness_teardown_removes_nested_tree(self, tmp_path):
        marker = tmp_path / "child_pid.txt"
        port = _free_port()
        try:
            result = run_verify(
                tmp_path,
                Recipe(name="x", start=_serve_command(marker, port), port=port),
                phases=("start",),
                ready_timeout=30,
            )
            assert result.readiness is not None and result.readiness.ready
            child_pid = _read_marker(marker)
            assert _wait_until(lambda: _pid_gone(child_pid), timeout=15.0), (
                f"nested child {child_pid} survived normal readiness teardown"
            )
            assert _wait_until(lambda: not _fixture_proc_pids(), timeout=15.0), (
                "fixture tree survived normal readiness teardown"
            )
            assert _listener_pids(port) == [], "port must be freed after teardown"
        finally:
            _reap_all_fixtures(marker)

    def test_readiness_timeout_removes_nested_tree(self, tmp_path):
        marker = tmp_path / "child_pid.txt"
        port = _free_port()
        try:
            result = run_verify(
                tmp_path,
                Recipe(name="x", start=_tree_command(marker), port=port),
                phases=("start",),
                ready_timeout=3.0,
            )
            assert result.readiness is not None
            assert not result.readiness.ready
            child_pid = _read_marker(marker)
            assert _wait_until(lambda: _pid_gone(child_pid), timeout=15.0), (
                f"nested child {child_pid} survived readiness-timeout teardown"
            )
            assert _wait_until(lambda: not _fixture_proc_pids(), timeout=15.0)
        finally:
            _reap_all_fixtures(marker)

    def test_phase_timeout_removes_nested_tree(self, tmp_path):
        marker = tmp_path / "child_pid.txt"
        try:
            result = run_verify(
                tmp_path,
                Recipe(name="x", test=[_tree_command(marker)]),
                phase_timeout=3.0,
                skip_start=True,
            )
            phase = result.phases[0]
            assert phase.timed_out
            assert not phase.teardown_failed
            child_pid = _read_marker(marker)
            assert _wait_until(lambda: _pid_gone(child_pid), timeout=15.0), (
                f"nested child {child_pid} survived phase-timeout teardown"
            )
            assert _wait_until(lambda: not _fixture_proc_pids(), timeout=15.0)
        finally:
            _reap_all_fixtures(marker)

    def test_preexisting_listener_preserved_and_fail_closed_before_spawn(
        self, tmp_path
    ):
        port = _free_port()
        marker = tmp_path / "child_pid.txt"

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(204)
                self.end_headers()

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        try:
            foreign_pid = _listener_pids(port)
            assert foreign_pid, "test server must be listening"

            result = run_verify(
                tmp_path,
                Recipe(name="x", start=_serve_command(marker, port), port=port),
                phases=("start",),
                ready_timeout=10,
            )
            assert result.readiness is not None
            assert not result.readiness.ready
            assert "already in use" in (result.readiness.error or "")
            assert not marker.exists(), "start command must never have run"
            assert _listener_pids(port) == foreign_pid, (
                "the foreign listener must be preserved untouched"
            )

            import urllib.request

            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=5
            ) as resp:
                assert resp.status == 204, "foreign server must still serve"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            _reap_all_fixtures(marker)

    def test_twenty_cycles_leave_zero_descendants_and_listeners(self, tmp_path):
        used_ports: list[int] = []
        try:
            for cycle in range(20):
                port = _free_port()
                deadline = time.monotonic() + 5.0
                while (_listener_pids(port) or 8090 <= port <= 8115) and time.monotonic() < deadline:
                    port = _free_port()
                used_ports.append(port)
                marker = tmp_path / f"cycle_{cycle:02d}_child.txt"

                result = run_verify(
                    tmp_path,
                    Recipe(name="x", start=_serve_command(marker, port), port=port),
                    phases=("start",),
                    ready_timeout=30,
                )
                assert result.readiness is not None and result.readiness.ready, (
                    f"cycle {cycle}: server never became ready — "
                    f"{result.readiness.error}"
                )
                child_pid = _read_marker(marker)
                assert _wait_until(lambda: _pid_gone(child_pid), timeout=15.0), (
                    f"cycle {cycle}: nested child {child_pid} leaked"
                )
                assert _wait_until(lambda: not _fixture_proc_pids(), timeout=15.0), (
                    f"cycle {cycle}: fixture tree leaked"
                )
                assert _listener_pids(port) == [], (
                    f"cycle {cycle}: port {port} still listening after teardown"
                )

            leaked_listeners = [p for p in used_ports if _listener_pids(p)]
            assert leaked_listeners == []
            assert _fixture_proc_pids() == []
        finally:
            _reap_all_fixtures(tmp_path)
