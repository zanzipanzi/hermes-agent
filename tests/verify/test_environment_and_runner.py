"""Tests for the verify environment manifest and the smoke runner."""

import http.server
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.verify.environment import (
    load_manifest,
    load_or_detect,
    manifest_path,
    save_manifest,
)
from agent.verify.recipes import Recipe
from agent.verify.runner import run_verify


class TestManifest:
    def test_roundtrip(self, tmp_path):
        recipe = Recipe(
            name="Next.js",
            kind="nextjs",
            bootstrap=["npm install"],
            build=["npm run build"],
            test=["npm test"],
            start="npm run dev",
            port=3000,
            readiness_path="/health",
        )
        path = save_manifest(tmp_path, recipe)
        assert path == manifest_path(tmp_path)
        payload = json.loads(path.read_text())
        assert payload["version"] == 1
        assert "updatedAt" in payload
        assert load_manifest(tmp_path) == recipe

    def test_missing_file(self, tmp_path):
        assert load_manifest(tmp_path) is None

    def test_malformed_json_tolerated(self, tmp_path):
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{oops", encoding="utf-8")
        assert load_manifest(tmp_path) is None

    def test_non_dict_tolerated(self, tmp_path):
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_manifest(tmp_path) is None

    def test_bare_recipe_shape_accepted(self, tmp_path):
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"name": "Custom", "test": ["true"]}), encoding="utf-8")
        recipe = load_manifest(tmp_path)
        assert recipe.name == "Custom"
        assert recipe.test == ["true"]

    def test_manifest_wins_over_detection(self, tmp_path):
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        save_manifest(tmp_path, Recipe(name="Custom", kind="custom", test=["true"]))
        recipe, source = load_or_detect(tmp_path)
        assert source == "manifest"
        assert recipe.name == "Custom"

    def test_detection_fallback(self, tmp_path):
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        recipe, source = load_or_detect(tmp_path)
        assert source == "detected"
        assert recipe.kind == "go"


class TestRunner:
    def test_all_phases_pass(self, tmp_path):
        recipe = Recipe(name="x", bootstrap=["true"], build=["true"], test=["true"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert result.ok
        assert [p.phase for p in result.phases] == ["bootstrap", "build", "test"]
        assert all(p.exit_code == 0 for p in result.phases)
        assert all(p.duration >= 0 for p in result.phases)

    def test_failure_stops_pipeline(self, tmp_path):
        recipe = Recipe(name="x", build=["false"], test=["true"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert not result.ok
        assert len(result.phases) == 1
        assert result.phases[0].exit_code == 1

    def test_output_captured(self, tmp_path):
        recipe = Recipe(name="x", test=["echo hello-verify"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert "hello-verify" in result.phases[0].output_tail

    def test_phase_selection(self, tmp_path):
        recipe = Recipe(name="x", bootstrap=["true"], build=["true"], test=["true"])
        result = run_verify(tmp_path, recipe, phases=("test",))
        assert [p.phase for p in result.phases] == ["test"]

    def test_phase_timeout(self, tmp_path):
        recipe = Recipe(name="x", test=["sleep 5"])
        result = run_verify(tmp_path, recipe, phase_timeout=0.3, skip_start=True)
        assert not result.ok
        assert result.phases[0].timed_out
        assert result.phases[0].exit_code is None

    def test_commands_run_in_project_root(self, tmp_path):
        (tmp_path / "marker.txt").write_text("here", encoding="utf-8")
        recipe = Recipe(name="x", test=["cat marker.txt"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert result.ok

    def test_result_to_dict(self, tmp_path):
        recipe = Recipe(name="x", test=["true"])
        payload = run_verify(tmp_path, recipe, skip_start=True).to_dict()
        assert payload["ok"] is True
        assert payload["recipe"] == "x"
        assert payload["phases"][0]["command"] == "true"
        assert payload["readiness"] is None


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestReadiness:
    def test_readiness_against_live_server(self, tmp_path):
        port = _free_port()
        recipe = Recipe(
            name="x",
            start=f"python3 -m http.server {port} --bind 127.0.0.1",
            port=port,
        )
        result = run_verify(tmp_path, recipe, phases=("start",), ready_timeout=15)
        assert result.readiness is not None
        assert result.readiness.ready
        assert result.readiness.status_code == 200
        assert result.readiness.url == f"http://127.0.0.1:{port}/"
        assert result.ok

    def test_readiness_timeout_when_nothing_listens(self, tmp_path):
        port = _free_port()
        recipe = Recipe(name="x", start="sleep 30", port=port)
        result = run_verify(tmp_path, recipe, phases=("start",), ready_timeout=1.5)
        assert result.readiness is not None
        assert not result.readiness.ready
        assert not result.ok

    def test_skip_start(self, tmp_path):
        recipe = Recipe(name="x", test=["true"], start="sleep 30", port=1)
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert result.readiness is None
        assert result.ok

    def test_start_skipped_after_phase_failure(self, tmp_path):
        recipe = Recipe(name="x", test=["false"], start="sleep 30", port=1)
        result = run_verify(tmp_path, recipe, stop_on_failure=False)
        assert result.readiness is None
        assert not result.ok

    def test_port_override(self, tmp_path):
        port = _free_port()

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
            recipe = Recipe(name="x", start="sleep 30", port=1)
            result = run_verify(
                tmp_path, recipe, phases=("start",), ready_timeout=10, port_override=port
            )
            assert result.readiness.ready
            assert result.readiness.status_code == 204
        finally:
            server.shutdown()
            thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Owned-tree phase execution — a phase timeout must reap the whole tree.
# Every helper here kills only processes this file spawned: parents are
# identified by the unique marker path in their command line, nested children
# by the PID recorded in the marker file, and every kill goes through the
# PID+start-time guarded lifecycle seam. If ownership can't be proven, the
# process is left alone.
# ---------------------------------------------------------------------------


def _fixture_tree_command(marker: Path, token: str = "FIXTURE-READY") -> str:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "verify_process_tree.py"
    return f'"{sys.executable}" "{fixture}" "{marker}" --token {token}'


def _read_pid_marker(marker: Path, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            text = marker.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        time.sleep(0.05)
    pytest.fail(f"fixture parent never wrote its child PID to {marker}")


def _pid_gone(pid: int) -> bool:
    import psutil

    return not psutil.pid_exists(pid)


def _wait_until_pid_gone(pid: int, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _pid_gone(pid):
            return True
        time.sleep(interval)
    return _pid_gone(pid)


def _reap_fixture_processes(marker: Path) -> None:
    """Best-effort teardown of only this test's fixture tree (see block doc)."""
    import psutil
    from tools.process_lifecycle import (
        capture_process_identity,
        terminate_process_tree,
    )

    needle = str(marker)
    targets: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except Exception:
            continue
        if needle in cmdline:
            targets.append(proc.info["pid"])
    if marker.exists():
        try:
            targets.append(int(marker.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            pass
    for pid in targets:
        try:
            if psutil.pid_exists(pid):
                terminate_process_tree(capture_process_identity(pid))
        except Exception:
            pass


class TestPhaseTimeoutTreeReaping:
    def test_phase_timeout_kills_nested_child_tree(self, tmp_path):
        marker = tmp_path / "child_pid.txt"
        recipe = Recipe(name="x", test=[_fixture_tree_command(marker)])
        try:
            result = run_verify(
                tmp_path, recipe, phase_timeout=2.0, skip_start=True
            )
            phase = result.phases[0]
            assert phase.timed_out
            child_pid = _read_pid_marker(marker)
            assert _wait_until_pid_gone(child_pid, timeout=15.0), (
                f"nested child pid {child_pid} survived the phase timeout"
            )
        finally:
            _reap_fixture_processes(marker)

    def test_phase_timeout_preserves_output_tail(self, tmp_path):
        marker = tmp_path / "child_pid.txt"
        token = "FIXTURE-TOKEN-7f3a"
        recipe = Recipe(
            name="x", test=[_fixture_tree_command(marker, token=token)]
        )
        try:
            result = run_verify(
                tmp_path, recipe, phase_timeout=2.0, skip_start=True
            )
            phase = result.phases[0]
            assert phase.timed_out
            assert token in phase.output_tail
        finally:
            _reap_fixture_processes(marker)
