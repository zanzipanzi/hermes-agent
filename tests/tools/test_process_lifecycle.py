"""Behavior tests for tools/process_lifecycle.py — the shared, PID-reuse-safe
owned-process tree-termination seam.

Safety contract mirrored from the implementation plan: every test spawns only
processes it owns (direct children of this pytest process), records PID plus
kernel start time before signalling, and cleans up in ``finally`` using the
recorded identity. If ownership cannot be proven, the fixture is deliberately
leaked and the test fails rather than signalling an unrelated process. Nothing
here may kill by executable name.
"""

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.process_lifecycle import (
    ProcessIdentity,
    TerminationStatus,
    capture_process_identity,
    terminate_process_tree,
)

_IS_WINDOWS = sys.platform == "win32"


def _pid_exists(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)


def _wait_until(predicate, timeout: float, interval: float = 0.05):
    """Bounded poll; returns True once ``predicate`` holds, False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _spawn_sleep(seconds: float = 60) -> subprocess.Popen:
    """Spawn a plain sleep process owned by this test."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def _spawn_nested_tree(marker_path: Path) -> subprocess.Popen:
    """Spawn ``parent python -> child python``; parent records the child PID."""
    parent_src = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        "with open(sys.argv[1], 'w') as fh:\n"
        "    fh.write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", parent_src, str(marker_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def _read_child_pid(marker_path: Path, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker_path.exists():
            text = marker_path.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        time.sleep(0.05)
    pytest.fail(f"nested parent never wrote its child PID to {marker_path}")


def _cleanup(*procs: subprocess.Popen) -> None:
    """Terminate only fixtures this test spawned, by recorded identity."""
    for proc in procs:
        if proc.poll() is not None:
            continue
        terminate_process_tree(capture_process_identity(proc.pid))
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class TestCaptureIdentity:
    def test_capture_identity_records_pid_and_start_time(self):
        proc = _spawn_sleep()
        try:
            identity = capture_process_identity(proc.pid)
            assert identity.pid == proc.pid
            assert identity.start_time is not None
            assert isinstance(identity.start_time, int)
            assert identity.start_time > 0
        finally:
            _cleanup(proc)

    def test_capture_identity_dead_pid_has_no_start_time(self):
        proc = _spawn_sleep(seconds=0.2)
        proc.wait(timeout=5)
        identity = capture_process_identity(proc.pid)
        assert identity.pid == proc.pid
        assert identity.start_time is None


class TestTerminateProcessTree:
    def test_terminate_refuses_recycled_pid_identity(self):
        proc = _spawn_sleep()
        try:
            real = capture_process_identity(proc.pid)
            assert real.start_time is not None, "platform lacks start-time support"
            forged = ProcessIdentity(pid=proc.pid, start_time=real.start_time + 1_000_000)
            result = terminate_process_tree(forged)
            assert result.status is TerminationStatus.refused_identity_mismatch
            # The guard refused to signal: the fixture must still be alive.
            assert _pid_exists(proc.pid)
            assert proc.poll() is None
        finally:
            _cleanup(proc)

    def test_terminate_is_idempotent_after_process_exit(self):
        proc = _spawn_sleep()
        try:
            identity = capture_process_identity(proc.pid)
            first = terminate_process_tree(identity)
            assert first.status is TerminationStatus.terminated
            assert _wait_until(lambda: not _pid_exists(proc.pid), timeout=10.0)
            second = terminate_process_tree(identity)
            assert second.status is TerminationStatus.already_exited
            third = terminate_process_tree(identity)
            assert third.status is TerminationStatus.already_exited
        finally:
            _cleanup(proc)

    def test_terminate_tree_removes_nested_child_processes(self, tmp_path):
        parent = _spawn_nested_tree(tmp_path / "child_pid.txt")
        try:
            child_pid = _read_child_pid(tmp_path / "child_pid.txt")
            assert _pid_exists(parent.pid) and _pid_exists(child_pid)
            identity = capture_process_identity(parent.pid)
            result = terminate_process_tree(identity)
            assert result.status is TerminationStatus.terminated
            assert _wait_until(lambda: not _pid_exists(parent.pid), timeout=10.0)
            assert _wait_until(lambda: not _pid_exists(child_pid), timeout=10.0), (
                "nested child survived tree termination"
            )
        finally:
            _cleanup(parent)


@pytest.mark.skipif(not _IS_WINDOWS, reason="Windows taskkill /T /F branch")
class TestWindowsTaskkillBranch:
    def test_windows_terminate_uses_taskkill_tree_force(self):
        from tools import process_lifecycle as pl

        proc = _spawn_sleep()
        captured = {}

        def recording_run(args, **kwargs):
            captured["args"] = list(args)
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0, stderr="", stdout="")

        try:
            identity = capture_process_identity(proc.pid)
            with patch.object(pl.subprocess, "run", side_effect=recording_run):
                result = terminate_process_tree(identity)
            assert result.status is TerminationStatus.terminated
            assert captured["args"][:2] == ["taskkill", "/PID"]
            assert captured["args"][2] == str(proc.pid)
            assert captured["args"][3:] == ["/T", "/F"], (
                "tree + force flags are required to reach descendants"
            )
        finally:
            # The patched run was a no-op kill, so clean up for real here.
            _cleanup(proc)


@pytest.mark.skipif(_IS_WINDOWS, reason="POSIX SIGTERM-children-first branch")
class TestPosixTreeBranch:
    def test_posix_terminate_signals_children_before_parent_and_escalates(self):
        import psutil

        from tools import process_lifecycle as pl

        events = []

        class _FakeProc:
            def __init__(self, pid, children=()):
                self.pid = pid
                self._children = list(children)

            def children(self, recursive=False):
                assert recursive is True
                return list(self._children)

            def terminate(self):
                events.append(("term", self.pid))

            def kill(self):
                events.append(("kill", self.pid))

            def is_running(self):
                return True

            def status(self):
                return psutil.STATUS_RUNNING

        child_a = _FakeProc(101)
        child_b = _FakeProc(102)
        parent = _FakeProc(12345, children=[child_a, child_b])

        with patch.object(psutil, "Process", return_value=parent), \
             patch.object(pl, "_proc_alive", side_effect=lambda p: True):
            result = terminate_process_tree(
                ProcessIdentity(pid=12345, start_time=None), grace_seconds=0.05
            )

        assert result.status is TerminationStatus.terminated
        assert events[:3] == [("term", 101), ("term", 102), ("term", 12345)], (
            "children must be signalled before the parent"
        )
        assert ("kill", 101) in events and ("kill", 102) in events, (
            "survivors must be escalated after the grace window"
        )
        assert events.index(("term", 12345)) < events.index(("kill", 101)), (
            "escalation happens only after every tree member got SIGTERM"
        )
