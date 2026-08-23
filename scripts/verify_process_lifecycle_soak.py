#!/usr/bin/env python
"""Windows lifecycle soak: 20 start/stop verify cycles, zero leaks allowed.

Drives ``run_verify`` start phases against the shared nested-tree fixture
(``cmd.exe -> python parent -> python child`` serving HTTP) on its own
temporary ports (ephemeral; the incident range 8090-8115 is explicitly
excluded), and after every cycle proves: the fixture child PID is gone, no
verifiably-fixture process remains, and the port is free again.

Safety contract (see the session-lifecycle plan): this script never scans
or signals processes by executable name. It records PID + kernel start time
for every fixture process it touches, kills only through the identity-
guarded lifecycle seam, and counts ``unowned_processes_touched`` (which must
stay zero). On the first leak it prints evidence and exits non-zero. Cleanup
in ``finally`` covers only processes whose command line proves ownership
(fixture path or the embedded ``vpt-marker`` tag).

Receipt on success::

    cycles=20
    leaked_processes=0
    leaked_listeners=0
    unowned_processes_touched=0
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.verify.recipes import Recipe  # noqa: E402
from agent.verify.runner import run_verify  # noqa: E402
from tools.process_lifecycle import (  # noqa: E402
    capture_process_identity,
    terminate_process_tree,
)

_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "verify_process_tree.py"
_OWNERSHIP_NEEDLES = ("verify_process_tree.py", "# vpt-marker")
_FORBIDDEN_PORT_RANGE = range(8090, 8116)


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


def _pick_port() -> int:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        port = _free_port()
        if port in _FORBIDDEN_PORT_RANGE:
            continue
        if not _listener_pids(port):
            return port
    raise SystemExit("soak: could not find a free temporary port")


def _fixture_procs() -> list[tuple[int, str]]:
    """(pid, cmdline) for every process whose cmdline proves fixture ownership."""
    import psutil

    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except Exception:
            continue
        if any(needle in cmdline for needle in _OWNERSHIP_NEEDLES):
            found.append((proc.info["pid"], cmdline))
    return found


def _pid_gone(pid: int) -> bool:
    import psutil

    return not psutil.pid_exists(pid)


def _wait_until(predicate, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _cleanup_owned(unowned_counter: list[int]) -> None:
    """Kill only verifiably-owned fixtures, recording PID + start time first."""
    for pid, _cmdline in _fixture_procs():
        identity = capture_process_identity(pid)
        print(f"  cleanup: terminating fixture pid={pid} start_time={identity.start_time}")
        result = terminate_process_tree(identity)
        if result.status.value != "terminated" and result.status.value != "already_exited":
            unowned_counter[0] += 1
            print(f"  cleanup: WARNING pid {pid} -> {result.status.value}: {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument(
        "--receipt", type=Path, default=None, help="optional path for the JSON receipt"
    )
    args = parser.parse_args()

    leaked_processes = 0
    leaked_listeners = 0
    unowned_processes_touched = [0]
    workdir = Path(tempfile.mkdtemp(prefix="verify_soak_"))
    evidence: list[dict] = []
    used_ports: list[int] = []

    print(f"soak: {args.cycles} cycles, fixture={_FIXTURE}")
    try:
        for cycle in range(args.cycles):
            port = _pick_port()
            used_ports.append(port)
            marker = workdir / f"cycle_{cycle:02d}_child.txt"
            command = f'"{sys.executable}" "{_FIXTURE}" "{marker}" --serve {port}'
            result = run_verify(
                workdir,
                Recipe(name="soak", start=command, port=port),
                phases=("start",),
                ready_timeout=args.ready_timeout,
            )
            readiness = result.readiness
            entry = {
                "cycle": cycle,
                "port": port,
                "ready": bool(readiness.ready) if readiness else False,
                "error": readiness.error if readiness else "no readiness result",
            }

            if readiness is None or not readiness.ready:
                entry["verdict"] = "READY_FAIL"
                evidence.append(entry)
                print(f"cycle {cycle}: NOT READY on port {port}: {entry['error']}")
                _cleanup_owned(unowned_processes_touched)
                return _fail(evidence, args, leaked_processes, leaked_listeners,
                             unowned_processes_touched[0])

            child_pid = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not marker.exists():
                time.sleep(0.05)
            if marker.exists():
                child_pid = int(marker.read_text(encoding="utf-8").strip())
                entry["child_pid"] = child_pid
                entry["child_start_time"] = capture_process_identity(
                    child_pid
                ).start_time
                if not _wait_until(lambda: _pid_gone(child_pid), timeout=15.0):
                    entry["verdict"] = "CHILD_LEAK"
                    leaked_processes += 1
                    evidence.append(entry)
                    print(f"cycle {cycle}: LEAK child pid {child_pid} still alive")
                    _cleanup_owned(unowned_processes_touched)
                    return _fail(evidence, args, leaked_processes, leaked_listeners,
                                 unowned_processes_touched[0])

            if not _wait_until(lambda: not _fixture_procs(), timeout=15.0):
                survivors = _fixture_procs()
                entry["verdict"] = "TREE_LEAK"
                entry["survivors"] = [
                    {"pid": pid, "cmdline": cl[:120]} for pid, cl in survivors
                ]
                leaked_processes += len(survivors)
                evidence.append(entry)
                print(f"cycle {cycle}: LEAK fixture survivors: {survivors}")
                _cleanup_owned(unowned_processes_touched)
                return _fail(evidence, args, leaked_processes, leaked_listeners,
                             unowned_processes_touched[0])

            if _listener_pids(port):
                entry["verdict"] = "LISTENER_LEAK"
                entry["listener_pids"] = _listener_pids(port)
                leaked_listeners += 1
                evidence.append(entry)
                print(f"cycle {cycle}: LEAK port {port} still listening")
                _cleanup_owned(unowned_processes_touched)
                return _fail(evidence, args, leaked_processes, leaked_listeners,
                             unowned_processes_touched[0])

            entry["verdict"] = "OK"
            evidence.append(entry)
            print(f"cycle {cycle}: ok (port {port}, child {child_pid} reaped)")

        # Cross-cycle final check: nothing listening on any used port.
        for port in used_ports:
            if _listener_pids(port):
                leaked_listeners += 1
                print(f"final: port {port} still listening")
        if _fixture_procs():
            leaked_processes += len(_fixture_procs())
            print(f"final: fixture survivors: {_fixture_procs()}")
    finally:
        _cleanup_owned(unowned_processes_touched)
        if args.receipt:
            args.receipt.write_text(
                json.dumps(
                    {
                        "cycles": args.cycles,
                        "leaked_processes": leaked_processes,
                        "leaked_listeners": leaked_listeners,
                        "unowned_processes_touched": unowned_processes_touched[0],
                        "evidence": evidence,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        shutil.rmtree(workdir, ignore_errors=True)

    print(
        f"cycles={args.cycles}\n"
        f"leaked_processes={leaked_processes}\n"
        f"leaked_listeners={leaked_listeners}\n"
        f"unowned_processes_touched={unowned_processes_touched[0]}"
    )
    if leaked_processes or leaked_listeners or unowned_processes_touched[0]:
        return 1
    return 0


def _fail(evidence, args, leaked_processes, leaked_listeners, unowned) -> int:
    print(
        f"cycles={args.cycles}\n"
        f"leaked_processes={leaked_processes}\n"
        f"leaked_listeners={leaked_listeners}\n"
        f"unowned_processes_touched={unowned}"
    )
    print("first-leak evidence (JSON):")
    print(json.dumps(evidence[-1], indent=2, default=str))
    return 1


if __name__ == "__main__":
    sys.exit(main())
