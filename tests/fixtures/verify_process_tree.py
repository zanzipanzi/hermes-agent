"""Nested-process fixture for verify lifecycle tests.

Usage::

    python verify_process_tree.py MARKER_FILE [--serve PORT] [--token TOKEN]

Spawns a nested child (so the caller's shell command produces a real
``shell -> this parent -> child`` tree), prints ``TOKEN`` to stdout, writes
the child's PID to ``MARKER_FILE``, then sleeps forever.

``--serve PORT`` makes the *child* serve HTTP on 127.0.0.1:PORT instead of
sleeping, for readiness/listener-ownership tests.
"""

import http.server
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    marker = Path(sys.argv[1])
    argv = sys.argv[2:]
    port = None
    token = "FIXTURE-READY"
    chatty_kb = 0
    if "--serve" in argv:
        port = int(argv[argv.index("--serve") + 1])
    if "--token" in argv:
        token = argv[argv.index("--token") + 1]
    if "--chatty" in argv:
        chatty_kb = int(argv[argv.index("--chatty") + 1])

    # Chatty prelude: write more than any platform pipe buffer BEFORE the
    # serving child exists. With no concurrent reader the pipe fills and this
    # write blocks forever — the server never starts.
    if chatty_kb:
        line = "x" * 120 + "\n"
        for _ in range(chatty_kb * 1024 // 121):
            sys.stdout.write(line)
        sys.stdout.flush()

    if port is not None:
        child_code = (
            "# vpt-marker\n"
            "import http.server\n"
            f"http.server.HTTPServer(('127.0.0.1', {port}), "
            "http.server.BaseHTTPRequestHandler).serve_forever()\n"
        )
    else:
        child_code = (
            "# vpt-marker\n"
            "import time\n"
            "[time.sleep(1) for _ in iter(int, 1)]\n"
        )

    child = subprocess.Popen([sys.executable, "-c", child_code])
    marker.write_text(str(child.pid), encoding="utf-8")
    print(f"{token}", flush=True)
    time.sleep(600)


if __name__ == "__main__":
    main()
