"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from gateway import kanban_watchers as watchers
from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


@pytest.mark.asyncio
async def test_gateway_dispatch_passes_worker_runtime_config(monkeypatch):
    """The shipped runtime key reaches dispatch_once on the gateway path."""
    from hermes_cli import kanban_db as kb

    captured = {}

    class DummyGateway(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = True

        def _release_kanban_dispatcher_lock(self):
            return None

    gateway = DummyGateway()

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
                "worker_max_runtime_seconds": 77,
            }
        },
    )
    monkeypatch.setattr(watchers, "_acquire_singleton_lock", lambda _path: (None, "unavailable"))
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kb, "list_boards", lambda **_kwargs: [{"slug": "default"}])

    class FakeConn:
        def close(self):
            return None

    monkeypatch.setattr(kb, "connect", lambda **_kwargs: FakeConn())
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda _conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda _conn: False)

    def fake_dispatch_once(_conn, **kwargs):
        captured.update(kwargs)
        gateway._running = False
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)

    async def direct_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    await gateway._kanban_dispatcher_watcher()

    assert captured["default_max_runtime_seconds"] == 77


