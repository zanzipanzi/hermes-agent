"""Orphaned async-delegation completions must be parked, never destroyed.

Regression for the observed data loss: a second TUI pane boots,
``restore_undelivered_completions`` rehydrates EVERY pending row (it has no
session filter) into the new process's queue, and the live notification
poller then discards each one it cannot prove ownership of --- with no
re-queue, no durable claim, and no state change. The rows stay
``delivery_state='pending'`` / ``delivery_attempts=0`` forever, recoverable
only by another process start, which repeats the same drop.

The shutdown drain in ``_notification_poller_loop`` already does the right
thing (``deferred.append(evt)`` for ``async_delegation``). The live loop must
match it: park the payload so the session that provably owns it can reclaim
it in-process.

Ordinary (non-delegation) addressed orphans keep the existing drop behavior --
they carry no durable row, so parking them would leak.
"""

import queue
from unittest.mock import patch

import pytest

from tui_gateway.server import (
    _park_orphaned_delegation,
    _reclaim_orphaned_delegations,
    _reset_orphaned_delegations_for_tests,
    _should_park_unowned_notification,
    _sweep_pending_delegations,
)


@pytest.fixture(autouse=True)
def _reset_parking():
    _reset_orphaned_delegations_for_tests()
    yield
    _reset_orphaned_delegations_for_tests()


def _evt(delegation_id="deleg_d81c666f", ui="42323a59", key="20260812_220219_8b4822"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "origin_ui_session_id": ui,
        "session_key": key,
        "parent_session_id": key,
        "is_batch": True,
        "status": "completed",
        "results": [{"task_index": 0, "status": "completed", "summary": "verdict"}],
    }


def _session(key="20260812_220219_8b4822", finalized=False):
    return {"session_key": key, "_finalized": finalized}


class TestShouldPark:
    def test_async_delegation_orphan_is_parked(self):
        assert _should_park_unowned_notification(_evt()) is True

    def test_ordinary_completion_orphan_is_still_dropped(self):
        evt = {"type": "completion", "session_id": "proc_1", "session_key": "gone"}
        assert _should_park_unowned_notification(evt) is False

    def test_watch_match_orphan_is_still_dropped(self):
        evt = {"type": "watch_match", "session_id": "proc_1", "pattern": "x"}
        assert _should_park_unowned_notification(evt) is False

    def test_delegation_without_id_is_not_parked(self):
        """No delegation_id means no durable row to reclaim against."""
        evt = _evt(delegation_id="")
        assert _should_park_unowned_notification(evt) is False


class TestParkAndReclaim:
    def test_owner_reclaims_by_origin_ui_id(self):
        evt = _evt()
        _park_orphaned_delegation(evt)
        got = _reclaim_orphaned_delegations("42323a59", _session())
        assert [e["delegation_id"] for e in got] == ["deleg_d81c666f"]

    def test_owner_reclaims_by_session_key(self):
        evt = _evt()
        _park_orphaned_delegation(evt)
        got = _reclaim_orphaned_delegations("some-other-tab", _session())
        assert [e["delegation_id"] for e in got] == ["deleg_d81c666f"]

    def test_foreign_session_reclaims_nothing(self):
        """The 02:41 pane must not adopt another session's results."""
        _park_orphaned_delegation(_evt())
        assert _reclaim_orphaned_delegations("1a1b32a0", _session(key="other_key")) == []

    def test_finalized_session_reclaims_nothing(self):
        _park_orphaned_delegation(_evt())
        assert _reclaim_orphaned_delegations("42323a59", _session(finalized=True)) == []

    def test_reclaim_removes_from_park_so_it_is_not_delivered_twice(self):
        _park_orphaned_delegation(_evt())
        assert len(_reclaim_orphaned_delegations("42323a59", _session())) == 1
        assert _reclaim_orphaned_delegations("42323a59", _session()) == []

    def test_parking_is_idempotent_by_delegation_id(self):
        """A poller re-seeing the same orphan must not grow the park unboundedly."""
        for _ in range(50):
            _park_orphaned_delegation(_evt())
        assert len(_reclaim_orphaned_delegations("42323a59", _session())) == 1

    def test_all_three_lost_reviews_are_reclaimable_together(self):
        for did in ("deleg_d81c666f", "deleg_f84f852b", "deleg_d3e15cfe"):
            _park_orphaned_delegation(_evt(delegation_id=did))
        got = _reclaim_orphaned_delegations("42323a59", _session())
        assert {e["delegation_id"] for e in got} == {
            "deleg_d81c666f",
            "deleg_f84f852b",
            "deleg_d3e15cfe",
        }

    def test_park_is_bounded(self):
        """A pathological producer must not grow the park without limit."""
        from tui_gateway.server import _ORPHANED_DELEGATION_LIMIT

        for i in range(_ORPHANED_DELEGATION_LIMIT + 25):
            _park_orphaned_delegation(_evt(delegation_id=f"deleg_{i:05d}"))
        got = _reclaim_orphaned_delegations("42323a59", _session())
        assert len(got) <= _ORPHANED_DELEGATION_LIMIT
        # Eviction is oldest-first: the newest delegations survive.
        assert f"deleg_{_ORPHANED_DELEGATION_LIMIT + 24:05d}" in {
            e["delegation_id"] for e in got
        }


class TestPendingSweep:
    """A live session re-derives what it is still owed from the durable rows.

    This is the path that would have delivered the three lost reviews: their
    rows stayed ``pending`` while the commissioning session was alive and
    idle, and nothing ever looked at the DB again after process start.
    """

    def _patch(self, events):
        return patch(
            "tools.async_delegation.pending_completion_events",
            return_value=events,
        )

    def test_owned_pending_completion_is_reenqueued(self):
        q = queue.Queue()
        with self._patch([_evt()]):
            assert _sweep_pending_delegations("42323a59", _session(), q) == 1
        assert q.get_nowait()["delegation_id"] == "deleg_d81c666f"

    def test_foreign_pending_completion_is_not_reenqueued(self):
        q = queue.Queue()
        with self._patch([_evt()]):
            assert _sweep_pending_delegations("1a1b32a0", _session("other"), q) == 0
        assert q.empty()

    def test_finalized_session_sweeps_nothing(self):
        q = queue.Queue()
        with self._patch([_evt()]):
            assert _sweep_pending_delegations("42323a59", _session(finalized=True), q) == 0
        assert q.empty()

    def test_all_three_lost_reviews_are_reenqueued(self):
        q = queue.Queue()
        events = [_evt(delegation_id=d) for d in
                  ("deleg_d81c666f", "deleg_f84f852b", "deleg_d3e15cfe")]
        with self._patch(events):
            assert _sweep_pending_delegations("42323a59", _session(), q) == 3
        assert q.qsize() == 3

    def test_busy_session_is_not_swept(self):
        """Nothing can be injected mid-turn; sweeping anyway piles up copies."""
        q = queue.Queue()
        busy = dict(_session(), running=True)
        with self._patch([_evt()]):
            assert _sweep_pending_delegations("42323a59", busy, q) == 0
        assert q.empty()

    def test_db_failure_is_contained(self):
        """A sweep must never take down the poller thread."""
        q = queue.Queue()
        with patch("tools.async_delegation.pending_completion_events",
                   side_effect=RuntimeError("db locked")):
            assert _sweep_pending_delegations("42323a59", _session(), q) == 0
        assert q.empty()
