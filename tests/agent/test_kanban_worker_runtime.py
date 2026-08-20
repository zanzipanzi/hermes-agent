"""Bounded, durable Kanban worker runtime contracts.

These tests exercise behavior rather than source shape: every outbound worker
cycle is fenced by the active claim and explicit API/token budgets, and each
accepted cycle leaves a durable board event plus a direct worker-log record.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture()
def claimed_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="bounded worker", assignee="default")
        assert kb.claim_task(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.current_run_id is not None
        assert task.claim_lock

    log_path = tmp_path / "worker.log"
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock)
    monkeypatch.setenv("HERMES_KANBAN_WORKER_LOG", str(log_path))
    monkeypatch.setenv("HERMES_KANBAN_MAX_API_TURNS", "2")
    monkeypatch.setenv("HERMES_KANBAN_MAX_TOTAL_TOKENS", "100")
    yield task_id, task.current_run_id, log_path


def test_read_only_cycles_cannot_bypass_api_turn_budget(claimed_worker):
    """A broad read-only audit gets the same hard cycle cap as mutating work."""
    from agent.kanban_worker_runtime import begin_api_cycle, complete_api_cycle

    task_id, run_id, _ = claimed_worker
    first = begin_api_cycle(session_id="worker-session", total_tokens=0)
    assert first.allowed and first.cycle == 1
    assert complete_api_cycle(
        session_id="worker-session",
        cycle=first.cycle,
        total_tokens=10,
        input_tokens=8,
        output_tokens=2,
    )

    second = begin_api_cycle(session_id="worker-session", total_tokens=10)
    assert second.allowed and second.cycle == 2
    assert complete_api_cycle(
        session_id="worker-session",
        cycle=second.cycle,
        total_tokens=20,
        input_tokens=16,
        output_tokens=4,
    )

    blocked = begin_api_cycle(session_id="worker-session", total_tokens=20)
    assert not blocked.allowed
    assert blocked.reason == "api_turn_budget_exhausted"

    with kb.connect_closing() as conn:
        events = [
            event for event in kb.list_events(conn, task_id)
            if event.run_id == run_id and event.kind == "worker_cycle_started"
        ]
    assert len(events) == 2


def test_total_token_budget_blocks_next_outbound_cycle(claimed_worker):
    from agent.kanban_worker_runtime import begin_api_cycle

    decision = begin_api_cycle(session_id="worker-session", total_tokens=100)

    assert not decision.allowed
    assert decision.reason == "token_budget_exhausted"
    assert decision.total_tokens == 100
    assert decision.max_total_tokens == 100


def test_cycle_progress_is_durable_in_board_and_worker_log(claimed_worker):
    from agent.kanban_worker_runtime import begin_api_cycle, complete_api_cycle

    task_id, run_id, log_path = claimed_worker
    decision = begin_api_cycle(session_id="worker-session", total_tokens=0)
    assert decision.allowed
    assert complete_api_cycle(
        session_id="worker-session",
        cycle=decision.cycle,
        total_tokens=42,
        input_tokens=40,
        output_tokens=2,
    )

    with kb.connect_closing() as conn:
        events = [
            event for event in kb.list_events(conn, task_id)
            if event.run_id == run_id and event.kind.startswith("worker_cycle_")
        ]
    assert [event.kind for event in events] == [
        "worker_cycle_started",
        "worker_cycle_completed",
    ]
    assert events[-1].payload == {
        "cycle": 1,
        "session_id": "worker-session",
        "total_tokens": 42,
        "input_tokens": 40,
        "output_tokens": 2,
    }

    log_text = log_path.read_text(encoding="utf-8")
    assert "cycle=1 stage=started" in log_text
    assert "cycle=1 stage=completed" in log_text
    assert "total_tokens=42" in log_text


def test_reclaimed_claim_fences_all_later_api_cycles(claimed_worker):
    """Even if process-tree termination is delayed, no later API call is admitted."""
    from agent.kanban_worker_runtime import begin_api_cycle

    task_id, run_id, _ = claimed_worker
    first = begin_api_cycle(session_id="worker-session", total_tokens=0)
    assert first.allowed

    with kb.connect_closing() as conn:
        assert kb.reclaim_task(conn, task_id, reason="test fence")

    blocked = begin_api_cycle(session_id="worker-session", total_tokens=0)
    assert not blocked.allowed
    assert blocked.reason == "claim_lost"

    with kb.connect_closing() as conn:
        starts = [
            event for event in kb.list_events(conn, task_id)
            if event.run_id == run_id and event.kind == "worker_cycle_started"
        ]
    assert len(starts) == 1


def test_non_worker_process_is_not_affected(monkeypatch: pytest.MonkeyPatch):
    from agent.kanban_worker_runtime import begin_api_cycle

    for key in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_MAX_API_TURNS",
        "HERMES_KANBAN_MAX_TOTAL_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)

    decision = begin_api_cycle(session_id="normal-session", total_tokens=10**9)

    assert decision.allowed
    assert decision.worker is False
