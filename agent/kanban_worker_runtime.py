"""Strict Kanban worker API-cycle fence and durable progress receipts.

This module is inert outside dispatcher-owned worker processes. A worker must
still own the exact task/run/claim tuple before every outbound model cycle, and
an accepted cycle is persisted before the call starts. That ordering is the
last-resort fence when reclaim has released the claim but process-tree teardown
is delayed by the OS.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Optional


@dataclass(frozen=True)
class WorkerCycleDecision:
    allowed: bool
    worker: bool
    reason: Optional[str] = None
    cycle: int = 0
    total_tokens: int = 0
    max_api_turns: Optional[int] = None
    max_total_tokens: Optional[int] = None


def _positive_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _worker_context() -> Optional[tuple[str, int, str, int, int]]:
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return None
    claim_lock = (os.environ.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    run_id = _positive_env("HERMES_KANBAN_RUN_ID")
    max_api_turns = _positive_env("HERMES_KANBAN_MAX_API_TURNS")
    max_total_tokens = _positive_env("HERMES_KANBAN_MAX_TOTAL_TOKENS")
    if not claim_lock or run_id is None or max_api_turns is None or max_total_tokens is None:
        return (task_id, run_id or 0, claim_lock, max_api_turns or 0, max_total_tokens or 0)
    return task_id, run_id, claim_lock, max_api_turns, max_total_tokens


def _append_log(*, cycle: int, stage: str, total_tokens: int, reason: str = "") -> None:
    raw_path = (os.environ.get("HERMES_KANBAN_WORKER_LOG") or "").strip()
    if not raw_path:
        return
    line = (
        f"[{int(time.time())}] worker_cycle cycle={cycle} stage={stage} "
        f"total_tokens={total_tokens}"
    )
    if reason:
        line += f" reason={reason}"
    line += "\n"
    try:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        # The board event is the canonical receipt. A log append failure is
        # surfaced by begin_api_cycle's DB path rather than crashing cleanup.
        pass


def _active_claim(row, *, run_id: int, claim_lock: str) -> bool:
    return bool(
        row
        and row["status"] == "running"
        and row["current_run_id"] == run_id
        and row["claim_lock"] == claim_lock
    )


def begin_api_cycle(*, session_id: str, total_tokens: int) -> WorkerCycleDecision:
    """Fence and durably admit one outbound worker API cycle.

    The event count is the cumulative run budget, so goal-loop continuations
    and any other in-process turn reset cannot restore the allowance.
    """
    context = _worker_context()
    total = max(0, int(total_tokens or 0))
    if context is None:
        return WorkerCycleDecision(allowed=True, worker=False, total_tokens=total)

    task_id, run_id, claim_lock, max_api_turns, max_total_tokens = context
    if not run_id or not claim_lock or not max_api_turns or not max_total_tokens:
        return WorkerCycleDecision(
            allowed=False,
            worker=True,
            reason="invalid_worker_context",
            total_tokens=total,
        )

    from hermes_cli import kanban_db as kb

    conn = None
    try:
        conn = kb.connect()
        with kb.write_txn(conn):
            row = conn.execute(
                "SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if not _active_claim(row, run_id=run_id, claim_lock=claim_lock):
                decision = WorkerCycleDecision(
                    allowed=False,
                    worker=True,
                    reason="claim_lost",
                    total_tokens=total,
                    max_api_turns=max_api_turns,
                    max_total_tokens=max_total_tokens,
                )
            elif total >= max_total_tokens:
                decision = WorkerCycleDecision(
                    allowed=False,
                    worker=True,
                    reason="token_budget_exhausted",
                    total_tokens=total,
                    max_api_turns=max_api_turns,
                    max_total_tokens=max_total_tokens,
                )
            else:
                count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM task_events "
                        "WHERE task_id = ? AND run_id = ? AND kind = 'worker_cycle_started'",
                        (task_id, run_id),
                    ).fetchone()[0]
                )
                if count >= max_api_turns:
                    decision = WorkerCycleDecision(
                        allowed=False,
                        worker=True,
                        reason="api_turn_budget_exhausted",
                        cycle=count,
                        total_tokens=total,
                        max_api_turns=max_api_turns,
                        max_total_tokens=max_total_tokens,
                    )
                else:
                    cycle = count + 1
                    now = int(time.time())
                    ttl = kb._resolve_claim_ttl_seconds()
                    conn.execute(
                        "UPDATE tasks SET last_heartbeat_at = ?, claim_expires = ? "
                        "WHERE id = ? AND current_run_id = ? AND claim_lock = ?",
                        (now, now + ttl, task_id, run_id, claim_lock),
                    )
                    conn.execute(
                        "UPDATE task_runs SET last_heartbeat_at = ?, claim_expires = ? "
                        "WHERE id = ? AND claim_lock = ?",
                        (now, now + ttl, run_id, claim_lock),
                    )
                    kb._append_event(
                        conn,
                        task_id,
                        "worker_cycle_started",
                        {
                            "cycle": cycle,
                            "session_id": session_id,
                            "total_tokens": total,
                            "max_api_turns": max_api_turns,
                            "max_total_tokens": max_total_tokens,
                        },
                        run_id=run_id,
                    )
                    decision = WorkerCycleDecision(
                        allowed=True,
                        worker=True,
                        cycle=cycle,
                        total_tokens=total,
                        max_api_turns=max_api_turns,
                        max_total_tokens=max_total_tokens,
                    )
    except Exception:
        decision = WorkerCycleDecision(
            allowed=False,
            worker=True,
            reason="progress_persistence_failed",
            total_tokens=total,
            max_api_turns=max_api_turns,
            max_total_tokens=max_total_tokens,
        )
    finally:
        if conn is not None:
            conn.close()

    _append_log(
        cycle=decision.cycle,
        stage="started" if decision.allowed else "blocked",
        total_tokens=total,
        reason=decision.reason or "",
    )
    if decision.reason in {"api_turn_budget_exhausted", "token_budget_exhausted"}:
        failure_conn = None
        try:
            failure_conn = kb.connect()
            kb._record_task_failure(
                failure_conn,
                task_id,
                (
                    f"Kanban worker {decision.reason}: cycle={decision.cycle}, "
                    f"total_tokens={total}, max_api_turns={max_api_turns}, "
                    f"max_total_tokens={max_total_tokens}"
                ),
                outcome="timed_out",
                release_claim=True,
                end_run=True,
                expected_run_id=run_id,
                event_payload_extra={
                    "reason": decision.reason,
                    "cycle": decision.cycle,
                    "total_tokens": total,
                    "max_api_turns": max_api_turns,
                    "max_total_tokens": max_total_tokens,
                },
            )
        except Exception:
            pass
        finally:
            if failure_conn is not None:
                failure_conn.close()
    return decision


def complete_api_cycle(
    *,
    session_id: str,
    cycle: int,
    total_tokens: int,
    input_tokens: int,
    output_tokens: int,
) -> bool:
    """Persist completion metrics iff this process still owns the exact claim."""
    context = _worker_context()
    if context is None:
        return True
    task_id, run_id, claim_lock, _max_api_turns, _max_total_tokens = context
    total = max(0, int(total_tokens or 0))
    conn = None
    try:
        from hermes_cli import kanban_db as kb

        conn = kb.connect()
        with kb.write_txn(conn):
            row = conn.execute(
                "SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if not _active_claim(row, run_id=run_id, claim_lock=claim_lock):
                return False
            kb._append_event(
                conn,
                task_id,
                "worker_cycle_completed",
                {
                    "cycle": int(cycle),
                    "session_id": session_id,
                    "total_tokens": total,
                    "input_tokens": max(0, int(input_tokens or 0)),
                    "output_tokens": max(0, int(output_tokens or 0)),
                },
                run_id=run_id,
            )
    except Exception:
        return False
    finally:
        if conn is not None:
            conn.close()

    _append_log(cycle=int(cycle), stage="completed", total_tokens=total)
    return True


__all__ = ["WorkerCycleDecision", "begin_api_cycle", "complete_api_cycle"]
