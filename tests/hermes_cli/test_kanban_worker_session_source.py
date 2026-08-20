"""Kanban worker runs must not surface as user conversations.

Workers spawn as `hermes chat -q "work kanban task <id>"`, which used to land in
state.db as an untitled `cli` row — the desktop sidebar then rendered one entry
per attempt, labeled with the worker's own prompt.
"""

import os
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    database.close()


def test_worker_spawn_tags_session_source_kanban(monkeypatch, tmp_path):
    """The dispatcher tags the worker's env so its session is a `kanban` row."""
    from hermes_cli import kanban_db as kb

    captured = {}

    class _Proc:
        pid = 4321

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _Proc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    monkeypatch.setattr(kb, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")

    task = kb.Task(
        id="t_b21733fb",
        title="ship it",
        body=None,
        assignee="default",
        status="in_progress",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    kb._default_spawn(task, workspace)

    assert captured["env"]["HERMES_SESSION_SOURCE"] == "kanban"


def test_worker_spawn_has_bounded_runtime_env_and_durable_launch_log(monkeypatch, tmp_path):
    """Bounds and the log path exist before the worker can make its first call."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli import kanban_db as kb

    captured = {}

    class _Proc:
        pid = 4321

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _Proc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    monkeypatch.setattr(kb, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")

    task = kb.Task(
        id="t_b21733fb",
        title="ship it",
        body=None,
        assignee="default",
        status="running",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=0,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock="host:123",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        max_runtime_seconds=60,
    )
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    kb._default_spawn(task, workspace)

    env = captured["env"]
    config = DEFAULT_CONFIG["kanban"]
    assert env["HERMES_KANBAN_MAX_API_TURNS"] == str(config["worker_max_api_turns"])
    assert env["HERMES_KANBAN_MAX_TOTAL_TOKENS"] == str(config["worker_max_total_tokens"])
    assert env["HERMES_KANBAN_WORKER_SESSION_ID"] == kb.worker_session_id(task.id, 7)
    log_path = Path(env["HERMES_KANBAN_WORKER_LOG"])
    assert log_path.exists()
    assert "worker_spawned" in log_path.read_text(encoding="utf-8")


def test_reclaim_finalizes_worker_session_and_live_compression_tip(db):
    """A reclaimed process leaves neither its root nor compression child live."""
    from hermes_cli import kanban_db as kb

    root = kb.worker_session_id("t_b21733fb", 7)
    child = f"{root}_child"
    db.create_session(session_id=root, source="kanban")
    db.end_session(root, "compression")
    db.create_session(session_id=child, source="kanban", parent_session_id=root)
    db.touch_session_activity(child, description="starting API call #2")

    assert kb._finalize_worker_session_lineage(root, reason="kanban_reclaimed")

    rows = {
        row["id"]: row
        for row in db._conn.execute(
            "SELECT id, ended_at, end_reason FROM sessions WHERE id IN (?, ?)",
            (root, child),
        )
    }
    assert rows[root]["end_reason"] == "compression"
    assert rows[child]["ended_at"] is not None
    assert rows[child]["end_reason"] == "kanban_reclaimed"


def test_kanban_rows_stay_out_of_the_session_list(db):
    """A `kanban` row is filtered by the same exclude the sidebar sends."""
    db.create_session(session_id="chat", source="desktop")
    db.append_message(session_id="chat", role="user", content="hey")
    db.create_session(session_id="worker", source="kanban")
    db.append_message(session_id="worker", role="user", content="work kanban task t_b21733fb")

    listed = db.list_sessions_rich(exclude_sources=["cron", "kanban", "subagent", "tool"])

    assert [row["id"] for row in listed] == ["chat"]


def test_retag_reclaims_legacy_worker_rows(db, tmp_path):
    """Rows written before the tag existed are identified by workspace cwd.

    Two rows, not one: the count has to survive ``set_meta`` reusing the same
    cursor, which would otherwise report the meta write's rowcount instead.
    """
    workspaces = tmp_path / "kanban" / "workspaces"
    db.create_session(session_id="legacy", source="cli", cwd=str(workspaces / "t_b21733fb"))
    db.create_session(session_id="legacy2", source="cli", cwd=str(workspaces / "t_c0ffee"))
    db.create_session(session_id="mine", source="cli", cwd=str(tmp_path / "www" / "repo"))

    assert db.retag_kanban_worker_sessions(str(workspaces)) == 2

    sources = {row[0]: row[1] for row in db._conn.execute("SELECT id, source FROM sessions")}
    assert sources == {"legacy": "kanban", "legacy2": "kanban", "mine": "cli"}


def test_retag_runs_once_per_workspaces_root(db, tmp_path):
    """The state_meta gate keeps the retag off every subsequent spawn."""
    workspaces = tmp_path / "kanban" / "workspaces"
    db.create_session(session_id="legacy", source="cli", cwd=str(workspaces / "t_a"))
    db.retag_kanban_worker_sessions(str(workspaces))

    # A row that a *new* worker would never write as `cli`; if the gate leaked,
    # a later sweep would grab it too.
    db.create_session(session_id="later", source="cli", cwd=str(workspaces / "t_b"))

    assert db.retag_kanban_worker_sessions(str(workspaces)) == 0
    row = db._conn.execute("SELECT source FROM sessions WHERE id = 'later'").fetchone()
    assert row[0] == "cli"


def test_retag_gate_is_per_board(db, tmp_path):
    """A second board's workspaces root still gets its own sweep.

    The gate is keyed on the root, so reclaiming board A must not convince the
    dispatcher that board B's legacy rows were already handled.
    """
    board_a = tmp_path / "kanban" / "boards" / "a" / "workspaces"
    board_b = tmp_path / "kanban" / "boards" / "b" / "workspaces"
    db.create_session(session_id="a1", source="cli", cwd=str(board_a / "t_a"))
    db.create_session(session_id="b1", source="cli", cwd=str(board_b / "t_b"))

    assert db.retag_kanban_worker_sessions(str(board_a)) == 1
    assert db.retag_kanban_worker_sessions(str(board_b)) == 1
