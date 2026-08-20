"""Conversation-loop enforcement for dispatcher-owned Kanban workers."""

import json
from types import SimpleNamespace

from agent.kanban_worker_runtime import WorkerCycleDecision


def _tool_response():
    call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments={"path": "README.md"}),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, reasoning=None, tool_calls=[call]),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )


class _CountingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        return _tool_response()


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_CountingCompletions())


def _agent(monkeypatch, *, max_iterations=2):
    from run_agent import AIAgent

    client = _FakeClient()
    monkeypatch.setattr("run_agent.OpenAI", lambda **_kwargs: client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *_args, **_kwargs: [{"function": {"name": "read_file"}}],
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda *_args, **_kwargs: json.dumps({"ok": True}),
    )
    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=max_iterations,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    setattr(agent, "_disable_streaming", True)
    return agent, client


def _worker_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:pid:token")


def test_reclaimed_worker_makes_no_model_call(monkeypatch):
    """The claim fence runs before, not after, the outbound provider call."""
    _worker_env(monkeypatch)
    monkeypatch.setattr(
        "agent.kanban_worker_runtime.begin_api_cycle",
        lambda **_kwargs: WorkerCycleDecision(
            worker=True,
            allowed=False,
            reason="claim_lost",
            cycle=1,
        ),
    )
    agent, client = _agent(monkeypatch)

    result = agent.run_conversation("continue auditing")

    assert client.chat.completions.calls == 0
    assert "claim_lost" in result["final_response"]


def test_worker_max_iterations_has_no_unbudgeted_summary_call(monkeypatch):
    """Read-only tool loops stop at the hard cap without a grace API turn."""
    _worker_env(monkeypatch)
    cycles = {"value": 0}

    def _begin(**_kwargs):
        cycles["value"] += 1
        return WorkerCycleDecision(
            worker=True,
            allowed=True,
            reason=None,
            cycle=cycles["value"],
        )

    monkeypatch.setattr("agent.kanban_worker_runtime.begin_api_cycle", _begin)
    monkeypatch.setattr(
        "agent.kanban_worker_runtime.complete_api_cycle",
        lambda **_kwargs: True,
    )
    agent, client = _agent(monkeypatch, max_iterations=2)

    result = agent.run_conversation("audit without mutating anything")

    assert client.chat.completions.calls == 2
    assert "API-turn budget" in result["final_response"]
