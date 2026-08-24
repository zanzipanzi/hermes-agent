"""Effective context-policy visibility (plan Lane B, task B2).

The usage payload, ``/context``, and the TUI must surface what the runtime
is actually doing — engine name, effective vs configured compression
threshold, whether Codex autoraise raised it, and an active compaction with
elapsed time — instead of only raw config.
"""

from __future__ import annotations

import importlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_ctxvis")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
    yield mod


def _agent(**comp_attrs) -> SimpleNamespace:
    comp = SimpleNamespace(
        name="compressor",
        last_prompt_tokens=60000,
        context_length=120000,
        compression_count=3,
        threshold_tokens=60000,
        threshold_percent=0.5,
        _configured_threshold_percent=0.5,
        compaction_started_at=None,
    )
    for k, v in comp_attrs.items():
        setattr(comp, k, v)
    return SimpleNamespace(
        model="test-model",
        context_compressor=comp,
        _compression_threshold_autoraised=None,
        session_input_tokens=10,
        session_output_tokens=20,
        session_reasoning_tokens=0,
        session_prompt_tokens=10,
        session_completion_tokens=20,
        session_total_tokens=30,
        session_api_calls=1,
    )


class TestUsagePayloadContextPolicy:
    def test_usage_reports_engine_and_thresholds(self, server):
        usage = server._get_usage(_agent())
        assert usage["context_engine"] == "compressor"
        assert usage["context_threshold_tokens"] == 60000
        assert usage["context_threshold_percent"] == 50.0
        assert usage["context_threshold_configured_percent"] == 50.0
        assert usage["context_threshold_autoraised"] is False
        assert "compaction_started_at" not in usage
        assert "compaction_elapsed_seconds" not in usage

    def test_usage_reports_configured_vs_effective_and_autoraise(self, server):
        agent = _agent(
            threshold_percent=0.85,
            _configured_threshold_percent=0.5,
            threshold_tokens=102000,
        )
        agent._compression_threshold_autoraised = {
            "from": 0.5, "to": 0.85, "model": "gpt-5.5-codex",
        }
        usage = server._get_usage(agent)
        assert usage["context_threshold_percent"] == 85.0
        assert usage["context_threshold_configured_percent"] == 50.0
        assert usage["context_threshold_autoraised"] is True

    def test_usage_reports_active_compaction_elapsed(self, server):
        agent = _agent(compaction_started_at=time.time() - 12.5)
        usage = server._get_usage(agent)
        assert "compaction_started_at" in usage
        assert 12.0 <= usage["compaction_elapsed_seconds"] <= 14.0

    def test_usage_engine_name_defaults_for_plugin_engines(self, server):
        agent = _agent()
        agent.context_compressor = SimpleNamespace(
            name="lcm",
            last_prompt_tokens=1000,
            context_length=120000,
            compression_count=0,
        )
        usage = server._get_usage(agent)
        assert usage["context_engine"] == "lcm"


class TestContextCommandRendering:
    def _session(self, agent) -> dict:
        return {
            "agent": agent,
            "history": [],
            "history_lock": threading.Lock(),
            "_metadata_mirror": {"model": "test-model", "provider": "openai"},
        }

    def test_context_renders_engine_and_configured_vs_effective(self, server):
        agent = _agent(
            threshold_percent=0.85,
            _configured_threshold_percent=0.5,
            threshold_tokens=102000,
        )
        agent._compression_threshold_autoraised = {"from": 0.5, "to": 0.85}
        out = server._format_live_context_output(self._session(agent))
        assert "Context engine: compressor" in out
        assert "50% configured" in out
        assert "85% effective" in out
        assert "102,000 tokens" in out
        assert "autoraised" in out

    def test_context_renders_plain_threshold_when_equal(self, server):
        out = server._format_live_context_output(self._session(_agent()))
        assert "Context engine: compressor" in out
        assert "50% (60,000 tokens)" in out
        assert "configured" not in out

    def test_context_renders_active_compaction_elapsed(self, server):
        agent = _agent(compaction_started_at=time.time() - 30.0)
        out = server._format_live_context_output(self._session(agent))
        assert "Compaction: running" in out
        assert "elapsed" in out


class TestCompressorCompactionTimestamp:
    """ContextCompressor.compress must bracket compaction with a timestamp."""

    def test_compress_records_start_and_clears_on_exit(self):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor.__new__(ContextCompressor)
        assert getattr(comp, "compaction_started_at", None) is None
        started_at = []

        def _fake_impl(self, *a, **k):
            started_at.append(getattr(self, "compaction_started_at", None))
            return []

        comp._compress_impl = _fake_impl.__get__(comp)
        result = comp.compress([{"role": "user", "content": "x"}])
        assert result == []
        assert started_at and started_at[0] is not None
        assert comp.compaction_started_at is None

    def test_compress_clears_timestamp_on_exception(self):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor.__new__(ContextCompressor)

        def _boom(self, *a, **k):
            raise RuntimeError("summary failed")

        comp._compress_impl = _boom.__get__(comp)
        with pytest.raises(RuntimeError):
            comp.compress([])
        assert comp.compaction_started_at is None
