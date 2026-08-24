"""Non-destructive context-health advisory (plan Lane B, task B3).

One advisory when current-window occupancy first crosses a configurable
band below the effective compression threshold. It must use real
``last_prompt_tokens`` (never cumulative lifetime totals), deduplicate
until occupancy falls back below the band, and never delete history,
auto-run /new, or force compression.
"""

from __future__ import annotations

from types import SimpleNamespace

import run_agent
from agent.context_compressor import ContextCompressor
from agent.turn_context import build_turn_context


def _bare_agent(**comp_attrs) -> run_agent.AIAgent:
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    comp = SimpleNamespace(
        threshold_tokens=100_000,
        context_length=120_000,
        last_prompt_tokens=0,
        compression_count=0,
    )
    for k, v in comp_attrs.items():
        setattr(comp, k, v)
    agent.context_compressor = comp
    agent.context_advisory_ratio = 0.85
    agent._context_advisory_fired = False
    warnings: list[str] = []
    agent._emit_warning = warnings.append
    agent._warnings = warnings
    return agent


class TestContextHealthAdvisory:
    def test_fires_once_when_crossing_band(self):
        agent = _bare_agent(last_prompt_tokens=88_000)
        agent._maybe_advise_context_health()
        agent._maybe_advise_context_health()
        agent._maybe_advise_context_health()
        assert len(agent._warnings) == 1
        text = agent._warnings[0]
        assert "88,000" in text
        assert "100,000" in text  # effective threshold reported
        assert "/context" in text and "/compress" in text
        assert "checkpoint" in text.lower()

    def test_no_fire_below_band(self):
        agent = _bare_agent(last_prompt_tokens=50_000)
        agent._maybe_advise_context_health()
        assert agent._warnings == []

    def test_rearms_after_falling_below_band(self):
        agent = _bare_agent(last_prompt_tokens=90_000)
        agent._maybe_advise_context_health()
        assert len(agent._warnings) == 1
        # A compression brought the window back down — advisory re-arms.
        agent.context_compressor.last_prompt_tokens = 30_000
        agent._maybe_advise_context_health()
        assert len(agent._warnings) == 1
        # ...and the next re-cross fires again.
        agent.context_compressor.last_prompt_tokens = 95_000
        agent._maybe_advise_context_health()
        assert len(agent._warnings) == 2

    def test_never_fires_from_lifetime_totals(self):
        agent = _bare_agent(last_prompt_tokens=0)
        # Cumulative lifetime usage is huge, but the current window is
        # unknown/empty — an advisory here would be a false alarm.
        agent.session_total_tokens = 2_000_000
        agent._maybe_advise_context_health()
        assert agent._warnings == []

    def test_silent_without_threshold_or_compressor(self):
        agent = _bare_agent(last_prompt_tokens=88_000, threshold_tokens=0)
        agent._maybe_advise_context_health()
        assert agent._warnings == []
        agent2 = _bare_agent(last_prompt_tokens=88_000)
        agent2.context_compressor = None
        agent2._maybe_advise_context_health()
        assert agent2._warnings == []

    def test_configurable_ratio(self):
        agent = _bare_agent(last_prompt_tokens=60_000)
        agent.context_advisory_ratio = 0.5
        agent._maybe_advise_context_health()
        assert len(agent._warnings) == 1

    def test_default_ratio_is_conservative(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        ratio = DEFAULT_CONFIG["compression"].get("advisory_ratio")
        assert ratio is not None
        assert 0.5 < float(ratio) < 1.0, "advisory must sit below the trigger"

    def test_mentions_autoraised_threshold_when_active(self):
        agent = _bare_agent(last_prompt_tokens=90_000)
        agent._compression_threshold_autoraised = {"from": 0.5, "to": 0.85}
        agent._maybe_advise_context_health()
        assert "autoraised" in agent._warnings[0]


class TestAdvisoryWiredIntoTurnContext:
    def test_build_turn_context_consults_advisory(self):
        """The advisory runs during turn-context build (real occupancy)."""
        import types

        from tests.agent.test_turn_context import _FakeAgent

        agent = _FakeAgent()
        called = []

        def _record():
            called.append(True)

        agent._maybe_advise_context_health = _record
        build_turn_context(
            agent=agent,
            user_message="hello",
            system_message=None,
            conversation_history=None,
            task_id=None,
            stream_callback=None,
            persist_user_message=None,
            restore_or_build_system_prompt=lambda *a, **k: None,
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda s: s,
            summarize_user_message_for_log=lambda s: s,
            set_session_context=lambda _sid: None,
            set_current_write_origin=lambda _o: None,
            ra=lambda: types.SimpleNamespace(
                _set_interrupt=lambda *a, **k: None
            ),
        )
        assert called, "build_turn_context must consult the advisory hook"
