"""
tests/live/test_gemini_live.py
Tests that require a live Gemini API key.
Skipped automatically unless FORGЕМIND_RUN_LIVE=1 is set.

Run from project ROOT:
    FORGEMIND_RUN_LIVE=1 pytest tests/live/test_gemini_live.py -v
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

requires_live = pytest.mark.skipif(
    os.environ.get("FORGEMIND_RUN_LIVE") != "1",
    reason="Set FORGEMIND_RUN_LIVE=1 to run live Gemini tests",
)


@requires_live
class TestDiagnosticAgentLiveMappings:
    """Suite 5 from old test_agent1.py — live Gemini sensor mapping check."""

    @pytest.mark.parametrize("fault_text, expected_sensor, allowed_severities", [
        ("bearing temperature surge on Machine 4",  "Xs4",  {"HIGH"}),
        ("pressure spike in hydraulic line",        "Xs2",  {"HIGH"}),
        ("vibration and shaking on CNC-Alpha",      "Xs7",  {"HIGH", "MEDIUM"}),
        ("RPM drop on motor drive",                 "Xs10", {"HIGH", "MEDIUM"}),
        ("coolant leak near pump",                  "Xs12", {"HIGH"}),
    ])
    def test_live_sensor_mapping(self, base_window, fault_text,
                                  expected_sensor, allowed_severities):
        from agents.diagnostic_agent import translate_fault_to_tensor
        _, spike_dict, used_fallback = translate_fault_to_tensor(base_window, fault_text)

        assert spike_dict["sensor_id"] == expected_sensor, \
            f"Got {spike_dict['sensor_id']} (fallback={used_fallback})"
        assert spike_dict["fault_severity"] in allowed_severities
        assert max(spike_dict["affected_window_positions"]) >= 40


@requires_live
class TestFloorManagerLiveDispatch:
    """Suite 7 from old test_agent_loop.py — live Gemini dispatch quality."""

    @pytest.mark.parametrize("fault_text, machine_id", [
        ("bearing temperature surge on Machine 4", 4),
        ("pressure spike in hydraulic line",       2),
    ])
    def test_live_dispatch_quality(self, base_window, dummy_oracle,
                                    fault_text, machine_id):
        import agents.agent_loop as agent_loop_module
        from agents.agent_loop import run_agent_loop, reset_factory
        reset_factory()
        agent_loop_module.OFFLINE_MODE = False

        r = run_agent_loop(fault_text, machine_id, base_window, dummy_oracle)
        assert r["valid"] is True
        assert r["dispatch_orders"].startswith("[Floor Manager]")
        assert len(r["dispatch_orders"]) <= 800
