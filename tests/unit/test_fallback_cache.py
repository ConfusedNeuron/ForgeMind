"""
tests/unit/test_fallback_cache.py
FallbackCache — offline scenario structure, keyword matching,
floor manager response template placeholders.
Zero network calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from agents.fallback_cache import CACHED_SCENARIOS, match_scenario
from agents.schemas import SensorSpike


class TestCachedScenarioStructure:
    REQUIRED_KEYS = {"trigger_keywords", "diagnostic_spike", "floor_manager_response"}

    def test_all_scenarios_have_required_keys(self):
        for name, scenario in CACHED_SCENARIOS.items():
            missing = self.REQUIRED_KEYS - set(scenario.keys())
            assert not missing, f"Scenario '{name}' missing keys: {missing}"

    def test_all_diagnostic_spikes_are_sensor_spike_instances(self):
        for name, scenario in CACHED_SCENARIOS.items():
            assert isinstance(scenario["diagnostic_spike"], SensorSpike), \
                f"Scenario '{name}' diagnostic_spike is not a SensorSpike"

    def test_all_trigger_keyword_lists_are_non_empty(self):
        """general_fault is the catch-all default and intentionally has no keywords."""
        CATCH_ALL_SCENARIOS = {"general_fault"}
        for name, scenario in CACHED_SCENARIOS.items():
            if name in CATCH_ALL_SCENARIOS:
                continue
            assert len(scenario["trigger_keywords"]) > 0, \
                f"Non-catch-all scenario '{name}' has no trigger keywords"

    def test_catch_all_scenario_exists(self):
        """general_fault catch-all must exist so unknown faults always get a response."""
        assert "general_fault" in CACHED_SCENARIOS

    def test_floor_manager_responses_contain_format_placeholders(self):
        """Responses must contain {machine_name}, {machine_id}, {capacity_pct}."""
        required_placeholders = {"{machine_name}", "{machine_id}", "{capacity_pct}"}
        for name, scenario in CACHED_SCENARIOS.items():
            resp = scenario["floor_manager_response"]
            missing = {p for p in required_placeholders if p not in resp}
            assert not missing, \
                f"Scenario '{name}' response missing placeholders: {missing}"

    def test_floor_manager_responses_start_with_prefix(self):
        for name, scenario in CACHED_SCENARIOS.items():
            resp = scenario["floor_manager_response"]
            assert resp.startswith("[Floor Manager]"), \
                f"Scenario '{name}' response missing [Floor Manager] prefix"


class TestMatchScenario:
    @pytest.mark.parametrize("text, expected_key", [
        ("bearing overheat on Machine 1",       "bearing_overheat"),
        ("temperature surge on spindle",         "bearing_overheat"),
        ("pressure spike in hydraulic line",     "pressure_surge"),
        ("vibration and shaking on CNC-Alpha",   "vibration_fault"),
    ])
    def test_keyword_matches_correct_scenario(self, text, expected_key):
        scenario = match_scenario(text)
        assert scenario is not None, f"No scenario matched for: {text!r}"
        # The matched scenario's spike sensor should match expected
        bearing_sensors = {"Xs4"}
        pressure_sensors = {"Xs2"}
        vibration_sensors = {"Xs7"}
        mapping = {
            "bearing_overheat": bearing_sensors,
            "pressure_surge":   pressure_sensors,
            "vibration_fault":  vibration_sensors,
        }
        if expected_key in mapping:
            assert scenario["diagnostic_spike"].sensor_id in mapping[expected_key]

    def test_match_returns_none_or_default_for_unknown_text(self):
        result = match_scenario("completely unrelated random text about nothing")
        # Either returns None or a default — should not raise
        assert result is None or isinstance(result, dict)

    def test_match_returns_dict(self):
        result = match_scenario("bearing temperature critical")
        assert isinstance(result, dict)
