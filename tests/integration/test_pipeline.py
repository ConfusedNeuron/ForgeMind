"""
tests/integration/test_pipeline.py
Full pipeline: InputGuard → DiagnosticAgent → DLOracle → CapacityAgent → FloorManager
Uses dummy_oracle fixture — no Gemini calls unless FORGЕМIND_RUN_LIVE=1 is set.

Run from project ROOT:
    pytest tests/integration/test_pipeline.py             # offline (default)
    pytest tests/integration/test_pipeline.py --live      # enable Gemini suites
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import agents.agent_loop as agent_loop_module
from agents.agent_loop import run_agent_loop, reset_factory, get_pipeline_status
from agents.capacity_agent import reset_all


@pytest.fixture(autouse=True)
def clean_state():
    reset_factory()
    yield
    reset_factory()


# ── InputGuard integration — bad inputs die before any API call ───────────────

class TestPipelineGuardRejection:
    @pytest.mark.parametrize("text, reason", [
        ("hello world",    "no keywords"),
        ("what's for lunch?", "not a fault"),
        ("x",              "single char"),
        ("a" * 501,        "too long"),
        ("the quick brown fox jumps over the lazy dog and runs away fast", "no fault keywords"),
    ])
    def test_invalid_input_rejected_before_llm(self, text, reason, base_window, dummy_oracle):
        r = run_agent_loop(text, 1, base_window, dummy_oracle)
        assert r["valid"] is False, f"Expected rejection for: {reason}"
        assert r["rejection_reason"]
        assert r["spike"] is None
        assert r["capacity_report"] is None


# ── Return dict contract — every key Team Terminal expects ────────────────────

class TestPipelineReturnSchema:
    TOP_LEVEL_KEYS = {
        "valid", "rejection_reason", "spike", "rul",
        "capacity_report", "dispatch_orders", "machine_statuses",
        "used_fallback", "latency_ms",
    }
    SPIKE_KEYS = {
        "sensor_id", "spike_value", "affected_window_positions",
        "fault_severity", "plain_english_summary",
    }
    CAPACITY_KEYS = {
        "machine_id", "machine_name", "status", "rul",
        "total_T", "total_PD", "machine_req", "capacity_pct", "breakeven_risk",
    }

    @pytest.fixture
    def pipeline_result(self, base_window, dummy_oracle):
        return run_agent_loop(
            "bearing temperature surge on Machine 4", 4, base_window, dummy_oracle
        )

    def test_all_top_level_keys_present(self, pipeline_result):
        missing = self.TOP_LEVEL_KEYS - set(pipeline_result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_spike_sub_keys_present(self, pipeline_result):
        assert pipeline_result["spike"] is not None
        missing = self.SPIKE_KEYS - set(pipeline_result["spike"].keys())
        assert not missing

    def test_capacity_report_sub_keys_present(self, pipeline_result):
        assert pipeline_result["capacity_report"] is not None
        missing = self.CAPACITY_KEYS - set(pipeline_result["capacity_report"].keys())
        assert not missing

    def test_machine_statuses_has_five_entries(self, pipeline_result):
        assert len(pipeline_result["machine_statuses"]) == 5

    def test_field_types(self, pipeline_result):
        assert isinstance(pipeline_result["valid"],          bool)
        assert isinstance(pipeline_result["rul"],            float)
        assert isinstance(pipeline_result["latency_ms"],     float)
        assert isinstance(pipeline_result["used_fallback"],  bool)
        assert isinstance(pipeline_result["dispatch_orders"], str)

    def test_dispatch_orders_starts_with_floor_manager(self, pipeline_result):
        assert pipeline_result["dispatch_orders"].startswith("[Floor Manager]")


# ── Stacked faults accumulate ─────────────────────────────────────────────────

class TestStackedFaults:
    def test_capacity_decreases_with_each_fault(self, base_window, dummy_oracle):
        faults = [
            ("bearing temperature surge on Machine 4", 4),
            ("pressure spike in hydraulic line",       2),
            ("vibration anomaly on CNC-Alpha",         1),
        ]
        prev_cap = 100.0
        for text, machine_id in faults:
            r = run_agent_loop(text, machine_id, base_window, dummy_oracle)
            cap = r["capacity_report"]["capacity_pct"]
            assert cap <= prev_cap
            assert r["capacity_report"]["status"] != "ONLINE"
            prev_cap = cap

    def test_final_capacity_below_100(self, base_window, dummy_oracle):
        for mid, text in [
            (4, "bearing temperature surge on Machine 4"),
            (2, "pressure spike in hydraulic line"),
        ]:
            run_agent_loop(text, mid, base_window, dummy_oracle)
        r = run_agent_loop("vibration on CNC-Alpha", 1, base_window, dummy_oracle)
        assert r["capacity_report"]["capacity_pct"] < 100.0


# ── reset_factory() restores full state ──────────────────────────────────────

class TestFactoryReset:
    def test_all_machines_online_after_reset(self, base_window, dummy_oracle):
        for mid in [1, 2, 3]:
            run_agent_loop("bearing temperature surge", mid, base_window, dummy_oracle)

        reset_factory()
        statuses = get_pipeline_status()
        assert all(m["status"] == "ONLINE" for m in statuses["machine_statuses"])

    def test_offline_mode_cleared_after_reset(self):
        agent_loop_module.OFFLINE_MODE = True
        reset_factory()
        assert agent_loop_module.OFFLINE_MODE is False


# ── Oracle failure — graceful degradation ────────────────────────────────────

class TestOracleFailure:
    def test_pipeline_completes_despite_oracle_crash(self, base_window, crashing_oracle):
        r = run_agent_loop(
            "bearing temperature surge on Machine 4", 4, base_window, crashing_oracle
        )
        assert r["valid"] is True

    def test_rul_defaults_to_25_on_oracle_crash(self, base_window, crashing_oracle):
        r = run_agent_loop(
            "bearing temperature surge on Machine 4", 4, base_window, crashing_oracle
        )
        assert r["rul"] == 25.0

    def test_capacity_report_populated_despite_oracle_crash(self, base_window, crashing_oracle):
        r = run_agent_loop(
            "bearing temperature surge on Machine 4", 4, base_window, crashing_oracle
        )
        assert r["capacity_report"] is not None

    def test_status_is_degraded_on_default_rul(self, base_window, crashing_oracle):
        r = run_agent_loop(
            "bearing temperature surge on Machine 4", 4, base_window, crashing_oracle
        )
        assert r["capacity_report"]["status"] == "DEGRADED"


# ── Forced offline mode — fallback cache takes over ───────────────────────────

class TestOfflineMode:
    @pytest.fixture(autouse=True)
    def force_offline(self):
        agent_loop_module.OFFLINE_MODE = True
        yield
        agent_loop_module.OFFLINE_MODE = False

    @pytest.mark.parametrize("text, machine_id", [
        ("bearing overheat on Machine 1",      1),
        ("pressure surge in hydraulic line",   2),
        ("vibration shaking on Machine 3",     3),
    ])
    def test_offline_pipeline_completes(self, text, machine_id, base_window, dummy_oracle):
        r = run_agent_loop(text, machine_id, base_window, dummy_oracle)
        assert r["valid"] is True
        assert r["used_fallback"] is True
        assert r["spike"] is not None
        assert r["dispatch_orders"].startswith("[Floor Manager]")

    def test_offline_dispatch_contains_live_capacity_numbers(
        self, base_window, dummy_oracle
    ):
        r = run_agent_loop(
            "bearing overheat on Machine 1", 1, base_window, dummy_oracle
        )
        cap_str = str(r["capacity_report"]["capacity_pct"])
        assert cap_str in r["dispatch_orders"], \
            f"Expected live capacity {cap_str} in offline dispatch"
