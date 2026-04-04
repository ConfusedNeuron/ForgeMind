"""
tests/integration/test_integration.py
DiagnosticAgent + CapacityAgent integration.
Uses a dummy oracle — no real DL model or Gemini calls needed.
Tests the full Agent1 → DL → Agent2 pipeline contract.

Run from project ROOT:  pytest tests/integration/test_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from agents.diagnostic_agent import translate_fault_to_tensor
from agents.capacity_agent import update_capacity, reset_all, get_factory_snapshot


@pytest.fixture(autouse=True)
def fresh_factory():
    reset_all()
    yield
    reset_all()


# ── Full Agent1 → oracle → Agent2 chain ──────────────────────────────────────

class TestDiagnosticToCapacityPipeline:
    """End-to-end: fault text → tensor + spike → RUL → capacity report."""

    @pytest.mark.parametrize("fault_text, expected_sensor, machine_id, expected_status", [
        ("bearing temperature surge on Machine 4", "Xs4",  4, "OFFLINE"),
        ("pressure spike in hydraulic line",       "Xs2",  2, "OFFLINE"),
        ("vibration anomaly on CNC-Alpha",         "Xs7",  1, "DEGRADED"),
    ])
    def test_fault_produces_expected_pipeline_output(
        self, base_window, dummy_oracle, fault_text, expected_sensor, machine_id, expected_status
    ):
        tensor, spike_dict, used_fallback = translate_fault_to_tensor(base_window, fault_text)
        rul = dummy_oracle(tensor)
        report = update_capacity(machine_id, rul)

        assert spike_dict["sensor_id"] == expected_sensor
        assert report["status"] == expected_status
        assert isinstance(report["capacity_pct"], float)
        assert 0.0 <= report["capacity_pct"] <= 100.0

    def test_tensor_shape_preserved_through_injection(self, base_window, dummy_oracle):
        tensor, _, _ = translate_fault_to_tensor(
            base_window, "bearing temperature surge on Machine 4"
        )
        assert tensor.shape == (50, 18)
        assert tensor.dtype == np.float32

    def test_spike_value_lands_in_correct_column(self, base_window, dummy_oracle):
        from agents.diagnostic_agent import SENSOR_TO_COL
        tensor, spike_dict, _ = translate_fault_to_tensor(
            base_window, "pressure spike in hydraulic line"
        )
        col = SENSOR_TO_COL[spike_dict["sensor_id"]]
        positions = spike_dict["affected_window_positions"]
        for pos in positions:
            assert abs(tensor[pos, col] - spike_dict["spike_value"]) < 1e-5

    def test_stacked_faults_reduce_capacity(self, base_window, dummy_oracle):
        """Each subsequent HIGH fault on a new machine reduces total capacity."""
        faults = [
            ("bearing temperature surge on Machine 4", 4),
            ("pressure spike in hydraulic line",       2),
        ]
        prev_cap = 100.0
        for fault_text, machine_id in faults:
            tensor, _, _ = translate_fault_to_tensor(base_window, fault_text)
            rul = dummy_oracle(tensor)
            report = update_capacity(machine_id, rul)
            assert report["capacity_pct"] <= prev_cap
            prev_cap = report["capacity_pct"]

    def test_reset_restores_full_capacity_after_faults(self, base_window, dummy_oracle):
        for mid, text in [(4, "bearing surge"), (3, "pressure spike")]:
            tensor, _, _ = translate_fault_to_tensor(base_window, text)
            update_capacity(mid, dummy_oracle(tensor))

        reset_all()
        snap = get_factory_snapshot()
        assert snap["capacity_pct"] == 100.0
        assert snap["breakeven_risk"] is False
