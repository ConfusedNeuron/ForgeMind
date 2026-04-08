"""
tests/unit/test_diagnostic_agent.py
DiagnosticAgent (Agent 1) — sensor mapping, domain validation,
spike injection, fallback keyword matching.
No Gemini calls. See tests/live/test_gemini_live.py for live API tests.

Run from project ROOT:  pytest tests/unit/test_diagnostic_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from pydantic import ValidationError

from agents.schemas import SensorSpike, FaultSeverity
from agents.diagnostic_agent import (
    translate_fault_to_tensor,
    SENSOR_TO_COL,
    VALID_SENSORS,
    _validate_domain,
    _inject_spike,
    _get_fallback,
)


# ── Sensor column mapping ─────────────────────────────────────────────────────

class TestSensorToColMapping:
    @pytest.mark.parametrize("sensor_id, expected_col", [
        ("W0", 0), ("W1", 1), ("W2", 2), ("W3", 3),
        ("Xs0", 4), ("Xs1", 5), ("Xs2", 6), ("Xs3", 7),
        ("Xs4", 8), ("Xs5", 9), ("Xs6", 10), ("Xs7", 11),
        ("Xs8", 12), ("Xs9", 13), ("Xs10", 14), ("Xs11", 15),
        ("Xs12", 16), ("Xs13", 17),
    ])
    def test_sensor_maps_to_correct_column(self, sensor_id, expected_col):
        assert SENSOR_TO_COL[sensor_id] == expected_col

    def test_no_missing_sensors(self):
        expected = {f"W{i}" for i in range(4)} | {f"Xs{i}" for i in range(14)}
        assert expected - set(SENSOR_TO_COL.keys()) == set()

    def test_no_extra_sensors(self):
        expected = {f"W{i}" for i in range(4)} | {f"Xs{i}" for i in range(14)}
        assert set(SENSOR_TO_COL.keys()) - expected == set()

    def test_total_sensor_count_is_18(self):
        assert len(SENSOR_TO_COL) == 18


# ── Domain validation ─────────────────────────────────────────────────────────

class TestDomainValidation:
    def test_valid_spike_passes(self, make_spike):
        ok, msg = _validate_domain(make_spike())
        assert ok is True

    @pytest.mark.parametrize("bad_id", [
        "Xs14", "Xs17", "W4", "W10", "Sensor1", "xs4", "XS4",
    ])
    def test_invalid_sensor_id_rejected(self, make_spike, bad_id):
        ok, _ = _validate_domain(make_spike(sensor_id=bad_id))
        assert ok is False

    def test_position_50_rejected(self, make_spike):
        ok, _ = _validate_domain(make_spike(affected_window_positions=[50]))
        assert ok is False

    def test_position_negative_rejected(self, make_spike):
        ok, _ = _validate_domain(make_spike(affected_window_positions=[-1]))
        assert ok is False

    def test_edge_positions_0_and_49_accepted(self, make_spike):
        ok, _ = _validate_domain(make_spike(affected_window_positions=[0, 49]))
        assert ok is True

    def test_empty_positions_rejected(self, make_spike):
        ok, _ = _validate_domain(make_spike(affected_window_positions=[]))
        assert ok is False

    def test_eleven_positions_rejected(self, make_spike):
        ok, _ = _validate_domain(make_spike(affected_window_positions=list(range(11))))
        assert ok is False

    def test_exactly_ten_positions_accepted(self, make_spike):
        ok, _ = _validate_domain(make_spike(affected_window_positions=list(range(10))))
        assert ok is True


# ── Spike injection ───────────────────────────────────────────────────────────

class TestSpikeInjection:
    def test_base_window_not_mutated(self, base_window, make_spike):
        original = base_window.copy()
        _inject_spike(base_window, make_spike())
        assert np.array_equal(base_window, original)

    def test_output_shape_preserved(self, base_window, make_spike):
        out = _inject_spike(base_window, make_spike())
        assert out.shape == (50, 18)

    def test_output_dtype_is_float32(self, base_window, make_spike):
        out = _inject_spike(base_window, make_spike())
        assert out.dtype == np.float32

    def test_spike_value_written_to_correct_cells(self, base_window, make_spike):
        """Primary sensor column should contain ramped values in raw units.
        The ramp converts spike_value from [0,1] to raw physical units,
        so we check that the column was modified (not the exact spike_value)."""
        spike = make_spike(sensor_id="Xs4", spike_value=0.95,
                           affected_window_positions=[47, 48, 49])
        out = _inject_spike(base_window, spike)
        col = SENSOR_TO_COL["Xs4"]   # col 8
        # Ramp pattern: last row should differ most from the base
        assert out[49, col] != base_window[49, col]
        # Values should be monotonically increasing (ramp up)
        assert out[49, col] >= out[0, col]

    def test_non_spiked_columns_mostly_unchanged(self, base_window, make_spike):
        """Columns NOT in the primary or correlated sensor group stay unchanged."""
        spike = make_spike(sensor_id="Xs4", affected_window_positions=[47, 48, 49])
        out = _inject_spike(base_window, spike)
        # Xs4 correlates with Xs2 (col 6) and Xs3 (col 7)
        # So columns outside {6, 7, 8} should be untouched
        unaffected_cols = [c for c in range(18) if c not in {6, 7, 8}]
        for col in unaffected_cols:
            assert np.array_equal(out[:, col], base_window[:, col]), \
                f"Column {col} was unexpectedly modified"

    def test_adjacent_column_untouched(self, base_window, make_spike):
        spike = make_spike(sensor_id="Xs4", affected_window_positions=[49])
        out = _inject_spike(base_window, spike)
        col = SENSOR_TO_COL["Xs4"]
        # col+1 = 9 (Xs5) is NOT in Xs4's correlation group, so should be unchanged
        assert np.array_equal(out[:, col + 1], base_window[:, col + 1])

    def test_different_sensor_lands_in_different_column(self, base_window, make_spike):
        spike_vib = make_spike(sensor_id="Xs7", spike_value=0.88,
                               affected_window_positions=[49])
        out = _inject_spike(base_window, spike_vib)
        col_vib = SENSOR_TO_COL["Xs7"]   # col 11
        # Primary sensor column should be modified
        assert out[49, col_vib] != base_window[49, col_vib]


# ── Fallback keyword matching ─────────────────────────────────────────────────

class TestFallbackKeywordMatching:
    @pytest.mark.parametrize("text, expected_sensor", [
        ("bearing temperature surge on Machine 4",  "Xs4"),
        ("temperature overheat detected",           "Xs4"),
        ("pressure spike in hydraulic line",        "Xs2"),
        ("vibration and shaking on CNC-Alpha",      "Xs7"),
        ("coolant leak near the pump",              "Xs12"),
        ("RPM fluctuation on motor",                "Xs10"),
        ("speed drop on drive belt",                "Xs10"),
        ("overload condition on machine",           "W0"),
        ("completely unrecognized random text here", "Xs4"),  # default
    ])
    def test_fallback_maps_text_to_correct_sensor(self, text, expected_sensor):
        spike = _get_fallback(text)
        assert spike.sensor_id == expected_sensor, \
            f"'{text}' → expected {expected_sensor}, got {spike.sensor_id}"

    def test_all_fallback_spikes_tagged(self):
        texts = [
            "bearing temperature surge", "pressure spike", "vibration shaking",
            "coolant leak", "RPM fluctuation", "completely unrecognized text",
        ]
        for text in texts:
            spike = _get_fallback(text)
            assert "[FALLBACK]" in spike.plain_english_summary, \
                f"Missing [FALLBACK] tag for: {text!r}"

    def test_fallback_returns_sensor_spike_instance(self):
        spike = _get_fallback("bearing overheat")
        assert isinstance(spike, SensorSpike)
