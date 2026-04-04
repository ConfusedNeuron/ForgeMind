"""
tests/unit/test_schemas.py
Pydantic schema validation — SensorSpike and FaultSeverity.
Zero network calls. Run standalone: pytest tests/unit/test_schemas.py
"""

import pytest
from pydantic import ValidationError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.schemas import SensorSpike, FaultSeverity


class TestFaultSeverity:
    def test_valid_values(self):
        assert FaultSeverity.LOW == "LOW"
        assert FaultSeverity.MEDIUM == "MEDIUM"
        assert FaultSeverity.HIGH == "HIGH"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FaultSeverity("CRITICAL")

    def test_case_sensitive(self):
        with pytest.raises(ValueError):
            FaultSeverity("high")


class TestSensorSpikeValid:
    def test_minimal_valid_spike(self, make_spike):
        spike = make_spike()
        assert spike.sensor_id == "Xs4"
        assert spike.spike_value == 0.95
        assert spike.fault_severity == FaultSeverity.HIGH

    def test_all_valid_w_sensors(self, make_spike):
        for sid in ["W0", "W1", "W2", "W3"]:
            assert make_spike(sensor_id=sid).sensor_id == sid

    def test_all_valid_xs_sensors(self, make_spike):
        for i in range(14):
            assert make_spike(sensor_id=f"Xs{i}").sensor_id == f"Xs{i}"

    def test_spike_value_near_bounds(self, make_spike):
        make_spike(spike_value=0.01)
        make_spike(spike_value=0.99)

    def test_single_position_allowed(self, make_spike):
        s = make_spike(affected_window_positions=[49])
        assert len(s.affected_window_positions) == 1

    def test_max_ten_positions(self, make_spike):
        s = make_spike(affected_window_positions=list(range(10)))
        assert len(s.affected_window_positions) == 10

    def test_edge_positions_0_and_49(self, make_spike):
        s = make_spike(affected_window_positions=[0, 49])
        assert 0 in s.affected_window_positions
        assert 49 in s.affected_window_positions

    def test_all_severity_levels(self, make_spike):
        for sev in [FaultSeverity.LOW, FaultSeverity.MEDIUM, FaultSeverity.HIGH]:
            assert make_spike(fault_severity=sev).fault_severity == sev


class TestSensorSpikeInvalid:
    """
    SensorSpike schema uses ge=0.0/le=1.0 and no position constraints —
    those are enforced by _validate_domain, not Pydantic.
    Only test what the schema ACTUALLY rejects.
    """

    def test_spike_value_negative_rejected(self, make_spike):
        with pytest.raises(ValidationError):
            make_spike(spike_value=-0.1)

    def test_spike_value_above_one_rejected(self, make_spike):
        with pytest.raises(ValidationError):
            make_spike(spike_value=1.1)

    def test_spike_value_zero_is_valid_in_schema(self, make_spike):
        # ge=0.0 means 0.0 is allowed by schema; _validate_domain rejects it
        s = make_spike(spike_value=0.0)
        assert s.spike_value == 0.0

    def test_spike_value_one_is_valid_in_schema(self, make_spike):
        # le=1.0 means 1.0 is allowed by schema; domain logic may reject it
        s = make_spike(spike_value=1.0)
        assert s.spike_value == 1.0

    def test_position_bounds_not_enforced_by_schema(self, make_spike):
        # Position validation lives in _validate_domain, not SensorSpike
        s = make_spike(affected_window_positions=[50, -1, 99])
        assert 50 in s.affected_window_positions

    def test_more_than_ten_positions_not_enforced_by_schema(self, make_spike):
        s = make_spike(affected_window_positions=list(range(20)))
        assert len(s.affected_window_positions) == 20

    def test_empty_positions_not_enforced_by_schema(self, make_spike):
        s = make_spike(affected_window_positions=[])
        assert s.affected_window_positions == []
