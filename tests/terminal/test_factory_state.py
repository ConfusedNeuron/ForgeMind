"""
tests/terminal/test_factory_state.py
FactoryState — initialization, update_from_agent_result, sensor ring buffers,
RUL history, window building, comms log, and reset.

Run:  pytest tests/terminal/test_factory_state.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from terminal.factory_state import FactoryState, MachineState


@pytest.fixture
def state():
    return FactoryState()


def _agent_result(machine_id=4, status="OFFLINE", rul=12.0,
                  available_time=0.0, capacity_pct=80.0,
                  machine_req=18.594, breakeven_risk=True):
    return {
        "valid": True,
        "machine_statuses": [
            {"id": mid,
             "status": "ONLINE" if mid != machine_id else status,
             "rul": 999.0 if mid != machine_id else rul,
             "available_time": 8.0 if mid != machine_id else available_time}
            for mid in range(1, 6)
        ],
        "capacity_report": {
            "capacity_pct":   capacity_pct,
            "machine_req":    machine_req,
            "breakeven_risk": breakeven_risk,
        },
    }


# ── Initialization ────────────────────────────────────────────────────────────

class TestInitialization:
    def test_five_machines_initialized(self, state):
        assert len(state.machines) == 5

    def test_all_machines_start_online(self, state):
        assert all(m.status == "ONLINE" for m in state.machines.values())

    def test_all_machines_start_rul_999(self, state):
        assert all(m.rul == 999.0 for m in state.machines.values())

    def test_all_machines_available_time_equals_base_time(self, state):
        assert all(m.available_time == m.base_time for m in state.machines.values())

    def test_sensor_history_has_18_lists(self, state):
        assert len(state.sensor_history) == 18

    def test_all_sensor_histories_start_empty(self, state):
        assert all(len(h) == 0 for h in state.sensor_history)

    def test_per_machine_sensor_history_has_5_machines(self, state):
        assert set(state.per_machine_sensor_history.keys()) == {1, 2, 3, 4, 5}

    def test_rul_history_has_5_machines(self, state):
        assert set(state.rul_history.keys()) == {1, 2, 3, 4, 5}

    def test_capacity_starts_at_100(self, state):
        assert state.capacity_pct == 100.0

    def test_breakeven_risk_starts_false(self, state):
        assert state.breakeven_risk is False

    def test_active_machine_id_is_1(self, state):
        assert state.active_machine_id == 1

    def test_machine_names_are_correct(self, state):
        expected = {1: "CNC-Alpha", 2: "CNC-Beta", 3: "Press-Gamma",
                    4: "Lathe-Delta", 5: "Mill-Epsilon"}
        for mid, name in expected.items():
            assert state.machines[mid].name == name


# ── update_from_agent_result ──────────────────────────────────────────────────

class TestUpdateFromAgentResult:
    def test_offline_machine_status_updated(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4, status="OFFLINE"))
        assert state.machines[4].status == "OFFLINE"

    def test_rul_updated_on_faulted_machine(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4, rul=12.0))
        assert state.machines[4].rul == 12.0

    def test_available_time_updated(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4, available_time=0.0))
        assert state.machines[4].available_time == 0.0

    def test_unaffected_machines_stay_online(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4))
        for mid in [1, 2, 3, 5]:
            assert state.machines[mid].status == "ONLINE"

    def test_capacity_pct_updated(self, state):
        state.update_from_agent_result(_agent_result(capacity_pct=80.0))
        assert state.capacity_pct == 80.0

    def test_breakeven_risk_updated(self, state):
        state.update_from_agent_result(_agent_result(breakeven_risk=True))
        assert state.breakeven_risk is True

    def test_invalid_result_does_not_update(self, state):
        state.update_from_agent_result({"valid": False, "machine_statuses": [], "capacity_report": {}})
        assert state.capacity_pct == 100.0
        assert all(m.status == "ONLINE" for m in state.machines.values())

    def test_rul_history_appended(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4, rul=12.0))
        assert 12.0 in state.rul_history[4]

    def test_rul_history_capped_at_max(self, state):
        for rul in range(15):
            state.update_from_agent_result(_agent_result(machine_id=4, rul=float(rul)))
        assert len(state.rul_history[4]) <= state.MAX_RUL_HISTORY


# ── Sensor ring buffers ───────────────────────────────────────────────────────

class TestSensorRingBuffers:
    def test_push_sensor_reading_fills_all_18_channels(self, state):
        vals = np.ones(18, dtype=np.float32) * 0.5
        state.push_sensor_reading(vals)
        assert all(len(h) == 1 for h in state.sensor_history)

    def test_ring_buffer_caps_at_history_length(self, state):
        vals = np.ones(18, dtype=np.float32) * 0.5
        for _ in range(state.HISTORY_LENGTH + 10):
            state.push_sensor_reading(vals)
        assert all(len(h) == state.HISTORY_LENGTH for h in state.sensor_history)

    def test_push_machine_sensor_reading_updates_per_machine_buffer(self, state):
        vals = np.ones(18, dtype=np.float32) * 0.7
        state.push_machine_sensor_reading(1, vals)
        assert any(len(h) > 0 for h in state.per_machine_sensor_history[1])

    def test_active_machine_reading_mirrors_to_shared_buffer(self, state):
        state.active_machine_id = 2
        vals = np.ones(18, dtype=np.float32) * 0.6
        state.push_machine_sensor_reading(2, vals)
        assert any(len(h) > 0 for h in state.sensor_history)

    def test_non_active_machine_reading_does_not_update_shared_buffer(self, state):
        state.active_machine_id = 1
        vals = np.ones(18, dtype=np.float32) * 0.6
        state.push_machine_sensor_reading(3, vals)
        assert all(len(h) == 0 for h in state.sensor_history)

    def test_short_sensor_vector_padded_to_18(self, state):
        vals = np.ones(10, dtype=np.float32) * 0.5
        state.push_machine_sensor_reading(1, vals)   # Should not raise

    def test_empty_sensor_vector_silently_skipped(self, state):
        vals = np.array([], dtype=np.float32)
        state.push_machine_sensor_reading(1, vals)   # Should not raise


# ── Window building ───────────────────────────────────────────────────────────

class TestWindowBuilding:
    def test_get_sensor_window_shape_is_50x18(self, state):
        w = state.get_sensor_window()
        assert w.shape == (50, 18)

    def test_get_sensor_window_dtype_is_float32(self, state):
        w = state.get_sensor_window()
        assert w.dtype == np.float32

    def test_get_sensor_window_empty_history_uses_random_baseline(self, state):
        w = state.get_sensor_window()
        # With empty history, values should be in [0.3, 0.7]
        assert float(w.min()) >= 0.3
        assert float(w.max()) <= 0.7

    def test_get_sensor_window_with_full_history_uses_real_data(self, state):
        vals = np.linspace(0.1, 0.9, 18, dtype=np.float32)
        for _ in range(55):
            state.push_sensor_reading(vals)
        w = state.get_sensor_window()
        # All values should be close to the pushed vals
        assert np.allclose(w, vals, atol=0.01)

    def test_get_machine_sensor_window_shape_is_50x18(self, state):
        w = state.get_machine_sensor_window(3)
        assert w.shape == (50, 18)

    def test_get_machine_sensor_window_falls_back_when_empty(self, state):
        """Machine 5 has no data → falls back to shared get_sensor_window."""
        vals = np.ones(18, dtype=np.float32) * 0.5
        for _ in range(55):
            state.push_sensor_reading(vals)
        w = state.get_machine_sensor_window(5)
        assert w.shape == (50, 18)


# ── Comms log ─────────────────────────────────────────────────────────────────

class TestCommsLog:
    def test_add_log_entry_appends(self, state):
        state.add_log_entry("System", "Test message")
        assert len(state.comms_log) == 1

    def test_log_entry_has_required_keys(self, state):
        state.add_log_entry("Diagnostic Agent", "Spike detected")
        entry = state.comms_log[0]
        assert "time" in entry and "agent" in entry and "message" in entry

    def test_log_entry_stores_correct_agent_and_message(self, state):
        state.add_log_entry("Floor Manager", "Halt production")
        assert state.comms_log[-1]["agent"] == "Floor Manager"
        assert state.comms_log[-1]["message"] == "Halt production"

    def test_log_capped_at_max_entries(self, state):
        for i in range(state.MAX_LOG_ENTRIES + 20):
            state.add_log_entry("System", f"msg {i}")
        assert len(state.comms_log) == state.MAX_LOG_ENTRIES


# ── reset_all ─────────────────────────────────────────────────────────────────

class TestResetAll:
    def test_all_machines_back_to_online(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4, status="OFFLINE"))
        state.reset_all()
        assert all(m.status == "ONLINE" for m in state.machines.values())

    def test_all_rul_reset_to_999(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4, rul=12.0))
        state.reset_all()
        assert all(m.rul == 999.0 for m in state.machines.values())

    def test_capacity_resets_to_100(self, state):
        state.update_from_agent_result(_agent_result(capacity_pct=60.0))
        state.reset_all()
        assert state.capacity_pct == 100.0

    def test_breakeven_risk_resets_to_false(self, state):
        state.update_from_agent_result(_agent_result(breakeven_risk=True))
        state.reset_all()
        assert state.breakeven_risk is False

    def test_rul_histories_cleared(self, state):
        state.update_from_agent_result(_agent_result(machine_id=4, rul=12.0))
        state.reset_all()
        assert all(len(v) == 0 for v in state.rul_history.values())

    def test_sensor_histories_cleared(self, state):
        vals = np.ones(18, dtype=np.float32)
        state.push_sensor_reading(vals)
        state.reset_all()
        assert all(len(h) == 0 for h in state.sensor_history)

    def test_maintenance_schedule_cleared(self, state):
        state.maintenance_schedule = [{"rank": 1}]
        state.reset_all()
        assert state.maintenance_schedule == []

    def test_shift_health_banner_resets_to_nominal(self, state):
        state.shift_health = ("CRITICAL", "red")
        state.reset_all()
        assert "NOMINAL" in state.shift_health[0]
    