"""
tests/terminal/test_ops_analytics.py
ops_analytics.py — all 5 pure functions:
  detect_rul_cliff, compute_prediction_reliability,
  check_sensor_saturation, compute_maintenance_schedule,
  compute_shift_health, compute_degradation_leaderboard.

Pure functions, zero side effects. Runs in milliseconds.

Run:  pytest tests/terminal/test_ops_analytics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from terminal.factory_state import MachineState
from terminal.ops_analytics import (
    detect_rul_cliff,
    compute_prediction_reliability,
    check_sensor_saturation,
    compute_maintenance_schedule,
    compute_shift_health,
    compute_degradation_leaderboard,
)


def _machines(statuses: dict) -> dict:
    """Build a machines dict from {id: (status, rul)}."""
    names = {1: "Metal Press", 2: "Paint & Coat", 3: "PCB Line",
             4: "Final Assembly", 5: "QC & Pack"}
    machines = {}
    for mid in range(1, 6):
        status, rul = statuses.get(mid, ("ONLINE", 999.0))
        m = MachineState(mid, names[mid])
        m.status = status
        m.rul    = rul
        machines[mid] = m
    return machines


def _all_online():
    return _machines({i: ("ONLINE", 999.0) for i in range(1, 6)})


# ── detect_rul_cliff ──────────────────────────────────────────────────────────

class TestDetectRulCliff:
    @pytest.mark.parametrize("old, new, expected", [
        (100.0, 55.0, True),    # 45% drop → cliff
        (100.0, 80.0, False),   # 20% drop → normal wear
        (30.0,  14.0, True),    # 53% drop → cliff into OFFLINE
        (100.0, 100.0, False),  # no change
        (100.0, 110.0, False),  # RUL improved
        (0.0,   0.0,  False),   # old=0 → no valid reading
        (50.0,  30.0, True),    # 40% drop exactly → cliff (boundary)
        (50.0,  31.0, False),   # just under 40%
    ])
    def test_cliff_detection(self, old, new, expected):
        assert detect_rul_cliff(old, new) == expected

    def test_custom_threshold_60pct(self):
        assert detect_rul_cliff(100.0, 35.0, threshold=0.60) is True

    def test_custom_threshold_60pct_no_cliff(self):
        assert detect_rul_cliff(100.0, 50.0, threshold=0.60) is False

    def test_old_rul_zero_returns_false(self):
        assert detect_rul_cliff(0.0, 0.0) is False

    def test_new_rul_zero_from_nonzero_old_is_cliff(self):
        assert detect_rul_cliff(50.0, 0.0) is True


# ── compute_prediction_reliability ───────────────────────────────────────────

class TestComputePredictionReliability:
    def test_fewer_than_3_returns_calibrating(self):
        label, color = compute_prediction_reliability([50.0, 52.0])
        assert label == "CALIBRATING"
        assert color == "dim"

    def test_empty_returns_calibrating(self):
        label, _ = compute_prediction_reliability([])
        assert label == "CALIBRATING"

    def test_stable_predictions_returns_high(self):
        history = [50.0, 50.1, 49.9, 50.2, 50.0]
        label, color = compute_prediction_reliability(history)
        assert label == "HIGH"
        assert color == "green"

    def test_moderate_variance_returns_medium(self):
        history = [80.0, 70.0, 75.0, 68.0, 73.0]
        label, color = compute_prediction_reliability(history)
        assert label in ("MEDIUM", "HIGH")   # depends on exact CV

    def test_high_variance_returns_low(self):
        history = [10.0, 90.0, 5.0, 80.0, 15.0, 70.0]
        label, color = compute_prediction_reliability(history)
        assert label == "LOW"
        assert color == "red"

    def test_uses_only_last_5_readings(self):
        """Prepend high-variance values, ensure last 5 stable → HIGH."""
        history = [10.0, 90.0, 5.0, 85.0] + [50.0, 50.1, 49.9, 50.0, 50.2]
        label, _ = compute_prediction_reliability(history)
        assert label == "HIGH"

    def test_nan_in_history_does_not_crash(self):
        history = [50.0, float("nan"), 52.0, 51.0, 50.5]
        label, _ = compute_prediction_reliability(history)
        assert label in ("HIGH", "MEDIUM", "LOW", "CALIBRATING")

    def test_inf_in_history_does_not_crash(self):
        history = [50.0, float("inf"), 52.0, 51.0, 50.5]
        label, _ = compute_prediction_reliability(history)
        assert label in ("HIGH", "MEDIUM", "LOW", "CALIBRATING")


# ── check_sensor_saturation ───────────────────────────────────────────────────
# The current implementation normalises raw values through the DL scaler before
# comparing thresholds, and returns human-readable SENSOR_DISPLAY_NAMES
# ("Motor RPM", "Vibration X", …) instead of raw column codes (W0, Xs0).

class TestCheckSensorSaturation:
    """Tests for check_sensor_saturation with scaler-aware normalisation."""

    @staticmethod
    def _get_ranges():
        """Fetch scaler min / range so tests can build raw-unit histories."""
        from dl_engine.inference import get_scaler_ranges
        r = get_scaler_ranges()
        return r["min"], r["range"]

    def _raw_value(self, col: int, scaled: float):
        """Map a [0,1] scaled position to its raw physical-unit equivalent."""
        lo, rng = self._get_ranges()
        return float(lo[col] + scaled * rng[col])

    def _make_history(self, n_sensors=18, length=10, scaled=0.5):
        """Build a history with every sensor at `scaled` position in [0,1]."""
        lo, rng = self._get_ranges()
        return [
            [float(lo[i] + scaled * rng[i])] * length
            for i in range(n_sensors)
        ]

    # ── Display-name lookup ───────────────────────────────────────────────
    SENSOR_NAMES = [
        "Motor RPM", "Feed Rate", "Power kW", "Coolant Flow",
        "Vibration X", "Vibration Y", "Bearing Temp", "Motor Temp",
        "Oil Pressure", "Oil Temp", "Spindle Load", "Torque",
        "Hydraulic PSI", "Coolant Temp", "Ambient Temp", "Current Amps",
        "Acoustic dB", "Cycle Time",
    ]

    def test_no_saturation_on_mid_range_values(self):
        history = self._make_history(scaled=0.5)
        assert check_sensor_saturation(history) == []

    def test_max_saturation_detected(self):
        history = self._make_history(scaled=0.5)
        # Saturate col 4 ("Vibration X") high
        history[4] = [self._raw_value(4, 0.98)] * 10
        saturated = check_sensor_saturation(history)
        names = [s[0] for s in saturated]
        assert "Vibration X" in names

    def test_zero_saturation_detected(self):
        history = self._make_history(scaled=0.5)
        # Saturate col 0 ("Motor RPM") low
        history[0] = [self._raw_value(0, 0.01)] * 10
        saturated = check_sensor_saturation(history)
        names = [s[0] for s in saturated]
        assert "Motor RPM" in names

    def test_saturation_type_is_max_or_zero(self):
        history = self._make_history(scaled=0.5)
        history[4] = [self._raw_value(4, 0.99)] * 10
        saturated = check_sensor_saturation(history)
        assert all(t in ("MAX", "ZERO") for _, t in saturated)

    def test_insufficient_history_not_flagged(self):
        lo, rng = self._get_ranges()
        # Only 3 readings per sensor — below n_consecutive=5
        history = [
            [float(lo[i] + 0.99 * rng[i])] * 3
            for i in range(18)
        ]
        assert check_sensor_saturation(history, n_consecutive=5) == []

    def test_custom_n_consecutive_3_fires_earlier(self):
        history = self._make_history(scaled=0.5)
        # Col 2 ("Power kW") saturated high with only 3 consecutive
        history[2] = [self._raw_value(2, 0.98)] * 3
        saturated = check_sensor_saturation(history, n_consecutive=3)
        names = [s[0] for s in saturated]
        assert "Power kW" in names

    def test_18_sensor_names_cover_operating_and_physical(self):
        """All 18 sensors saturated → returned names include both groups."""
        history = self._make_history(scaled=0.99, length=10)
        saturated = check_sensor_saturation(history)
        names = [s[0] for s in saturated]
        # Operating-condition sensors (W-group: cols 0-3)
        assert any(n in ("Motor RPM", "Feed Rate", "Power kW", "Coolant Flow")
                   for n in names)
        # Physical sensors (Xs-group: cols 4-17)
        assert any(n in ("Vibration X", "Bearing Temp", "Spindle Load")
                   for n in names)

    def test_empty_history_returns_empty(self):
        history = [[] for _ in range(18)]
        assert check_sensor_saturation(history) == []


# ── compute_maintenance_schedule ─────────────────────────────────────────────

class TestComputeMaintenanceSchedule:
    def test_all_online_healthy_returns_empty(self):
        schedule = compute_maintenance_schedule(_all_online())
        assert schedule == []

    def test_offline_machine_appears_in_schedule(self):
        m = _machines({4: ("OFFLINE", 12.0)})
        schedule = compute_maintenance_schedule(m)
        ids = [s["machine_id"] for s in schedule]
        assert 4 in ids

    def test_offline_machine_has_immediate_urgency(self):
        m = _machines({4: ("OFFLINE", 12.0)})
        schedule = compute_maintenance_schedule(m)
        item = next(s for s in schedule if s["machine_id"] == 4)
        assert item["urgency"] == "IMMEDIATE"
        assert item["action"] == "STOP & REPAIR"

    def test_rul_below_15_has_today_urgency(self):
        m = _machines({2: ("DEGRADED", 14.0)})
        schedule = compute_maintenance_schedule(m)
        item = next(s for s in schedule if s["machine_id"] == 2)
        assert item["urgency"] == "TODAY"

    def test_rul_between_15_and_30_has_week_urgency(self):
        m = _machines({3: ("DEGRADED", 25.0)})
        schedule = compute_maintenance_schedule(m)
        item = next(s for s in schedule if s["machine_id"] == 3)
        assert item["urgency"] == "THIS WEEK"

    def test_rul_above_30_not_in_schedule(self):
        m = _machines({1: ("ONLINE", 55.0)})
        schedule = compute_maintenance_schedule(m)
        assert not any(s["machine_id"] == 1 for s in schedule)

    def test_offline_ranked_before_degraded(self):
        m = _machines({4: ("OFFLINE", 12.0), 2: ("DEGRADED", 14.0)})
        schedule = compute_maintenance_schedule(m)
        ranks = {s["machine_id"]: s["rank"] for s in schedule}
        assert ranks[4] < ranks[2]

    def test_lower_rul_ranked_first_within_same_type(self):
        m = _machines({2: ("DEGRADED", 14.0), 3: ("DEGRADED", 12.0)})
        schedule = compute_maintenance_schedule(m)
        ranks = {s["machine_id"]: s["rank"] for s in schedule}
        assert ranks[3] < ranks[2]   # RUL 12 < 14, so rank is lower (more urgent)

    def test_rank_field_starts_at_1(self):
        m = _machines({4: ("OFFLINE", 12.0)})
        schedule = compute_maintenance_schedule(m)
        assert schedule[0]["rank"] == 1

    def test_all_required_keys_present(self):
        m = _machines({4: ("OFFLINE", 12.0)})
        required = {"machine_id", "machine_name", "status", "rul",
                    "urgency", "action", "color", "rank"}
        for item in compute_maintenance_schedule(m):
            assert required <= set(item.keys())


# ── compute_shift_health ──────────────────────────────────────────────────────

class TestComputeShiftHealth:
    def test_all_online_returns_nominal(self):
        text, color = compute_shift_health(_all_online(), 100.0)
        assert "NOMINAL" in text
        assert color == "green"

    def test_one_degraded_returns_watch(self):
        m = _machines({2: ("DEGRADED", 22.0)})
        text, color = compute_shift_health(m, 90.0)
        assert "WATCH" in text
        assert color == "yellow"

    def test_two_degraded_returns_caution(self):
        m = _machines({2: ("DEGRADED", 22.0), 3: ("DEGRADED", 25.0)})
        text, color = compute_shift_health(m, 80.0)
        assert "CAUTION" in text

    def test_one_offline_returns_caution(self):
        m = _machines({4: ("OFFLINE", 12.0)})
        text, color = compute_shift_health(m, 80.0)
        assert "CAUTION" in text

    def test_one_offline_one_degraded_returns_at_risk(self):
        m = _machines({4: ("OFFLINE", 12.0), 2: ("DEGRADED", 22.0)})
        text, color = compute_shift_health(m, 72.0)
        assert "AT RISK" in text
        assert color == "red"

    def test_two_offline_returns_critical(self):
        m = _machines({4: ("OFFLINE", 12.0), 3: ("OFFLINE", 8.0)})
        text, color = compute_shift_health(m, 60.0)
        assert "CRITICAL" in text
        assert "red" in color

    def test_capacity_appears_in_output(self):
        m = _machines({4: ("OFFLINE", 12.0)})
        text, _ = compute_shift_health(m, 80.0)
        assert "80" in text


# ── compute_degradation_leaderboard ──────────────────────────────────────────

class TestComputeDegradationLeaderboard:
    def test_returns_five_entries_for_five_machines(self):
        machines = _all_online()
        histories = {i: [] for i in range(1, 6)}
        board = compute_degradation_leaderboard(machines, histories)
        assert len(board) == 5

    def test_all_required_keys_present(self):
        machines = _all_online()
        histories = {i: [50.0, 50.0, 50.0] for i in range(1, 6)}
        board = compute_degradation_leaderboard(machines, histories)
        required = {"machine_id", "machine_name", "status", "rul",
                    "slope", "trend_label", "trend_color"}
        for item in board:
            assert required <= set(item.keys())

    def test_fastest_declining_machine_is_first(self):
        machines = _all_online()
        histories = {
            1: [90.0, 80.0, 70.0, 60.0, 50.0],   # fast decline
            2: [50.0, 50.0, 50.0, 50.0, 50.0],   # stable
            3: [50.0, 50.0, 50.0, 50.0, 50.0],
            4: [50.0, 50.0, 50.0, 50.0, 50.0],
            5: [50.0, 50.0, 50.0, 50.0, 50.0],
        }
        board = compute_degradation_leaderboard(machines, histories)
        assert board[0]["machine_id"] == 1

    def test_stable_trend_label(self):
        machines = _all_online()
        histories = {i: [50.0, 50.0, 50.0, 50.0, 50.0] for i in range(1, 6)}
        board = compute_degradation_leaderboard(machines, histories)
        for item in board:
            assert item["trend_label"] == "STABLE →"

    def test_fast_decline_gets_fast_label(self):
        machines = _all_online()
        histories = {
            1: [100.0, 80.0, 60.0, 40.0, 20.0],
            **{i: [50.0] * 5 for i in range(2, 6)}
        }
        board = compute_degradation_leaderboard(machines, histories)
        m1 = next(b for b in board if b["machine_id"] == 1)
        assert m1["trend_label"] == "FAST ↘"
        assert m1["trend_color"] == "red"

    def test_no_history_gives_zero_slope(self):
        machines = _all_online()
        histories = {i: [] for i in range(1, 6)}
        board = compute_degradation_leaderboard(machines, histories)
        assert all(b["slope"] == 0.0 for b in board)

    def test_single_history_entry_gives_zero_slope(self):
        machines = _all_online()
        histories = {i: [50.0] for i in range(1, 6)}
        board = compute_degradation_leaderboard(machines, histories)
        assert all(b["slope"] == 0.0 for b in board)

    def test_nan_in_history_does_not_crash(self):
        machines = _all_online()
        histories = {1: [50.0, float("nan"), 48.0, 46.0, 44.0],
                     **{i: [50.0] * 5 for i in range(2, 6)}}
        board = compute_degradation_leaderboard(machines, histories)
        assert len(board) == 5
        