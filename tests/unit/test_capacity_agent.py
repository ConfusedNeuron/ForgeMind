"""
tests/unit/test_capacity_agent.py
CapacityAgent (Agent 2) — constants, RUL→status transitions,
factory-wide math, cumulative state, return schema.

Run from project ROOT:  pytest tests/unit/test_capacity_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from agents.capacity_agent import (
    update_capacity, get_all_machine_statuses,
    get_factory_snapshot, reset_all, MACHINES,
    HEALTHY_BASELINE_REQ, BREAKEVEN_THRESHOLD,
    RUL_OFFLINE_THRESHOLD, RUL_DEGRADED_THRESHOLD,
)


def approx(a, b, tol=0.01):
    if a == float("inf") and b == float("inf"):
        return True
    return abs(a - b) <= tol


@pytest.fixture(autouse=True)
def fresh_factory():
    """Reset factory state before every test."""
    reset_all()
    yield
    reset_all()


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_healthy_baseline_req(self):
        assert approx(HEALTHY_BASELINE_REQ, 14.875, 0.001)

    def test_breakeven_threshold_is_7pct_above_baseline(self):
        assert approx(BREAKEVEN_THRESHOLD, 14.875 * 1.07, 0.001)

    def test_five_machines(self):
        assert len(MACHINES) == 5

    def test_all_machines_start_online(self):
        assert all(m["status"] == "ONLINE" for m in MACHINES.values())

    def test_all_base_time_is_8hrs(self):
        assert all(m["base_time"] == 8.0 for m in MACHINES.values())

    def test_total_product_demand_is_595(self):
        assert sum(m["product_demand"] for m in MACHINES.values()) == 595


# ── RUL → status transitions ──────────────────────────────────────────────────

class TestStatusTransitions:
    @pytest.mark.parametrize("rul, expected", [
        (RUL_OFFLINE_THRESHOLD,        "OFFLINE"),
        (RUL_OFFLINE_THRESHOLD - 0.1,  "OFFLINE"),
        (RUL_OFFLINE_THRESHOLD + 0.1,  "DEGRADED"),
        (RUL_DEGRADED_THRESHOLD,       "DEGRADED"),
        (RUL_DEGRADED_THRESHOLD - 0.1, "DEGRADED"),
        (RUL_DEGRADED_THRESHOLD + 0.1, "ONLINE"),
        (999.0,                        "ONLINE"),
    ])
    def test_rul_maps_to_correct_status(self, rul, expected):
        reset_all()
        report = update_capacity(1, rul)
        assert report["status"] == expected

    def test_degraded_available_time_is_halved(self):
        update_capacity(1, 25.0)   # DEGRADED
        assert MACHINES[1]["available_time"] == 4.0

    def test_offline_available_time_is_zero(self):
        update_capacity(1, 10.0)   # OFFLINE
        assert MACHINES[1]["available_time"] == 0.0


# ── Factory-wide capacity math ────────────────────────────────────────────────

class TestCapacityMath:
    def test_all_online_baseline(self):
        snap = get_factory_snapshot()
        assert approx(snap["total_T"], 40.0)
        assert approx(snap["machine_req"], 14.875)
        assert approx(snap["capacity_pct"], 100.0)
        assert snap["breakeven_risk"] is False

    def test_one_machine_degraded(self):
        r = update_capacity(4, 22.0)  # DEGRADED
        assert approx(r["total_T"], 36.0)
        assert approx(r["machine_req"], 16.528)
        assert approx(r["capacity_pct"], 90.0)
        assert r["breakeven_risk"] is True

    def test_one_machine_offline(self):
        r = update_capacity(4, 12.0)  # OFFLINE
        assert approx(r["total_T"], 32.0)
        assert approx(r["machine_req"], 18.594)
        assert approx(r["capacity_pct"], 80.0)
        assert r["breakeven_risk"] is True

    def test_two_machines_offline(self):
        update_capacity(4, 12.0)
        r = update_capacity(3, 8.0)
        assert approx(r["total_T"], 24.0)
        assert approx(r["machine_req"], 24.792)
        assert approx(r["capacity_pct"], 60.0)

    def test_all_offline_edge_case(self):
        for mid in range(1, 6):
            update_capacity(mid, 5.0)
        snap = get_factory_snapshot()
        assert approx(snap["total_T"], 0.0)
        assert snap["machine_req"] == float("inf")
        assert approx(snap["capacity_pct"], 0.0)
        assert snap["breakeven_risk"] is True


# ── Cumulative state ──────────────────────────────────────────────────────────

class TestCumulativeState:
    def test_second_fault_same_machine_stays_offline(self):
        update_capacity(4, 12.0)
        r = update_capacity(4, 5.0)
        assert r["status"] == "OFFLINE"

    def test_second_machine_offline_reduces_capacity_further(self):
        r1 = update_capacity(4, 12.0)
        r3 = update_capacity(3, 8.0)
        assert r3["total_T"] < r1["total_T"]
        assert approx(r3["total_T"], 24.0)

    def test_reset_restores_full_capacity(self):
        update_capacity(4, 12.0)
        update_capacity(3, 8.0)
        reset_all()
        snap = get_factory_snapshot()
        assert approx(snap["total_T"], 40.0)
        assert approx(snap["capacity_pct"], 100.0)
        assert snap["breakeven_risk"] is False
        assert all(m["status"] == "ONLINE" for m in MACHINES.values())


# ── Return schema contract ────────────────────────────────────────────────────

class TestReturnSchema:
    REQUIRED_KEYS = {
        "machine_id", "machine_name", "status", "rul",
        "total_T", "total_PD", "machine_req", "capacity_pct", "breakeven_risk",
    }

    def test_all_required_keys_present(self):
        r = update_capacity(4, 12.0)
        assert self.REQUIRED_KEYS <= set(r.keys())

    def test_field_types(self):
        r = update_capacity(4, 12.0)
        assert isinstance(r["machine_id"],    int)
        assert isinstance(r["machine_name"],  str)
        assert isinstance(r["status"],        str)
        assert isinstance(r["rul"],           float)
        assert isinstance(r["capacity_pct"],  float)
        assert isinstance(r["breakeven_risk"], bool)

    def test_invalid_machine_id_raises(self):
        with pytest.raises(KeyError):
            update_capacity(6, 10.0)

    def test_machine_id_zero_raises(self):
        with pytest.raises(KeyError):
            update_capacity(0, 10.0)
    