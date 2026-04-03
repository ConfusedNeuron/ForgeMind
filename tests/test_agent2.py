"""
tests/test_agent2.py
Agent 2 — Capacity Agent: automated tests.

Run from project ROOT:
    python tests/test_agent2.py
    python tests/test_agent2.py --save
"""

import sys
import argparse
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.capacity_agent import (
    update_capacity,
    get_all_machine_statuses,
    get_factory_snapshot,
    reset_all,
    MACHINES,
    HEALTHY_BASELINE_REQ,
    BREAKEVEN_THRESHOLD,
    RUL_OFFLINE_THRESHOLD,
    RUL_DEGRADED_THRESHOLD,
)

results: list[dict] = []
output_buffer = StringIO()


def log(msg: str):
    print(msg)
    output_buffer.write(msg + "\n")


def record(name: str, passed: bool, detail: str = ""):
    results.append({"name": name, "passed": passed, "detail": detail})
    status = "✓" if passed else "✗"
    line = f"  {status} {name}"
    if detail:
        line += f"  [{detail}]"
    log(line)


def approx(a: float, b: float, tol: float = 0.01) -> bool:
    if a == float("inf") and b == float("inf"):
        return True
    return abs(a - b) <= tol


# ─────────────────────────────────────────────────────────────────────────────

def suite_constants():
    log("\\n── SUITE 1: Constants and baseline ──────────────────────────────────")
    record("HEALTHY_BASELINE_REQ ≈ 14.875",
           approx(HEALTHY_BASELINE_REQ, 14.875, 0.001))
    record("BREAKEVEN_THRESHOLD ≈ 15.916",
           approx(BREAKEVEN_THRESHOLD, 14.875 * 1.07, 0.001))
    record("5 machines in MACHINES dict", len(MACHINES) == 5)
    record("All machines start ONLINE",
           all(m["status"] == "ONLINE" for m in MACHINES.values()))
    record("All base_time == 8.0",
           all(m["base_time"] == 8.0 for m in MACHINES.values()))
    record("Total product demand == 595",
           sum(m["product_demand"] for m in MACHINES.values()) == 595)


def suite_status_transitions():
    log("\\n── SUITE 2: RUL → status transitions ───────────────────────────────")
    reset_all()

    # Boundary conditions
    cases = [
        (RUL_OFFLINE_THRESHOLD,       "OFFLINE"),   # exactly at threshold
        (RUL_OFFLINE_THRESHOLD - 0.1, "OFFLINE"),   # just below
        (RUL_OFFLINE_THRESHOLD + 0.1, "DEGRADED"),  # just above
        (RUL_DEGRADED_THRESHOLD,      "DEGRADED"),  # exactly at threshold
        (RUL_DEGRADED_THRESHOLD - 0.1,"DEGRADED"),  # just below
        (RUL_DEGRADED_THRESHOLD + 0.1,"ONLINE"),    # just above
        (999.0,                        "ONLINE"),    # far above
    ]
    for rul, expected_status in cases:
        reset_all()
        report = update_capacity(1, rul)
        passed = report["status"] == expected_status
        record(
            f"RUL={rul:.1f} → {expected_status}",
            passed,
            f"got {report['status']}" if not passed else ""
        )

    # DEGRADED capacity is 50%
    reset_all()
    update_capacity(1, 25.0)   # DEGRADED
    record("DEGRADED machine available_time == 4.0 hrs",
            MACHINES[1]["available_time"] == 4.0,
            f"got {MACHINES[1]['available_time']}")

    # OFFLINE capacity is 0
    reset_all()
    update_capacity(1, 10.0)   # OFFLINE
    record("OFFLINE machine available_time == 0.0 hrs",
            MACHINES[1]["available_time"] == 0.0,
            f"got {MACHINES[1]['available_time']}")


def suite_capacity_math():
    log("\\n── SUITE 3: Factory-wide capacity math ──────────────────────────────")
    reset_all()

    # All ONLINE baseline
    snap = get_factory_snapshot()
    record("All ONLINE: total_T == 40.0",       approx(snap["total_T"], 40.0))
    record("All ONLINE: machine_req ≈ 14.875",  approx(snap["machine_req"], 14.875))
    record("All ONLINE: capacity_pct == 100.0", approx(snap["capacity_pct"], 100.0))
    record("All ONLINE: breakeven_risk == False", snap["breakeven_risk"] == False)

    # Machine 4 DEGRADED (RUL=22)
    reset_all()
    r = update_capacity(4, 22.0)
    record("M4 DEGRADED: total_T == 36.0",       approx(r["total_T"], 36.0))
    record("M4 DEGRADED: machine_req ≈ 16.528",  approx(r["machine_req"], 16.528))
    record("M4 DEGRADED: capacity_pct == 90.0",  approx(r["capacity_pct"], 90.0))
    record("M4 DEGRADED: breakeven_risk == True", r["breakeven_risk"] == True)

    # Machine 4 OFFLINE (RUL=12)
    reset_all()
    r = update_capacity(4, 12.0)
    record("M4 OFFLINE: total_T == 32.0",        approx(r["total_T"], 32.0))
    record("M4 OFFLINE: machine_req ≈ 18.594",   approx(r["machine_req"], 18.594))
    record("M4 OFFLINE: capacity_pct == 80.0",   approx(r["capacity_pct"], 80.0))
    record("M4 OFFLINE: breakeven_risk == True",  r["breakeven_risk"] == True)

    # Stacked faults: M3 and M4 both OFFLINE
    reset_all()
    update_capacity(4, 12.0)
    r = update_capacity(3, 8.0)
    record("M3+M4 OFFLINE: total_T == 24.0",     approx(r["total_T"], 24.0))
    record("M3+M4 OFFLINE: machine_req ≈ 24.792",approx(r["machine_req"], 24.792))
    record("M3+M4 OFFLINE: capacity_pct == 60.0",approx(r["capacity_pct"], 60.0))

    # All OFFLINE edge case
    reset_all()
    for mid in range(1, 6):
        update_capacity(mid, 5.0)
    snap = get_factory_snapshot()
    record("All OFFLINE: total_T == 0.0",            approx(snap["total_T"], 0.0))
    record("All OFFLINE: machine_req == inf",         snap["machine_req"] == float("inf"))
    record("All OFFLINE: capacity_pct == 0.0",        approx(snap["capacity_pct"], 0.0))
    record("All OFFLINE: breakeven_risk == True",     snap["breakeven_risk"] == True)


def suite_cumulative_state():
    log("\\n── SUITE 4: Cumulative state (faults stack) ─────────────────────────")
    reset_all()

    # First fault
    r1 = update_capacity(4, 12.0)  # OFFLINE
    # Second fault on same machine — gets worse (already OFFLINE, stays OFFLINE)
    r2 = update_capacity(4, 5.0)
    record("Second fault on same machine keeps OFFLINE", r2["status"] == "OFFLINE")
    record("total_T unchanged when already OFFLINE",
           approx(r1["total_T"], r2["total_T"]))

    # Fault on different machine stacks
    r3 = update_capacity(3, 8.0)   # second machine OFFLINE
    record("Second machine OFFLINE reduces total_T further",
           r3["total_T"] < r1["total_T"])
    record("Two OFFLINEs: total_T == 24.0", approx(r3["total_T"], 24.0))

    # reset_all restores everything
    reset_all()
    snap = get_factory_snapshot()
    record("reset_all() restores total_T to 40.0",    approx(snap["total_T"], 40.0))
    record("reset_all() restores capacity_pct to 100", approx(snap["capacity_pct"], 100.0))
    record("reset_all() sets breakeven_risk to False",  snap["breakeven_risk"] == False)
    record("reset_all() sets all machines ONLINE",
           all(m["status"] == "ONLINE" for m in MACHINES.values()))


def suite_return_schema():
    log("\\n── SUITE 5: Return dict schema ──────────────────────────────────────")
    reset_all()
    r = update_capacity(4, 12.0)

    required_keys = {
        "machine_id", "machine_name", "status", "rul",
        "total_T", "total_PD", "machine_req", "capacity_pct", "breakeven_risk"
    }
    for key in required_keys:
        record(f"Key '{key}' present in return dict", key in r)

    record("machine_id is int",       isinstance(r["machine_id"], int))
    record("machine_name is str",     isinstance(r["machine_name"], str))
    record("status is str",           isinstance(r["status"], str))
    record("breakeven_risk is bool",  isinstance(r["breakeven_risk"], bool))
    record("rul is float",            isinstance(r["rul"], float))
    record("capacity_pct is float",   isinstance(r["capacity_pct"], float))

    # Invalid machine_id raises KeyError
    try:
        update_capacity(6, 10.0)
        record("machine_id=6 raises KeyError", False, "no exception raised")
    except KeyError:
        record("machine_id=6 raises KeyError", True)

    try:
        update_capacity(0, 10.0)
        record("machine_id=0 raises KeyError", False, "no exception raised")
    except KeyError:
        record("machine_id=0 raises KeyError", True)


def print_summary():
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    log(f"\\n{'═' * 60}")
    log(f"  RESULTS: {passed}/{total} passed   {failed} failed")
    log(f"{'═' * 60}")
    if failed:
        log("\\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                log(f"    ✗ {r['name']}  {r['detail']}")


def save_report():
    report_dir = PROJECT_ROOT / "tests" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "agent2_report.txt"
    with open(path, "w") as f:
        f.write(output_buffer.getvalue())
    print(f"\\n  Report saved → {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log("  AGENT 2 — CAPACITY AGENT TEST SUITE")
    log("=" * 60)

    suite_constants()
    suite_status_transitions()
    suite_capacity_math()
    suite_cumulative_state()
    suite_return_schema()
    print_summary()

    if args.save:
        save_report()
