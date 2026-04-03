# tests/test_agent_loop.py
"""
Full pipeline test: Input Guard → Agent 1 → DL Oracle → Agent 2 → Agent 3

Run from project ROOT:
    python tests/test_agent_loop.py           # live Gemini calls
    python tests/test_agent_loop.py --no-live # forces offline/fallback mode, zero API calls
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.agent_loop as agent_loop_module
from agents.agent_loop import run_agent_loop, reset_factory, get_pipeline_status
from agents.capacity_agent import reset_all

results = []

def record(name, passed, detail=""):
    results.append({"name": name, "passed": passed})
    icon = "✓" if passed else "✗"
    print(f"  {icon} {name}" + (f"  [{detail}]" if detail else ""))

def make_base_window():
    """Clean baseline tensor — all sensors nominal (0.3–0.5 range)."""
    return (np.random.rand(50, 18) * 0.2 + 0.3).astype(np.float32)

# ── Dummy oracle ──────────────────────────────────────────────────────────────
def dummy_oracle(tensor: np.ndarray) -> float:
    """Stand-in for Team DL's predict_rul(). Max spike value → RUL bucket."""
    max_val = float(tensor.max())
    if max_val > 0.90:   return 10.0   # HIGH spike  → OFFLINE
    elif max_val > 0.75: return 22.0   # MED spike   → DEGRADED
    else:                return 55.0   # LOW/none    → ONLINE

def crashing_oracle(tensor: np.ndarray) -> float:
    """Simulates Team DL's function being broken."""
    raise RuntimeError("Model weights not loaded")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1: Input guard rejection (no LLM, instant)
# ─────────────────────────────────────────────────────────────────────────────
def suite_guard_rejection():
    print("\n── Suite 1: Input guard rejects bad inputs ──────────────────────────")
    reset_factory()

    bad_inputs = [
        ("hello world",                          "too short / no keywords"),
        ("what's for lunch?",                    "not a fault"),
        ("x",                                    "single char"),
        ("a" * 501,                              "too long"),
        ("the quick brown fox jumps over the lazy dog and runs away fast", "no fault keywords"),
    ]

    for text, reason in bad_inputs:
        r = run_agent_loop(text, 1, make_base_window(), dummy_oracle)
        record(f"Rejected: '{text[:30]}...' ({reason})", not r["valid"])
        record(f"rejection_reason populated for: '{text[:20]}'", bool(r["rejection_reason"]))
        record(f"spike is None for rejected input", r["spike"] is None)
        record(f"capacity_report is None for rejected input", r["capacity_report"] is None)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2: Return dict schema — every key Team Terminal expects
# ─────────────────────────────────────────────────────────────────────────────
def suite_return_schema():
    print("\n── Suite 2: Return dict schema completeness ─────────────────────────")
    reset_factory()

    r = run_agent_loop(
        "bearing temperature surge on Machine 4",
        4, make_base_window(), dummy_oracle
    )

    required_keys = {
        "valid", "rejection_reason", "spike", "rul",
        "capacity_report", "dispatch_orders", "machine_statuses",
        "used_fallback", "latency_ms",
    }
    missing = required_keys - set(r.keys())
    record("All top-level keys present", not missing, str(missing) if missing else "")

    # spike sub-keys
    if r["spike"]:
        spike_keys = {"sensor_id", "spike_value", "affected_window_positions",
                      "fault_severity", "plain_english_summary"}
        missing_spike = spike_keys - set(r["spike"].keys())
        record("spike dict has all sub-keys", not missing_spike,
               str(missing_spike) if missing_spike else "")

    # capacity_report sub-keys
    if r["capacity_report"]:
        cap_keys = {"machine_id", "machine_name", "status", "rul",
                    "total_T", "total_PD", "machine_req",
                    "capacity_pct", "breakeven_risk"}
        missing_cap = cap_keys - set(r["capacity_report"].keys())
        record("capacity_report has all sub-keys", not missing_cap,
               str(missing_cap) if missing_cap else "")

    # machine_statuses — all 5 machines
    record("machine_statuses has 5 entries", len(r["machine_statuses"]) == 5)

    # Types
    record("valid is bool",          isinstance(r["valid"], bool))
    record("rul is float",           isinstance(r["rul"], float))
    record("latency_ms is float",    isinstance(r["latency_ms"], float))
    record("used_fallback is bool",  isinstance(r["used_fallback"], bool))
    record("dispatch_orders is str", isinstance(r["dispatch_orders"], str))

    # Dispatch orders format
    record("dispatch_orders starts with [Floor Manager]",
           r["dispatch_orders"].startswith("[Floor Manager]"))

    print(f"\n  Dispatch: {r['dispatch_orders'][:100]}...")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3: Stacked faults accumulate across calls
# ─────────────────────────────────────────────────────────────────────────────
def suite_stacked_faults():
    print("\n── Suite 3: Stacked faults reduce capacity cumulatively ─────────────")
    reset_factory()

    faults = [
        ("bearing temperature surge on Machine 4", 4),
        ("pressure spike in hydraulic line",        2),
        ("vibration anomaly on CNC-Alpha",          1),
    ]

    prev_cap = 100.0
    for fault_text, machine_id in faults:
        r = run_agent_loop(fault_text, machine_id, make_base_window(), dummy_oracle)
        cap = r["capacity_report"]["capacity_pct"]
        status = r["capacity_report"]["status"]

        record(
            f"M{machine_id} fault: capacity ≤ previous ({prev_cap}% → {cap}%)",
            cap <= prev_cap,
        )
        record(f"M{machine_id} not ONLINE after HIGH spike", status != "ONLINE")
        prev_cap = cap

    record("Final capacity < 100%", prev_cap < 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 4: reset_factory() fully restores state
# ─────────────────────────────────────────────────────────────────────────────
def suite_reset():
    print("\n── Suite 4: reset_factory() restores full state ─────────────────────")

    # Degrade several machines
    for mid in [1, 2, 3]:
        run_agent_loop("bearing temperature surge", mid, make_base_window(), dummy_oracle)

    before_reset = run_agent_loop(
        "pressure surge on machine", 4, make_base_window(), dummy_oracle
    )
    cap_before = before_reset["capacity_report"]["capacity_pct"]

    reset_factory()

    # ✅ Check state IMMEDIATELY after reset — before any run_agent_loop call
    record("OFFLINE_MODE cleared after reset", not agent_loop_module.OFFLINE_MODE)
    statuses = get_pipeline_status()
    all_online = all(m["status"] == "ONLINE" for m in statuses["machine_statuses"])
    record("All 5 machines ONLINE after reset", all_online)

    # Force offline for post-reset call — avoids re-triggering OFFLINE_MODE via 429
    agent_loop_module.OFFLINE_MODE = True
    after_reset = run_agent_loop(
        "pressure surge on machine", 4, make_base_window(), dummy_oracle
    )
    agent_loop_module.OFFLINE_MODE = False

    cap_after = after_reset["capacity_report"]["capacity_pct"]
    record("Capacity is higher after reset", cap_after > cap_before,
           f"{cap_before}% → {cap_after}%")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 5: Oracle failure handling
# ─────────────────────────────────────────────────────────────────────────────
def suite_oracle_failure():
    print("\n── Suite 5: Crashing oracle defaults gracefully ─────────────────────")
    reset_factory()

    r = run_agent_loop(
        "bearing temperature surge on Machine 4",
        4, make_base_window(), crashing_oracle
    )

    record("Pipeline completes despite oracle crash", r["valid"] == True)
    record("RUL defaults to 25.0 (DEGRADED)",        r["rul"] == 25.0)
    record("Capacity report still populated",         r["capacity_report"] is not None)
    record("Dispatch orders still populated",         bool(r["dispatch_orders"]))
    record("Status is DEGRADED (RUL=25)",
           r["capacity_report"]["status"] == "DEGRADED")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 6: Offline mode — force OFFLINE_MODE=True, verify cache takes over
# ─────────────────────────────────────────────────────────────────────────────
def suite_offline_mode():
    print("\n── Suite 6: Forced offline mode — fallback cache active ─────────────")
    reset_factory()

    # Force offline before calling — simulates Gemini being unavailable
    agent_loop_module.OFFLINE_MODE = True

    faults = [
        ("bearing overheat on Machine 1",           1),
        ("pressure surge in hydraulic line",        2),
        ("vibration shaking on Machine 3",          3),
    ]

    for fault_text, machine_id in faults:
        r = run_agent_loop(fault_text, machine_id, make_base_window(), dummy_oracle)

        record(f"Offline pipeline completes: '{fault_text[:35]}'", r["valid"])
        record(f"used_fallback=True in offline mode", r["used_fallback"])
        record(f"spike populated from cache", r["spike"] is not None)
        record(f"dispatch starts with [Floor Manager]",
               r["dispatch_orders"].startswith("[Floor Manager]"))

        # Live numbers — capacity_pct from Agent 2 must appear in dispatch
        cap_str = str(r["capacity_report"]["capacity_pct"])
        record(
            f"Live cap% ({cap_str}) in offline dispatch",
            cap_str in r["dispatch_orders"]
        )

    # Reset offline mode
    agent_loop_module.OFFLINE_MODE = False
    record("OFFLINE_MODE reset to False after suite", not agent_loop_module.OFFLINE_MODE)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 7: Latency check (live mode only)
# ─────────────────────────────────────────────────────────────────────────────
def suite_latency():
    print("\n── Suite 7: Latency checks ──────────────────────────────────────────")
    reset_factory()

    # Offline should be < 200ms
    agent_loop_module.OFFLINE_MODE = True
    r = run_agent_loop("bearing overheat on Machine 4", 4, make_base_window(), dummy_oracle)
    record(
        f"Offline latency < 200ms (got {r['latency_ms']}ms)",
        r["latency_ms"] < 200,
        f"{r['latency_ms']}ms"
    )
    agent_loop_module.OFFLINE_MODE = False

    # Live Gemini should be < 10s (generous for free tier)
    r = run_agent_loop("pressure spike on Machine 2", 2, make_base_window(), dummy_oracle)
    record(
        f"Live latency < 10000ms (got {r['latency_ms']}ms)",
        r["latency_ms"] < 10000,
        f"{r['latency_ms']}ms"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-live", action="store_true",
                        help="Force OFFLINE_MODE=True — zero Gemini API calls")
    args = parser.parse_args()

    if args.no_live:
        agent_loop_module.OFFLINE_MODE = True
        print("  [Config] --no-live: OFFLINE_MODE forced True — no API calls")

    print("=" * 60)
    print("  PIPELINE — Agent 1 + 2 + 3 (test_agent_loop.py)")
    print("=" * 60)

    suite_guard_rejection()
    suite_return_schema()
    suite_stacked_faults()
    suite_reset()
    suite_oracle_failure()
    suite_offline_mode()

    if not args.no_live:
        suite_latency()
    else:
        print("\n── Suite 7 skipped (--no-live) ─────────────────────────────────────")

    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed}/{total} passed   {total - passed} failed")
    print(f"{'═' * 60}")
