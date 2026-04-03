"""
tests/test_agent1.py
Agent 1 — Diagnostic Agent: automated unit tests + latency + mapping checks.

Run from project ROOT:
    python tests/test_agent1.py           # prints results to terminal
    python tests/test_agent1.py --save    # also writes results/agent1_report.txt
"""

import sys
import time
import argparse
import os
from pathlib import Path
from io import StringIO

# ── Add project root to path so `from agents.xxx import ...` works ────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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

# ── Result collector ──────────────────────────────────────────────────────────

results: list[dict] = []   # each entry: {name, passed, detail}
output_buffer = StringIO() # captures all print output for --save


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


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1: Sensor column mapping
# ─────────────────────────────────────────────────────────────────────────────

def suite_sensor_col_map():
    log("\n── SUITE 1: SENSOR_TO_COL mapping ──────────────────────────────────")

    cases = [
        ("W0",   0),
        ("W1",   1),
        ("W2",   2),
        ("W3",   3),
        ("Xs0",  4),
        ("Xs1",  5),
        ("Xs2",  6),   # pressure
        ("Xs3",  7),
        ("Xs4",  8),   # bearing temp  ← critical
        ("Xs5",  9),
        ("Xs6",  10),
        ("Xs7",  11),  # vibration
        ("Xs8",  12),
        ("Xs9",  13),
        ("Xs10", 14),  # RPM
        ("Xs11", 15),
        ("Xs12", 16),  # coolant
        ("Xs13", 17),
    ]

    for sensor_id, expected_col in cases:
        got = SENSOR_TO_COL.get(sensor_id)
        passed = got == expected_col
        record(
            f"SENSOR_TO_COL['{sensor_id}'] == {expected_col}",
            passed,
            f"got {got}" if not passed else ""
        )

    # Check no extra/missing keys
    all_expected = {f"W{i}" for i in range(4)} | {f"Xs{i}" for i in range(14)}
    missing  = all_expected - set(SENSOR_TO_COL.keys())
    extra    = set(SENSOR_TO_COL.keys()) - all_expected
    record("No missing sensors in SENSOR_TO_COL", len(missing) == 0,
           f"missing: {missing}" if missing else "")
    record("No extra sensors in SENSOR_TO_COL",  len(extra) == 0,
           f"extra: {extra}" if extra else "")


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2: _validate_domain
# ─────────────────────────────────────────────────────────────────────────────

def suite_validate_domain():
    log("\n── SUITE 2: _validate_domain ────────────────────────────────────────")

    def make_spike(**kwargs):
        base = dict(
            sensor_id="Xs4", spike_value=0.95,
            affected_window_positions=[45, 46, 47, 48, 49],
            fault_severity=FaultSeverity.HIGH,
            plain_english_summary="Test spike."
        )
        base.update(kwargs)
        return SensorSpike(**base)

    # Valid spike should pass
    ok, msg = _validate_domain(make_spike())
    record("Valid spike passes domain validation", ok, msg)

    # Invalid sensor IDs
    for bad_id in ["Xs14", "Xs17", "W4", "W10", "Sensor1", "xs4", "XS4"]:
        ok, msg = _validate_domain(make_spike(sensor_id=bad_id))
        record(f"sensor_id='{bad_id}' is rejected", not ok)

    # Out-of-range positions
    ok, _ = _validate_domain(make_spike(affected_window_positions=[50]))
    record("Position 50 is rejected (max is 49)", not ok)

    ok, _ = _validate_domain(make_spike(affected_window_positions=[-1]))
    record("Position -1 is rejected", not ok)

    ok, _ = _validate_domain(make_spike(affected_window_positions=[0, 49]))
    record("Positions [0, 49] are accepted (edge values)", ok)

    # Empty positions
    ok, _ = _validate_domain(make_spike(affected_window_positions=[]))
    record("Empty positions list is rejected", not ok)

    # Too many positions
    ok, _ = _validate_domain(make_spike(affected_window_positions=list(range(11))))
    record("11 positions rejected (max is 10)", not ok)

    ok, _ = _validate_domain(make_spike(affected_window_positions=list(range(10))))
    record("Exactly 10 positions accepted", ok)


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3: _inject_spike
# ─────────────────────────────────────────────────────────────────────────────

def suite_inject_spike():
    log("\n── SUITE 3: _inject_spike ───────────────────────────────────────────")

    base = np.random.rand(50, 18).astype(np.float32)
    base_original = base.copy()

    spike = SensorSpike(
        sensor_id="Xs4", spike_value=0.95,
        affected_window_positions=[47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Test."
    )

    injected = _inject_spike(base, spike)

    # base_window must not be modified
    record("base_window is not mutated by _inject_spike",
           np.array_equal(base, base_original))

    # Output shape and dtype
    record("Output shape is (50, 18)",  injected.shape == (50, 18))
    record("Output dtype is float32",   injected.dtype == np.float32)

    # Correct column (Xs4 = col 8) at positions 47,48,49 is spiked
    col = SENSOR_TO_COL["Xs4"]  # = 8
    for pos in [47, 48, 49]:
        passed = abs(injected[pos, col] - 0.95) < 1e-6
        record(f"Row {pos}, col {col} (Xs4) spiked to 0.95", passed,
               f"got {injected[pos, col]:.4f}" if not passed else "")

    # Rows 0–46 of col 8 must be unchanged
    unchanged = all(injected[r, col] == base[r, col] for r in range(47))
    record("Rows 0–46 of Xs4 column unchanged", unchanged)

    # Column 9 (Xs5) must be completely unchanged
    record("Adjacent column Xs5 (col 9) not touched",
           np.array_equal(injected[:, col + 1], base[:, col + 1]))

    # Test that a different sensor lands in a different column
    spike_vib = SensorSpike(
        sensor_id="Xs7", spike_value=0.88,
        affected_window_positions=[49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="Test vibration."
    )
    injected2 = _inject_spike(base, spike_vib)
    col_vib = SENSOR_TO_COL["Xs7"]  # = 11
    record("Xs7 spike lands in column 11",
           abs(injected2[49, col_vib] - 0.88) < 1e-6)
    record("Xs4 column untouched when Xs7 is spiked",
           np.array_equal(injected2[:, col], base[:, col]))


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 4: _get_fallback
# ─────────────────────────────────────────────────────────────────────────────

def suite_get_fallback():
    log("\n── SUITE 4: _get_fallback keyword matching ──────────────────────────")

    cases = [
        ("bearing temperature surge on Machine 4",    "Xs4"),   # bearing > temperature
        ("temperature overheat detected",              "Xs4"),
        ("pressure spike in hydraulic line",           "Xs2"),
        ("vibration and shaking on CNC-Alpha",         "Xs7"),
        ("coolant leak near the pump",                 "Xs12"),
        ("RPM fluctuation on motor",                   "Xs10"),
        ("speed drop on drive belt",                   "Xs10"),
        ("overload condition on machine",              "W0"),
        ("completely unrecognized random text here",   "Xs4"),   # default
    ]

    for text, expected_sensor in cases:
        spike = _get_fallback(text)
        passed = spike.sensor_id == expected_sensor
        record(
            f"'{text[:42]:42s}' → {expected_sensor}",
            passed,
            f"got {spike.sensor_id}" if not passed else ""
        )

    # All fallback spikes must contain [FALLBACK] tag
    for text, _ in cases:
        spike = _get_fallback(text)
        passed = "[FALLBACK]" in spike.plain_english_summary
        record(
            f"Fallback for '{text[:35]}...' has [FALLBACK] tag",
            passed
        )


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 5: Mapping check (live Gemini call per fault)
# ─────────────────────────────────────────────────────────────────────────────

def suite_mapping_check():
    log("\n── SUITE 5: Live sensor mapping check (Gemini API) ──────────────────")

    base = np.random.rand(50, 18).astype(np.float32)

    expected_mapping = [
        ("bearing temperature surge on Machine 4",  "Xs4",  {"HIGH"}),
        ("pressure spike in hydraulic line",         "Xs2",  {"HIGH"}),
        ("vibration and shaking on CNC-Alpha",       "Xs7",  {"HIGH", "MEDIUM"}),
        ("RPM drop on motor drive",                  "Xs10", {"HIGH", "MEDIUM"}),
        ("coolant leak near pump",                   "Xs12", {"HIGH"}),
    ]

    for text, expected_sensor, allowed_severities in expected_mapping:
        _, spike_dict, used_fallback = translate_fault_to_tensor(base, text)

        sensor_ok   = spike_dict["sensor_id"] == expected_sensor
        severity_ok = spike_dict["fault_severity"] in allowed_severities
        positions   = spike_dict["affected_window_positions"]
        recent_ok   = max(positions) >= 40    # fault should appear near end of window
        fallback_note = " [FALLBACK]" if used_fallback else ""

        record(
            f"Sensor: '{text[:40]:40s}' → {expected_sensor}",
            sensor_ok,
            f"got {spike_dict['sensor_id']}{fallback_note}" if not sensor_ok else fallback_note
        )
        record(
            f"Severity check for '{text[:35]:35s}'",
            severity_ok,
            f"got {spike_dict['fault_severity']}, allowed {allowed_severities}" if not severity_ok else ""
        )
        record(
            f"Recent positions for '{text[:35]:35s}'",
            recent_ok,
            f"max position = {max(positions)} (should be ≥ 40)" if not recent_ok else f"max={max(positions)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 6: Latency
# ─────────────────────────────────────────────────────────────────────────────

def suite_latency():
    log("\n── SUITE 6: Latency (single live Gemini call) ───────────────────────")

    base   = np.random.rand(50, 18).astype(np.float32)
    prompt = "bearing temperature surge on Machine 4"

    # Warm-up: Python import + client init already done, this is a cold API call
    t0 = time.perf_counter()
    _, spike_dict, _ = translate_fault_to_tensor(base, prompt)
    latency = time.perf_counter() - t0

    target_ok = latency < 3.0
    record(
        f"Latency < 3.0s (got {latency:.2f}s)",
        target_ok,
        "⚠ Add thinking_budget=0 to GenerateContentConfig" if not target_ok else ""
    )

    log(f"\n  Latency:  {latency:.2f}s")
    log(f"  Sensor:   {spike_dict['sensor_id']}")
    log(f"  Value:    {spike_dict['spike_value']:.2f}")
    log(f"  Severity: {spike_dict['fault_severity']}")
    log(f"  Positions:{spike_dict['affected_window_positions']}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    log(f"\n{'═' * 60}")
    log(f"  RESULTS: {passed}/{total} passed   {failed} failed")
    log(f"{'═' * 60}")

    if failed > 0:
        log("\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                log(f"    ✗ {r['name']}  {r['detail']}")


def save_report():
    report_dir = PROJECT_ROOT / "tests" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "agent1_report.txt"

    with open(report_path, "w") as f:
        f.write(output_buffer.getvalue())

    print(f"\n  Report saved → {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save results to tests/results/agent1_report.txt")
    parser.add_argument("--no-live", action="store_true",
                        help="Skip suites 5 and 6 (no Gemini API calls)")
    args = parser.parse_args()

    log("=" * 60)
    log("  AGENT 1 — DIAGNOSTIC AGENT TEST SUITE")
    log("=" * 60)

    suite_sensor_col_map()
    suite_validate_domain()
    suite_inject_spike()
    suite_get_fallback()

    if not args.no_live:
        suite_mapping_check()
        suite_latency()
    else:
        log("\n── Suites 5 & 6 skipped (--no-live) ────────────────────────────────")

    print_summary()

    if args.save:
        save_report()
