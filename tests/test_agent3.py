# tests/test_agent3.py
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.floor_manager import issue_dispatch_orders, _validate_output, _template_fallback

results = []

def record(name, passed, detail=""):
    results.append({"name": name, "passed": passed})
    icon = "✓" if passed else "✗"
    print(f"  {icon} {name}" + (f"  [{detail}]" if detail else ""))

def make_report(status, machine_id=4, rul=12.0, cap=80.0, req=18.594, risk=True):
    names = {1:"CNC-Alpha", 2:"CNC-Beta", 3:"Press-Gamma", 4:"Lathe-Delta", 5:"Mill-Epsilon"}
    return {
        "machine_id": machine_id, "machine_name": names[machine_id],
        "status": status, "rul": rul, "total_T": 32.0, "total_PD": 595,
        "machine_req": req, "capacity_pct": cap, "breakeven_risk": risk,
    }

# ── Suite 1: _validate_output ─────────────────────────────────────────────────
def suite_validate():
    print("\n── Suite 1: _validate_output ────────────────────────────────────────")
    r = make_report("OFFLINE")

    ok, _ = _validate_output("[Floor Manager] Lathe-Delta is OFFLINE. Halt. Reroute. 80.0%.", r)
    record("Valid output passes", ok)

    ok, _ = _validate_output("Lathe-Delta is OFFLINE.", r)
    record("Missing [Floor Manager] rejected", not ok)

    ok, _ = _validate_output("", r)
    record("Empty string rejected", not ok)

    ok, _ = _validate_output("[Floor Manager] " + "x" * 800, r)
    record("Over 800 chars rejected", not ok)

    ok, _ = _validate_output("[Floor Manager] Machine 4 is OFFLINE.", r)
    record("Missing machine name rejected", not ok)

# ── Suite 2: _template_fallback ───────────────────────────────────────────────
def suite_template():
    print("\n── Suite 2: _template_fallback ──────────────────────────────────────")

    r = make_report("OFFLINE", machine_id=4, rul=12.0, cap=80.0, req=18.594, risk=True)
    t = _template_fallback(r)
    print(f"\n  OFFLINE output:\n  {t}\n")
    record("OFFLINE: starts with [Floor Manager]",  t.startswith("[Floor Manager]"))
    record("OFFLINE: contains machine name",         "Lathe-Delta" in t)
    record("OFFLINE: contains RUL (12.0)",           "12.0" in t)
    record("OFFLINE: contains capacity (80.0)",      "80.0" in t)
    record("OFFLINE: mentions maintenance",          "maintenance" in t.lower())
    record("OFFLINE: passes validate",               _validate_output(t, r)[0])

    r = make_report("DEGRADED", machine_id=2, rul=22.0, cap=90.0, req=16.528, risk=True)
    t = _template_fallback(r)
    print(f"\n  DEGRADED output:\n  {t}\n")
    record("DEGRADED: starts with [Floor Manager]", t.startswith("[Floor Manager]"))
    record("DEGRADED: contains machine name",        "CNC-Beta" in t)
    record("DEGRADED: contains RUL (22.0)",          "22.0" in t)
    record("DEGRADED: mentions 50%",                 "50%" in t)
    record("DEGRADED: passes validate",              _validate_output(t, r)[0])

    r = make_report("ONLINE", machine_id=1, rul=55.0, cap=100.0, req=14.875, risk=False)
    t = _template_fallback(r)
    print(f"\n  ONLINE output:\n  {t}\n")
    record("ONLINE: starts with [Floor Manager]",   t.startswith("[Floor Manager]"))
    record("ONLINE: contains machine name",          "CNC-Alpha" in t)
    record("ONLINE: contains RUL (55.0)",            "55.0" in t)
    record("ONLINE: passes validate",                _validate_output(t, r)[0])

    # Live numbers check
    r = make_report("OFFLINE", machine_id=3, rul=9.5, cap=60.0, req=24.792, risk=True)
    t = _template_fallback(r)
    record("Live numbers: RUL 9.5 in output",        "9.5" in t)
    record("Live numbers: cap 60.0 in output",       "60.0" in t)
    record("Live numbers: req 24.792 in output",     "24.792" in t)
    record("Live numbers: Press-Gamma in output",    "Press-Gamma" in t)

# ── Suite 3: Live Gemini ──────────────────────────────────────────────────────
def suite_live():
    print("\n── Suite 3: Live Gemini calls ───────────────────────────────────────")
    for status, mid, rul, cap, req, risk in [
        ("OFFLINE",  4, 12.0, 80.0,  18.594, True),
        ("DEGRADED", 2, 22.0, 90.0,  16.528, True),
        ("ONLINE",   1, 55.0, 100.0, 14.875, False),
    ]:
        r = make_report(status, mid, rul, cap, req, risk)
        dispatch, used_fallback = issue_dispatch_orders(r)
        tag = " [FALLBACK]" if used_fallback else ""
        print(f"\n  {status} output{tag}:\n  {dispatch}\n")
        record(f"{status}: starts with [Floor Manager]{tag}", dispatch.startswith("[Floor Manager]"))
        record(f"{status}: machine name present{tag}",        r["machine_name"] in dispatch)
        record(f"{status}: under 800 chars{tag}",             len(dispatch) <= 800)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-live", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  AGENT 3 — FLOOR MANAGER TEST SUITE")
    print("=" * 60)

    suite_validate()
    suite_template()

    if not args.no_live:
        suite_live()
    else:
        print("\n── Suite 3 skipped (--no-live) ─────────────────────────────────────")

    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed}/{total} passed   {total - passed} failed")
    print(f"{'═' * 60}")
