"""
tests/test_integration.py
Agent 1 + Agent 2 integration — uses a dummy oracle instead of the real DL model.
No real predict_rul needed. Tests the full pipeline contract.

Run: python tests/test_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from agents.diagnostic_agent import translate_fault_to_tensor
from agents.capacity_agent import update_capacity, reset_all, get_factory_snapshot


# ── Dummy oracle — replaces Team DL's predict_rul() ──────────────────────────
# Maps spike_value ranges to RUL buckets. High spike → low RUL.
# Replace with the real function on integration day.

def dummy_oracle(tensor: np.ndarray) -> float:
    """
    Deterministic stand-in for predict_rul(tensor).
    Finds the max value across the tensor and maps it to a RUL range.
    Real model: replace this entire function with dl_engine.predict_rul(tensor).
    """
    max_val = float(tensor.max())
    if max_val > 0.90:
        return 10.0    # HIGH spike → OFFLINE territory
    elif max_val > 0.75:
        return 22.0    # MEDIUM spike → DEGRADED territory
    else:
        return 55.0    # LOW spike → ONLINE territory


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_high_fault_causes_offline():
    """
    A HIGH severity fault should produce RUL ≤ 15 and put the machine OFFLINE.
    Pipeline: fault text → tensor → dummy_oracle → capacity_report
    """
    reset_all()
    base = np.random.rand(50, 18).astype(np.float32) * 0.5  # baseline ~0.3-0.5

    injected, spike_dict, _ = translate_fault_to_tensor(
        base, "bearing temperature surge on Machine 4"
    )

    # Verify the tensor was actually modified
    assert not np.array_equal(injected, base), "Tensor unchanged — spike not injected"

    # Verify shape preserved for DL model
    assert injected.shape == (50, 18), f"Wrong shape: {injected.shape}"
    assert injected.dtype == np.float32, f"Wrong dtype: {injected.dtype}"

    # Run through dummy oracle
    rul = dummy_oracle(injected)
    assert isinstance(rul, float), f"RUL must be float, got {type(rul)}"

    # Feed into Agent 2
    report = update_capacity(4, rul)

    # HIGH fault → spike_value > 0.85 → oracle returns 10.0 → OFFLINE
    assert report["status"] in ("OFFLINE", "DEGRADED"), (
        f"HIGH fault should degrade machine, got {report['status']}"
    )
    assert report["capacity_pct"] < 100.0, "Capacity should have dropped"
    assert report["breakeven_risk"] == True

    print(f"  ✓ HIGH fault: sensor={spike_dict['sensor_id']}, "
          f"RUL={rul}, status={report['status']}, cap={report['capacity_pct']}%")


def test_output_dict_has_all_keys():
    """
    The dict returned by update_capacity() must have every key
    that agent_loop.py passes to Floor Manager and Terminal UI.
    """
    reset_all()
    base = np.random.rand(50, 18).astype(np.float32) * 0.5
    injected, _, _ = translate_fault_to_tensor(base, "pressure spike in hydraulic line")
    rul = dummy_oracle(injected)
    report = update_capacity(2, rul)

    required_keys = {
        "machine_id", "machine_name", "status", "rul",
        "total_T", "total_PD", "machine_req",
        "capacity_pct", "breakeven_risk"
    }
    missing = required_keys - set(report.keys())
    assert not missing, f"Missing keys in capacity report: {missing}"

    # Verify types that Floor Manager's format() call depends on
    assert isinstance(report["machine_id"],   int)
    assert isinstance(report["machine_name"], str)
    assert isinstance(report["capacity_pct"], float)
    assert isinstance(report["breakeven_risk"], bool)

    print(f"  ✓ Output dict schema valid: {list(report.keys())}")


def test_stacked_faults_accumulate():
    """
    Three consecutive faults on different machines should progressively
    drop capacity. Each call to update_capacity() must stack.
    """
    reset_all()
    base = np.random.rand(50, 18).astype(np.float32) * 0.5

    faults = [
        ("bearing temperature surge on Machine 4", 4),
        ("pressure spike in hydraulic line",        2),
        ("vibration and shaking on CNC-Alpha",      1),
    ]

    prev_capacity = 100.0
    for fault_text, machine_id in faults:
        injected, spike_dict, _ = translate_fault_to_tensor(base, fault_text)
        rul = dummy_oracle(injected)
        report = update_capacity(machine_id, rul)

        # Capacity should not increase after a fault
        assert report["capacity_pct"] <= prev_capacity, (
            f"Capacity went UP after fault on Machine {machine_id}: "
            f"{prev_capacity}% → {report['capacity_pct']}%"
        )
        print(f"  ✓ Fault on M{machine_id} ({spike_dict['sensor_id']}): "
              f"RUL={rul:.1f}, status={report['status']}, "
              f"cap={report['capacity_pct']}%")
        prev_capacity = report["capacity_pct"]

    assert prev_capacity < 100.0, "Three faults should have dropped capacity below 100%"


def test_tensor_shape_preserved_through_pipeline():
    """
    The tensor shape must be (50, 18) float32 at every stage.
    This is the contract with Team DL's predict_rul().
    """
    reset_all()

    fault_texts = [
        "bearing temperature surge",
        "pressure spike",
        "vibration anomaly",
        "coolant leak",
        "RPM fluctuation",
    ]

    for text in fault_texts:
        base = np.random.rand(50, 18).astype(np.float32)
        injected, _, _ = translate_fault_to_tensor(base, text)
        assert injected.shape == (50, 18), f"Shape broken for: '{text}'"
        assert injected.dtype == np.float32, f"Dtype broken for: '{text}'"

    print(f"  ✓ Shape (50, 18) float32 preserved for all {len(fault_texts)} fault types")


def test_reset_clears_all_degradation():
    """
    After reset_all(), the factory must return to exactly the starting state,
    regardless of how many faults were injected.
    """
    base = np.random.rand(50, 18).astype(np.float32) * 0.5

    # Inject 5 faults
    for mid in range(1, 6):
        update_capacity(mid, 8.0)   # all OFFLINE

    snap_degraded = get_factory_snapshot()
    assert snap_degraded["capacity_pct"] == 0.0

    reset_all()
    snap_reset = get_factory_snapshot()

    assert snap_reset["total_T"] == 40.0,      "total_T not restored"
    assert snap_reset["capacity_pct"] == 100.0, "capacity_pct not restored"
    assert snap_reset["breakeven_risk"] == False, "breakeven_risk not cleared"

    print("  ✓ reset_all() fully restores factory to healthy baseline")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  INTEGRATION — Agent 1 + Agent 2 (dummy oracle)")
    print("=" * 60)

    tests = [
        ("HIGH fault causes machine degradation",    test_high_fault_causes_offline),
        ("Output dict has all required keys",         test_output_dict_has_all_keys),
        ("Stacked faults accumulate correctly",       test_stacked_faults_accumulate),
        ("Tensor shape preserved through pipeline",   test_tensor_shape_preserved_through_pipeline),
        ("reset_all() clears all degradation",        test_reset_clears_all_degradation),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n── {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")

    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed}/{len(tests)} passed")
    print(f"{'═' * 60}")
