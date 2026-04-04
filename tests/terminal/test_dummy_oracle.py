"""
tests/terminal/test_dummy_oracle.py
dummy_oracle.py — STUB_MODE determinism, call count, signature contract.

Run:  pytest tests/terminal/test_dummy_oracle.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import terminal.dummy_oracle as oracle_module
from terminal.dummy_oracle import predict_rul, reset_call_count


NOMINAL_TENSOR = np.random.rand(50, 18).astype(np.float32)


@pytest.fixture(autouse=True)
def reset_oracle():
    """Reset call counter and mode between tests."""
    original_mode = oracle_module.STUB_MODE
    reset_call_count()
    yield
    oracle_module.STUB_MODE = original_mode
    reset_call_count()


# ── Return type contract ──────────────────────────────────────────────────────

class TestReturnTypeContract:
    def test_returns_float(self):
        oracle_module.STUB_MODE = "fixed_healthy"
        assert isinstance(predict_rul(NOMINAL_TENSOR), float)

    def test_returns_non_negative_healthy(self):
        oracle_module.STUB_MODE = "fixed_healthy"
        assert predict_rul(NOMINAL_TENSOR) >= 0.0

    def test_returns_non_negative_offline(self):
        oracle_module.STUB_MODE = "fixed_offline"
        assert predict_rul(NOMINAL_TENSOR) >= 0.0

    def test_accepts_any_float32_tensor_shape_5018(self):
        oracle_module.STUB_MODE = "fixed_healthy"
        t = np.zeros((50, 18), dtype=np.float32)
        assert isinstance(predict_rul(t), float)


# ── Fixed modes — exact values ────────────────────────────────────────────────

class TestFixedModes:
    def test_fixed_offline_returns_15(self):
        oracle_module.STUB_MODE = "fixed_offline"
        assert predict_rul(NOMINAL_TENSOR) == 15.0

    def test_fixed_degraded_returns_25(self):
        oracle_module.STUB_MODE = "fixed_degraded"
        assert predict_rul(NOMINAL_TENSOR) == 25.0

    def test_fixed_healthy_returns_80(self):
        oracle_module.STUB_MODE = "fixed_healthy"
        assert predict_rul(NOMINAL_TENSOR) == 80.0

    def test_fixed_offline_is_deterministic_across_calls(self):
        oracle_module.STUB_MODE = "fixed_offline"
        results = [predict_rul(NOMINAL_TENSOR) for _ in range(10)]
        assert all(r == 15.0 for r in results)

    def test_fixed_modes_map_to_correct_status_zone(self):
        """Validate modes produce values that match the RUL threshold zones."""
        oracle_module.STUB_MODE = "fixed_offline"
        assert predict_rul(NOMINAL_TENSOR) < 20      # OFFLINE zone

        oracle_module.STUB_MODE = "fixed_degraded"
        assert 20 <= predict_rul(NOMINAL_TENSOR) <= 30  # DEGRADED zone

        oracle_module.STUB_MODE = "fixed_healthy"
        assert predict_rul(NOMINAL_TENSOR) > 30      # ONLINE zone


# ── random_decay mode ─────────────────────────────────────────────────────────

class TestRandomDecayMode:
    def test_random_decay_returns_float(self):
        oracle_module.STUB_MODE = "random_decay"
        assert isinstance(predict_rul(NOMINAL_TENSOR), float)

    def test_random_decay_is_non_negative(self):
        oracle_module.STUB_MODE = "random_decay"
        for _ in range(20):
            assert predict_rul(NOMINAL_TENSOR) >= 0.0

    def test_random_decay_trends_downward_with_call_count(self):
        """Over 15 calls, mean should drop meaningfully (base degrades)."""
        oracle_module.STUB_MODE = "random_decay"
        early  = [predict_rul(NOMINAL_TENSOR) for _ in range(3)]
        for _ in range(10):
            predict_rul(NOMINAL_TENSOR)
        late   = [predict_rul(NOMINAL_TENSOR) for _ in range(3)]
        assert np.mean(late) < np.mean(early)

    def test_reset_call_count_restores_high_rul(self):
        oracle_module.STUB_MODE = "random_decay"
        for _ in range(12):
            predict_rul(NOMINAL_TENSOR)
        late = predict_rul(NOMINAL_TENSOR)

        reset_call_count()
        early = predict_rul(NOMINAL_TENSOR)
        assert early > late   # resetting should give higher base RUL


# ── Call counter ──────────────────────────────────────────────────────────────

class TestCallCounter:
    def test_call_count_increments_on_each_predict(self):
        oracle_module.STUB_MODE = "random_decay"
        reset_call_count()
        for _ in range(5):
            predict_rul(NOMINAL_TENSOR)
        assert oracle_module._call_count == 5

    def test_reset_call_count_sets_to_zero(self):
        oracle_module.STUB_MODE = "random_decay"
        for _ in range(7):
            predict_rul(NOMINAL_TENSOR)
        reset_call_count()
        assert oracle_module._call_count == 0

    def test_fixed_modes_still_increment_call_count(self):
        oracle_module.STUB_MODE = "fixed_offline"
        reset_call_count()
        for _ in range(3):
            predict_rul(NOMINAL_TENSOR)
        assert oracle_module._call_count == 3
        