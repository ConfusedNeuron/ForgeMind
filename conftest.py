"""
conftest.py
Shared pytest fixtures for the ForgeMind test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.schemas import SensorSpike, FaultSeverity


@pytest.fixture
def base_window() -> np.ndarray:
    """Clean 50×18 float32 tensor — all sensors nominal (0.3–0.5 range)."""
    rng = np.random.default_rng(42)
    return (rng.random((50, 18)) * 0.2 + 0.3).astype(np.float32)


@pytest.fixture
def dummy_oracle():
    """Stand-in for predict_rul(). Max spike value → RUL bucket."""
    def _oracle(tensor: np.ndarray) -> float:
        max_val = float(tensor.max())
        if max_val > 0.90:   return 10.0   # HIGH spike  → OFFLINE
        elif max_val > 0.75: return 22.0   # MED spike   → DEGRADED
        else:                return 55.0   # LOW/none    → ONLINE
    return _oracle


@pytest.fixture
def crashing_oracle():
    """Simulates the DL model being unavailable."""
    def _oracle(tensor: np.ndarray) -> float:
        raise RuntimeError("Model weights not loaded")
    return _oracle


@pytest.fixture
def make_spike():
    """Factory for SensorSpike instances with sensible defaults."""
    def _make(**kwargs):
        defaults = dict(
            sensor_id="Xs4",
            spike_value=0.95,
            affected_window_positions=[45, 46, 47, 48, 49],
            fault_severity=FaultSeverity.HIGH,
            plain_english_summary="Test spike.",
        )
        defaults.update(kwargs)
        return SensorSpike(**defaults)
    return _make


@pytest.fixture
def make_capacity_report():
    """Factory for capacity report dicts matching Agent 2's return schema."""
    _names = {1: "CNC-Alpha", 2: "CNC-Beta", 3: "Press-Gamma",
              4: "Lathe-Delta", 5: "Mill-Epsilon"}

    def _make(status, machine_id=4, rul=12.0, cap=80.0, req=18.594, risk=True):
        return {
            "machine_id":    machine_id,
            "machine_name":  _names[machine_id],
            "status":        status,
            "rul":           rul,
            "total_T":       32.0,
            "total_PD":      595,
            "machine_req":   req,
            "capacity_pct":  cap,
            "breakeven_risk": risk,
        }
    return _make


def approx_equal(a: float, b: float, tol: float = 0.01) -> bool:
    if a == float("inf") and b == float("inf"):
        return True
    return abs(a - b) <= tol
