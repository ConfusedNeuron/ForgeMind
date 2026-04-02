"""
Stub RUL predictor for development and testing (Days 1–9).

Returns semi-realistic values so the new ops features can be tested
without the real DL model:
  - Cliff detection needs RUL to occasionally drop sharply
  - Prediction reliability needs some variance across calls
  - Maintenance schedule needs values spanning all 3 zones

On Day 10, replace:
    from terminal.dummy_oracle import predict_rul
with:
    from dl_engine.inference import predict_rul

The signature is identical. One line change.

──────────────────────────────────────────────────────
TEST MODE CHEATSHEET
──────────────────────────────────────────────────────
Change STUB_MODE to try different behaviours:

  "fixed_offline"   → always 15.0  (tests OFFLINE path)
  "fixed_degraded"  → always 25.0  (tests DEGRADED path)
  "fixed_healthy"   → always 80.0  (tests ONLINE path)
  "random_decay"    → (DEFAULT) realistic random degradation,
                      enables cliff detection + reliability tests
──────────────────────────────────────────────────────
"""

import numpy as np

STUB_MODE = "random_decay"   # ← change this for targeted testing

# State for random_decay mode: tracks call count to simulate gradual wear
_call_count = 0


def reset_call_count() -> None:
    """Reset the degradation counter. Called by Ctrl+R in app.py."""
    global _call_count
    _call_count = 0


def predict_rul(sensor_tensor: np.ndarray) -> float:
    """
    Stub RUL prediction function.

    Args:
        sensor_tensor: numpy array of shape (50, 18)
                       — 50 timesteps × 18 sensors (W0-W3, Xs0-Xs13), RAW

    Returns:
        float: RUL prediction (shift-cycles remaining)
    """
    global _call_count
    _call_count += 1

    if STUB_MODE == "fixed_offline":
        return 15.0

    elif STUB_MODE == "fixed_degraded":
        return 25.0

    elif STUB_MODE == "fixed_healthy":
        return 80.0

    else:  # "random_decay" — default
        # Simulate a machine that degrades realisitcally:
        #  - Mean RUL starts high, drops with each fault injected
        #  - Gaussian noise gives variation (so reliability score changes)
        #  - Occasional large drops simulate cliff events (40%+ drop)

        # Base RUL: starts at ~90 and trends toward ~10 over ~15 events
        base  = max(5.0, 90.0 - (_call_count * 5.5))

        # Random noise: ±15% variation around the base
        noise = np.random.normal(0, base * 0.12)

        # Occasional cliff: ~15% chance of a 50–70% sudden drop
        if np.random.random() < 0.15:
            cliff_factor = np.random.uniform(0.30, 0.50)  # lose 50–70% of base
            rul = base * cliff_factor + noise
        else:
            rul = base + noise

        return float(max(0.0, rul))
