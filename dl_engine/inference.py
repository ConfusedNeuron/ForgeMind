# dl_engine/inference.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT FILE — this is the ONLY file other teams import.
# Public API: predict_rul(sensor_tensor: np.ndarray) -> float
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path

# ── Model definition (must match training) ────────────────────────────────────
class CNNLSTM_RUL(nn.Module):
    def __init__(self, n_features=18, window=50, cnn_filters=64,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size  = cnn_filters * 2,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        return self.regressor(x).squeeze(-1)


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_model  = None
_scaler = None


def load_model(
    weights_path: str = "dl_engine/weights/best_model.pt",
    scaler_path : str = "dl_engine/weights/scaler.pkl",
):
    """Load model weights and scaler. Called automatically on first predict_rul()."""
    global _model, _scaler

    checkpoint = torch.load(weights_path, map_location="cpu")  # CPU-safe on GTX 1650
    cfg = checkpoint.get("config", {})

    _model = CNNLSTM_RUL(
        n_features  = cfg.get("n_features",  18),
        window      = cfg.get("window",      50),
        cnn_filters = cfg.get("cnn_filters", 64),
        lstm_hidden = cfg.get("lstm_hidden", 128),
        lstm_layers = cfg.get("lstm_layers", 2),
        dropout     = cfg.get("dropout",     0.3),
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _scaler = joblib.load(scaler_path)
    print(f"[inference.py] Model loaded from {weights_path}  "
          f"(best epoch={checkpoint.get('epoch','?')}  "
          f"val_RMSE={checkpoint.get('val_rmse', float('nan')):.3f})")


def predict_rul(sensor_tensor: np.ndarray) -> float:
    """
    Predict Remaining Useful Life for a single window of sensor data.

    Parameters
    ----------
    sensor_tensor : np.ndarray, shape (50, 18)
        One sliding window — 50 time-steps × 18 features
        (4 operating conditions + 14 physical sensors), RAW (unscaled).
        The scaler is applied internally.

    Returns
    -------
    float
        Predicted RUL in production shift-cycles. Always >= 0.
    """
    if _model is None:
        load_model()

    assert sensor_tensor.shape == (50, 18), (
        f"predict_rul expects shape (50, 18), got {sensor_tensor.shape}"
    )

    scaled = _scaler.transform(sensor_tensor.astype(np.float32))  # (50, 18)
    scaled = np.clip(scaled, 0.0, 1.0)
    t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)    # (1, 50, 18)

    with torch.no_grad():
        rul = _model(t).item()

    return max(0.0, rul)


# ── Scaler-range utilities ────────────────────────────────────────────────────
# These expose the scaler's learned min/max so other modules can produce
# tensors in the correct raw-unit domain (the model fails silently on
# synthetic [0,1] data because the scaler collapses it to near-zero).

def get_healthy_baseline(noise_std_frac: float = 0.02) -> np.ndarray:
    """
    Build a (50, 18) tensor representing nominal operating conditions
    in RAW physical units (pre-scaling).

    Model probing revealed:
      scaled=0.10 → RUL ≈ 71  (healthy)
      scaled=0.50 → RUL ≈ 70  (mid-degradation)
      scaled=0.90 → RUL ≈ 1   (end-of-life)

    We use the low end of the scaler range (0.10) because the CNN-LSTM
    associates low scaled values with early-in-life / healthy conditions.

    Args:
        noise_std_frac: noise magnitude as a fraction of each feature's range.
                        0.02 means ±2% jitter.  Set to 0.0 for deterministic output.

    Returns:
        np.ndarray, shape (50, 18), dtype float32
    """
    if _scaler is None:
        load_model()

    HEALTHY_SCALED_POS = 0.10   # empirically gives highest RUL (~71)

    lo   = _scaler.data_min_                     # (18,)
    rng  = _scaler.data_range_                   # (18,)
    healthy_raw = lo + HEALTHY_SCALED_POS * rng   # raw-unit "healthy" vector

    baseline = np.tile(healthy_raw, (50, 1)).astype(np.float32)

    if noise_std_frac > 0:
        noise = np.random.normal(0, noise_std_frac * rng, size=(50, 18))
        baseline += noise.astype(np.float32)

    return baseline


def raw_value_for_scaled(sensor_col: int, scaled_target: float) -> float:
    """
    Inverse-map a desired [0, 1] scaled value back to raw physical units.

    Example: if the scaler learned  min=400, max=550 for column 8 (Xs4),
    then raw_value_for_scaled(8, 0.95) → 400 + 0.95 * 150 = 542.5

    The diagnostic agent stores spike_value in [0, 1]; this converts it
    to the raw value that predict_rul()'s internal scaler.transform()
    will map back to 0.95.

    Args:
        sensor_col:    column index 0–17 in the (50, 18) tensor
        scaled_target: desired position in [0, 1] after scaling

    Returns:
        float — raw-unit value
    """
    if _scaler is None:
        load_model()

    lo  = _scaler.data_min_[sensor_col]
    rng = _scaler.data_range_[sensor_col]
    return float(lo + scaled_target * rng)


def get_scaler_ranges() -> dict:
    """
    Return the scaler's learned min/max per feature for debugging.

    Returns:
        dict with keys 'min', 'max', 'range' — each a (18,) numpy array
    """
    if _scaler is None:
        load_model()

    return {
        "min":   _scaler.data_min_.copy(),
        "max":   _scaler.data_min_ + _scaler.data_range_,
        "range": _scaler.data_range_.copy(),
    }