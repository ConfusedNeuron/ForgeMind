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