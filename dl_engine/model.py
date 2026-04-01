# dl_engine/model.py
# CNN-LSTM architecture for RUL prediction on N-CMAPSS DS02.
# This is the authoritative model definition — inference.py duplicates
# this class locally so it remains a zero-dependency contract file.

import torch
import torch.nn as nn


class CNNLSTM_RUL(nn.Module):
    """
    1D-CNN extracts local temporal patterns across the 18 sensors.
    LSTM models the long-range degradation trajectory.
    Regression head outputs a single RUL float.

    Input shape  : (Batch, Window=50, Features=18)
    Output shape : (Batch,)  — one RUL float per sample
    """

    def __init__(
        self,
        n_features: int  = 18,
        window: int      = 50,
        cnn_filters: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float   = 0.3,
    ):
        super().__init__()

        # ── 1D-CNN Block ──────────────────────────────────────────────────────
        # Conv1d expects (Batch, Channels, Length).
        # Input is permuted (B, W, F) → (B, F, W) before this block.
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters,      kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # After CNN: (B, cnn_filters*2, window)

        # ── LSTM Block ────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = cnn_filters * 2,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── Regression Head ───────────────────────────────────────────────────
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # → single RUL float
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window, features)
        x = x.permute(0, 2, 1)          # → (B, features, window) for Conv1d
        x = self.cnn(x)                  # → (B, cnn_filters*2, window)
        x = x.permute(0, 2, 1)          # → (B, window, cnn_filters*2) for LSTM
        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers, B, hidden)
        x = h_n[-1]                      # last layer hidden state → (B, hidden)
        return self.regressor(x).squeeze(-1)   # → (B,)
