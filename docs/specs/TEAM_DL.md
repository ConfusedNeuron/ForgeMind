# TEAM DL — Technical Runbook

## CNN-LSTM Remaining Useful Life Prediction Engine

**Owners:** 2 people · **Environment:** Kaggle H100 (training) → GTX 1650 (inference)

**Your single deliverable:** A function called `predict_rul(sensor_tensor) -> float` that other teams call. Nothing else matters. You own `/dl_engine/*`.

---

## 0. What You're Building — The 30-Second Version

```
['A_dev', 'A_test', 'A_var',
 'T_dev', 'T_test', 'T_var',
 'W_dev', 'W_test', 'W_var',
 'X_s_dev', 'X_s_test', 'X_s_var',
 'X_v_dev', 'X_v_test', 'X_v_var',
 'Y_dev', 'Y_test']
Sensor window (50 timesteps × 18 features)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  CNN Block (Conv1d × 2)       │
    │  Local temporal patterns      │
    │  (B, 50, 18) → (B, 50, 128)   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  LSTM Block (2-layer)         │
    │  Degradation trajectory       │
    │  (B, 50, 128) → (B, 128)      │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Regression Head              │
    │  Linear(128→64→1)             │
    │  (B, 128) → (B,)              │
    └───────────────┬───────────────┘
                    │
                    ▼
            RUL prediction (float)
            "This machine has ~22 shift-cycles left"

```

Team Agent sends you a `(50, 18)` numpy array. You return a float. That's the entire contract.

---

## 1. Dependencies — Complete Package List

```bash
# Kaggle environment (training) — most are pre-installed
pip install h5py numpy torch scikit-learn joblib

# Local environment (inference only)
pip install torch numpy scikit-learn joblib
```

| Package | Version | What it does in this project |
|---|---|---|
| **torch** (PyTorch) | ≥2.0 | CNN-LSTM model definition, training loop, inference |
| **numpy** | ≥1.24 | Array manipulation, data loading, tensor conversion |
| **h5py** | ≥3.8 | Reading the N-CMAPSS `.h5` dataset file |
| **scikit-learn** | ≥1.2 | `MinMaxScaler` for feature normalization |
| **joblib** | ≥1.3 | Serializing the fitted scaler to `.pkl` for inference |
| **matplotlib** | ≥3.7 | Training curves visualization (optional, Kaggle only) |

---

## 2. Dataset: N-CMAPSS DS02

**Source:** `kaggle.com/datasets/shreyaravi0/aircraft`

**File:** `N-CMAPSS_DS02-006.h5` (≈2.45 GB)

This is the N-CMAPSS (New CMAPSS) dataset — turbofan engine degradation under realistic flight conditions. DS02 is specifically for data-driven prognostics. The dataset is pre-split into development (training) and test sets inside the `.h5` file.

### 2.1 HDF5 Schema

The file uses `_dev` and `_test` suffixed keys. The `_dev` suffix is the training set, `_test` is the test set. The `_var` keys contain variable name strings for reference.

| HDF5 Key | Split | Contents | Shape |
|---|---|---|---|
| `W_dev` | Train | Operating conditions (Mach, altitude, TRA, T2) | `(N_train, 4)` |
| `W_test` | Test | Operating conditions | `(N_test, 4)` |
| `X_s_dev` | Train | Physical sensor readings (14 sensors) | `(N_train, 14)` |
| `X_s_test` | Test | Physical sensor readings | `(N_test, 14)` |
| `X_v_dev` | Train | Virtual sensor readings — **DO NOT USE** | `(N_train, 14)` |
| `X_v_test` | Test | Virtual sensor readings — **DO NOT USE** | `(N_test, 14)` |
| `T_dev` | Train | Health state auxiliary variables — **DO NOT USE** | `(N_train, 4)` |
| `T_test` | Test | Health state auxiliary variables — **DO NOT USE** | `(N_test, 4)` |
| `Y_dev` | Train | RUL ground truth labels | `(N_train, 1)` |
| `Y_test` | Test | RUL ground truth labels | `(N_test, 1)` |
| `A_dev` | Train | Unit/cycle metadata | `(N_train, 4)` |
| `A_test` | Test | Unit/cycle metadata | `(N_test, 4)` |
| `W_var` | — | Variable names for W columns | string array |
| `X_s_var` | — | Variable names for X_s columns | string array |
| `X_v_var` | — | Variable names for X_v columns | string array |
| `A_var` | — | Variable names for A columns | string array |
| `T_var` | — | Variable names for T columns | string array |

### 2.2 What You Use

**Model input:** `W` (4 operating conditions) + `X_s` (14 physical sensors) = **18 features per timestep.**

Concatenate `[W, X_s]` along axis=1 to get your 18-column feature matrix.

**Do NOT use:**
- `X_v` (virtual sensors) — model-derived, adds complexity without DL challenge
- `T` (health state) — leaks future information into the model

### 2.3 Train / Test Split

The split is baked into the `.h5` file. You do NOT manually split by unit IDs.

| Set | HDF5 suffix | Engine Units | Purpose |
|---|---|---|---|
| Development (train) | `_dev` | 2, 5, 10, 16, 18, 20 (6 units) | Model training and validation |
| Test | `_test` | 11, 14, 15 (3 units) | Final RMSE evaluation |

Verify this on Day 1 by printing `np.unique(A_dev[:, 0].astype(int))`.

### 2.4 Feature Column Layout

When you concatenate `[W, X_s]`, the resulting 18 columns are ordered as:

| Column | Source | Sensor ID | Factory fiction name |
|---|---|---|---|
| 0 | W | W0 | Load setting (Mach) |
| 1 | W | W1 | Ambient pressure (altitude) |
| 2 | W | W2 | Throttle resolver angle (TRA) |
| 3 | W | W3 | Inlet temperature (T2) |
| 4 | X_s | Xs0 | Physical sensor 1 |
| 5 | X_s | Xs1 | Physical sensor 2 |
| 6 | X_s | Xs2 | Physical sensor 3 — pressure |
| 7 | X_s | Xs3 | Physical sensor 4 |
| 8 | X_s | Xs4 | Physical sensor 5 — bearing temperature |
| 9 | X_s | Xs5 | Physical sensor 6 |
| 10 | X_s | Xs6 | Physical sensor 7 |
| 11 | X_s | Xs7 | Physical sensor 8 — vibration |
| 12 | X_s | Xs8 | Physical sensor 9 |
| 13 | X_s | Xs9 | Physical sensor 10 |
| 14 | X_s | Xs10 | Physical sensor 11 — RPM |
| 15 | X_s | Xs11 | Physical sensor 12 |
| 16 | X_s | Xs12 | Physical sensor 13 |
| 17 | X_s | Xs13 | Physical sensor 14 |

Team Agent uses the "Sensor ID" column to inject spikes. You don't need to care about this mapping — your model just sees 18 floats.

---

## 3. Data Pipeline — Complete, Copy-Paste Ready

```python
# dl_engine/dataset.py

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib


# ════════════════════════════════════════════════════════════════════
# STEP 1: Load the .h5 file
# ════════════════════════════════════════════════════════════════════

def load_ncmapss(h5_path, sampling=10):
    """
    Load N-CMAPSS DS02 dataset.

    Args:
        h5_path:  Path to the .h5 file
        sampling: Take every Nth sample to reduce memory.
                  10 = 0.1Hz (recommended for pipeline validation)
                  1  = full 1Hz (final training run if H100 memory allows)

    Returns:
        W_train, Xs_train, Y_train, A_train,
        W_test,  Xs_test,  Y_test,  A_test
    """
    with h5py.File(h5_path, 'r') as f:
        print("HDF5 keys:", list(f.keys()))

        # ── Training (development) set ──
        W_train  = np.array(f['W_dev'],   dtype=np.float32)[::sampling]
        Xs_train = np.array(f['X_s_dev'], dtype=np.float32)[::sampling]
        Y_train  = np.array(f['Y_dev'],   dtype=np.float32)[::sampling]
        A_train  = np.array(f['A_dev'],   dtype=np.float32)[::sampling]

        # ── Test set ──
        W_test   = np.array(f['W_test'],   dtype=np.float32)[::sampling]
        Xs_test  = np.array(f['X_s_test'], dtype=np.float32)[::sampling]
        Y_test   = np.array(f['Y_test'],   dtype=np.float32)[::sampling]
        A_test   = np.array(f['A_test'],   dtype=np.float32)[::sampling]

        # ── Variable names (for EDA reference) ──
        W_var  = [s.decode() for s in np.array(f['W_var'])]
        Xs_var = [s.decode() for s in np.array(f['X_s_var'])]
        print(f"W  columns ({len(W_var)}): {W_var}")
        print(f"Xs columns ({len(Xs_var)}): {Xs_var}")

    print(f"\nTrain shapes: W={W_train.shape}, Xs={Xs_train.shape}, "
          f"Y={Y_train.shape}, A={A_train.shape}")
    print(f"Test shapes:  W={W_test.shape}, Xs={Xs_test.shape}, "
          f"Y={Y_test.shape}, A={A_test.shape}")

    # Verify unit IDs
    train_units = np.unique(A_train[:, 0].astype(int))
    test_units  = np.unique(A_test[:, 0].astype(int))
    print(f"\nTrain units: {train_units}")
    print(f"Test units:  {test_units}")
    print(f"Total train points: {len(W_train):,}")
    print(f"Total test points:  {len(W_test):,}")

    # RUL stats
    print(f"\nRUL (train): min={Y_train.min():.1f}, max={Y_train.max():.1f}, "
          f"mean={Y_train.mean():.1f}")
    print(f"RUL (test):  min={Y_test.min():.1f}, max={Y_test.max():.1f}, "
          f"mean={Y_test.mean():.1f}")

    return W_train, Xs_train, Y_train, A_train, W_test, Xs_test, Y_test, A_test


# ════════════════════════════════════════════════════════════════════
# STEP 2: Concatenate features and normalize
# ════════════════════════════════════════════════════════════════════

def prepare_features(W_train, Xs_train, Y_train, W_test, Xs_test, Y_test):
    """
    Concatenate W + Xs → 18 features, fit MinMaxScaler on train only.

    Returns:
        X_train_scaled, Y_train_flat,
        X_test_scaled,  Y_test_flat,
        scaler   ← SAVE THIS — required for local inference
    """
    # Concatenate: (N, 4) + (N, 14) → (N, 18)
    X_train = np.concatenate([W_train, Xs_train], axis=1)
    X_test  = np.concatenate([W_test,  Xs_test],  axis=1)

    # Flatten Y from (N, 1) → (N,)
    Y_train_flat = Y_train.squeeze(-1)
    Y_test_flat  = Y_test.squeeze(-1)

    # Min-Max normalize: fit on TRAIN only, transform BOTH
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\nScaled X_train: {X_train_scaled.shape}, "
          f"range [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    print(f"Scaled X_test:  {X_test_scaled.shape}, "
          f"range [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")

    return X_train_scaled, Y_train_flat, X_test_scaled, Y_test_flat, scaler


# ════════════════════════════════════════════════════════════════════
# STEP 3: Sliding window dataset (respects unit boundaries)
# ════════════════════════════════════════════════════════════════════

WINDOW = 50
STRIDE = 1

class NCMAPSSDataset(Dataset):
    """
    Converts a flat (N, 18) feature matrix into sliding windows
    of shape (num_windows, 50, 18).

    CRITICAL: Windows are built PER UNIT. If you naively slide across
    the entire concatenated array, windows near unit transitions will
    contain data from two different engines. This is physically
    meaningless and will silently corrupt training.
    """
    def __init__(self, X, Y, A, window=50, stride=1):
        """
        Args:
            X: (N, 18) scaled feature matrix
            Y: (N,) RUL labels
            A: (N, 4+) metadata — column 0 is unit_id
            window: sliding window length (50)
            stride: step between consecutive windows (1)
        """
        self.samples = []
        self.labels  = []

        unit_ids = A[:, 0].astype(int)
        unique_units = np.unique(unit_ids)

        for uid in unique_units:
            mask = unit_ids == uid
            X_unit = X[mask]
            Y_unit = Y[mask]

            # Slide within this unit ONLY — never cross unit boundaries
            for i in range(0, len(X_unit) - window, stride):
                self.samples.append(X_unit[i : i + window])       # (50, 18)
                self.labels.append(Y_unit[i + window - 1])        # scalar

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels  = np.array(self.labels,  dtype=np.float32)
        print(f"  Created {len(self.samples):,} windows "
              f"from {len(unique_units)} units (window={window}, stride={stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx]),   # (50, 18)
            torch.tensor(self.labels[idx])     # scalar RUL
        )


# ════════════════════════════════════════════════════════════════════
# STEP 4: Full pipeline — call this from your Kaggle notebook
# ════════════════════════════════════════════════════════════════════

def build_dataloaders(h5_path, batch_size=512, sampling=10, window=50, stride=1):
    """
    End-to-end: .h5 file → DataLoaders ready for training.

    Usage in notebook:
        train_loader, test_loader, scaler = build_dataloaders(
            '/kaggle/input/aircraft/N-CMAPSS_DS02-006.h5'
        )
    """
    # Load raw data
    W_tr, Xs_tr, Y_tr, A_tr, W_te, Xs_te, Y_te, A_te = load_ncmapss(h5_path, sampling)

    # Prepare features + fit scaler
    X_tr, Y_tr_flat, X_te, Y_te_flat, scaler = prepare_features(
        W_tr, Xs_tr, Y_tr, W_te, Xs_te, Y_te
    )

    # Save scaler immediately — you WILL need this for local inference
    joblib.dump(scaler, '/kaggle/working/scaler.pkl')
    print("\n✅ Scaler saved to /kaggle/working/scaler.pkl")

    # Build windowed datasets (respecting unit boundaries)
    print("\nBuilding train windows...")
    train_ds = NCMAPSSDataset(X_tr, Y_tr_flat, A_tr, window, stride)
    print("Building test windows...")
    test_ds  = NCMAPSSDataset(X_te, Y_te_flat, A_te, window, stride)

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\nTrain: {len(train_ds):,} windows → {len(train_loader)} batches")
    print(f"Test:  {len(test_ds):,} windows → {len(test_loader)} batches")

    return train_loader, test_loader, scaler
```

### 3.1 Subsampling Strategy

| `sampling` | Effective rate | Memory | When to use |
|---|---|---|---|
| `10` | 0.1 Hz | ~1/10th | Pipeline validation, debugging, hyperparameter search |
| `5` | 0.2 Hz | ~1/5th | Intermediate — good accuracy/speed tradeoff |
| `1` | 1.0 Hz (full) | Full (~2.5 GB raw) | Final training run if H100 memory allows |

**Start with `sampling=10`. Always.** Validate the full pipeline end-to-end before scaling up.

### 3.2 Why Windows Must Respect Unit Boundaries

The flat arrays contain data from multiple engine units concatenated together. A naive sliding window across the entire array produces windows that straddle two different engines — e.g., the last 20 timesteps of Unit 2 followed by the first 30 timesteps of Unit 5. This is physically meaningless (two completely different machines) and will silently corrupt your training data. The `NCMAPSSDataset` class windows **per unit** to prevent this.

---

## 4. Model Architecture

```python
# dl_engine/model.py

import torch
import torch.nn as nn


class CNNLSTM_RUL(nn.Module):
    """
    CNN-LSTM for Remaining Useful Life regression.

    Architecture:
        Input (B, 50, 18)
          → Conv1d block (local temporal pattern extraction)
          → LSTM block (degradation trajectory modeling)
          → Regression head (single RUL float)

    Why CNN before LSTM?
        Conv1d captures short-range sensor correlations (vibration spikes,
        thermal transients) within a few timesteps. LSTM then models the
        longer-range degradation trend across the full 50-step window.

    Why BatchNorm1d?
        N-CMAPSS has massive intra-unit variance across flight phases
        (takeoff vs cruise vs descent). BatchNorm stabilizes gradients
        and accelerates convergence.
    """
    def __init__(self, n_features=18, window=50, cnn_filters=64,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        # ── CNN Block ──
        # Conv1d expects (Batch, Channels, Length)
        # We permute (B, 50, 18) → (B, 18, 50) before this
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── LSTM Block ──
        # After CNN + permute back: (B, 50, 128) → LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,    # 128
            hidden_size=lstm_hidden,       # 128
            num_layers=lstm_layers,        # 2
            batch_first=True,
            dropout=dropout
        )

        # ── Regression Head ──
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, 50, 18)
        x = x.permute(0, 2, 1)            # → (B, 18, 50)  for Conv1d
        x = self.cnn(x)                    # → (B, 128, 50)
        x = x.permute(0, 2, 1)            # → (B, 50, 128) for LSTM
        _, (h_n, _) = self.lstm(x)         # h_n: (2, B, 128)
        x = h_n[-1]                        # → (B, 128) last layer hidden
        return self.regressor(x).squeeze(-1)  # → (B,)
```

### 4.1 Tensor Shape Trace

```
Input:                (B,  50,  18)    ← DataLoader output
After permute:        (B,  18,  50)    ← Conv1d needs (B, Channels, Length)
After Conv1d #1:      (B,  64,  50)    ← 64 filters, kernel=3, padding=1 preserves length
After BatchNorm+ReLU: (B,  64,  50)
After Conv1d #2:      (B, 128,  50)    ← 128 filters
After BatchNorm+ReLU: (B, 128,  50)
After Dropout:        (B, 128,  50)
After permute back:   (B,  50, 128)    ← LSTM needs (B, Length, Features)
LSTM h_n:             (2,   B, 128)    ← 2 layers, hidden state at last timestep
h_n[-1]:              (B, 128)         ← take last layer only
After Linear(128→64): (B,  64)
After ReLU+Dropout:   (B,  64)
After Linear(64→1):   (B,   1)
After squeeze:        (B,)             ← final RUL prediction per sample
```

### 4.2 Parameter Count

```python
model = CNNLSTM_RUL()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")
# Expected: ~400K–500K parameters — lightweight, trains fast on H100
```

---

## 5. Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Window size | 50 | 50-timestep sliding window |
| Batch size | 512 | H100 handles 1024+; 512 is stable and fast |
| Learning rate | 1e-3 | Adam optimizer |
| Epochs | 50 (max) | With early stopping — likely stops around 25–35 |
| Loss function | MSELoss | Standard for RUL regression |
| Scheduler | ReduceLROnPlateau | patience=5, factor=0.5 |
| Early stopping | 10 epochs | Stop if val_loss stalls for 10 consecutive epochs |
| Dropout | 0.3 | Applied in CNN block and regression head |
| LSTM hidden | 128 | 2-layer LSTM |
| CNN filters | 64 | Doubles to 128 in second conv layer |

**Evaluation metrics:**
- **Primary:** RMSE = √(MSE) — penalizes large errors proportionally
- **Secondary:** PHM asymmetric scoring function — late predictions penalized exponentially harder than early ones

**Expected training time on H100:** ~3–5 min/epoch → 50 epochs ≈ 3–4 hours total, within the 12-hour Kaggle session cap.

---

## 6. Training Loop — Complete

```python
# dl_engine/train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time


CONFIG = {
    "window":       50,
    "batch_size":   512,
    "lr":           1e-3,
    "epochs":       50,
    "early_stop":   10,
    "dropout":      0.3,
    "lstm_hidden":  128,
    "cnn_filters":  64,
}


def train_model(model, train_loader, test_loader, config, device='cuda'):
    """
    Full training loop with checkpointing, early stopping, and logging.
    Run this inside your Kaggle notebook.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

    for epoch in range(config['epochs']):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)    # (B, 50, 18)
            Y_batch = Y_batch.to(device)    # (B,)

            optimizer.zero_grad()
            pred = model(X_batch)            # (B,)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_losses.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_true.append(Y_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_rmse   = np.sqrt(val_loss)

        # ── Scheduler ──
        scheduler.step(val_loss)

        # ── Logging ──
        elapsed = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val RMSE: {val_rmse:.2f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.1f}s")

        # ── Checkpoint (every improvement) ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'config': config
            }, '/kaggle/working/best_model.pt')
            print(f"  ✅ Checkpoint saved (val_rmse={val_rmse:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop']:
                print(f"\n⛔ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {config['early_stop']} epochs)")
                break

    # ── Final summary ──
    print(f"\nTraining complete.")
    print(f"Best val_rmse: {np.sqrt(best_val_loss):.2f}")
    print(f"Checkpoint: /kaggle/working/best_model.pt")
    print(f"Scaler:     /kaggle/working/scaler.pkl")

    return history
```

### 6.1 PHM Asymmetric Scoring Function

The PHM community uses this alongside RMSE. Late predictions (predicting more remaining life than actually exists) are penalized exponentially harder — because telling someone their machine is safe when it's about to fail is far more dangerous than being cautious.

```python
def phm_score(y_true, y_pred):
    """
    Asymmetric scoring function for RUL prediction.

    d < 0: early prediction (safe) — gentle exponential penalty
    d > 0: late prediction (dangerous) — harsh exponential penalty
    """
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,    # early: gentle
        np.exp( d / 10.0) - 1     # late:  harsh
    )
    return np.sum(scores)
```

### 6.2 Visualization (Kaggle Notebook)

```python
import matplotlib.pyplot as plt

def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')

    ax2.plot(history['val_rmse'], label='Val RMSE', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE (cycles)')
    ax2.legend()
    ax2.set_title('Validation RMSE')

    plt.tight_layout()
    plt.savefig('/kaggle/working/training_curves.png', dpi=150)
    plt.show()
```

---

## 7. Checkpoint Protocol — Non-Negotiable

Kaggle sessions can timeout at any point. Checkpointing is mandatory.

```python
# This happens automatically inside train_model() above.
# At the END of every epoch where val_loss improves:

torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
    'val_rmse': val_rmse,
    'config': config
}, '/kaggle/working/best_model.pt')

# The scaler is saved ONCE during build_dataloaders():
joblib.dump(scaler, '/kaggle/working/scaler.pkl')
```

### 7.1 Resuming After a Session Timeout

If Kaggle kills your session mid-training:

```python
# Resume from checkpoint
checkpoint = torch.load('/kaggle/working/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
best_val_loss = checkpoint['val_loss']
print(f"Resuming from epoch {start_epoch}, best val_loss={best_val_loss:.4f}")
```

### 7.2 What to Download from Kaggle

After training completes, download these two files from the Kaggle output tab:

| File | Size (approx) | Contains |
|---|---|---|
| `best_model.pt` | ~2–5 MB | Model weights, optimizer state, training metadata |
| `scaler.pkl` | ~2 KB | Fitted MinMaxScaler (18 feature min/max from training data) |

Commit both to `dl_engine/weights/` in the repo **before Day 10 (Integration Day)**.

---

## 8. GTX 1650 Smoke Test — Day 5

Do NOT wait until integration day to discover environment issues. On Day 5, run this on the local machine with randomly initialized weights:

```python
# smoke_test.py — run on the LOCAL machine, not Kaggle

import torch
import numpy as np
import sys

print(f"Python version:  {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version:   {np.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

# 1. Import your model class
from dl_engine.model import CNNLSTM_RUL

# 2. Create + save a randomly initialized model
model = CNNLSTM_RUL()
torch.save({'model_state_dict': model.state_dict(), 'epoch': 0}, 'test_weights.pt')
print("✅ Model saved")

# 3. Reload on CPU (exactly how inference.py does it)
checkpoint = torch.load('test_weights.pt', map_location='cpu')
model2 = CNNLSTM_RUL()
model2.load_state_dict(checkpoint['model_state_dict'])
model2.eval()
print("✅ Model loaded on CPU")

# 4. Forward pass with correct input shape
dummy = torch.randn(1, 50, 18)
with torch.no_grad():
    out = model2(dummy)
print(f"✅ Forward pass: output shape={out.shape}, value={out.item():.4f}")

# 5. Test the scaler path (create a dummy scaler)
from sklearn.preprocessing import MinMaxScaler
import joblib

dummy_data = np.random.randn(100, 18).astype(np.float32)
scaler = MinMaxScaler()
scaler.fit(dummy_data)
joblib.dump(scaler, 'test_scaler.pkl')
scaler2 = joblib.load('test_scaler.pkl')
scaled = scaler2.transform(np.random.randn(50, 18).astype(np.float32))
print(f"✅ Scaler save/load/transform works. Scaled shape: {scaled.shape}")

# 6. Clean up
import os
os.remove('test_weights.pt')
os.remove('test_scaler.pkl')

print(f"\n{'='*50}")
print(f"ALL SMOKE TESTS PASSED")
print(f"{'='*50}")
print(f"\n⚠️  Compare this PyTorch version with Kaggle's:")
print(f"   Local:  {torch.__version__}")
print(f"   Kaggle: (check your notebook — run torch.__version__)")
```

**What to check:**
- All 6 steps print ✅
- PyTorch version matches Kaggle (mismatches cause silent `torch.load()` failures)
- If CUDA is available, the test still works on CPU (inference must run on CPU)

---

## 9. Inference API — THE CONTRACT

This is the file that Team Agent and Team UI import. **Do not change the function signature.**

```python
# dl_engine/inference.py

import torch
import joblib
import numpy as np
from .model import CNNLSTM_RUL

_model  = None
_scaler = None


def load_model(weights_path='dl_engine/weights/best_model.pt',
               scaler_path='dl_engine/weights/scaler.pkl'):
    """Load model and scaler. Called automatically on first predict_rul() call."""
    global _model, _scaler

    checkpoint = torch.load(weights_path, map_location='cpu')
    _model = CNNLSTM_RUL()
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()

    _scaler = joblib.load(scaler_path)

    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', '?')}, "
          f"val_rmse={checkpoint.get('val_rmse', '?'):.2f})")


def predict_rul(sensor_tensor: np.ndarray) -> float:
    """
    THE CONTRACT FUNCTION — do not change this signature.

    Args:
        sensor_tensor: numpy array of shape (50, 18)
                       Raw (pre-normalization) sensor values.
                       Columns 0-3 = W (operating conditions)
                       Columns 4-17 = Xs (physical sensors)

    Returns:
        float: RUL prediction in production shift-cycles, clamped to >= 0.0

    Notes:
        - Scaling is handled internally via the saved scaler.
          Do NOT pre-scale the input.
        - Model runs on CPU. No GPU required.
        - First call triggers lazy model loading (~2s one-time cost).
        - Subsequent calls are <10ms.
    """
    if _model is None:
        load_model()

    # Validate input shape
    assert sensor_tensor.shape == (50, 18), \
        f"Expected shape (50, 18), got {sensor_tensor.shape}"

    # Scale using the SAME scaler that was fit during training
    scaled = _scaler.transform(sensor_tensor)   # (50, 18)

    # Convert to tensor and add batch dimension
    t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 50, 18)

    # Predict
    with torch.no_grad():
        rul = _model(t).item()

    return max(0.0, rul)  # RUL cannot be negative
```

### 9.1 What Other Teams Send You

Team Agent produces a `(50, 18)` numpy array of **raw sensor values** (not pre-scaled). Your function scales internally using the saved scaler. This is by design — other teams never touch your scaler.

### 9.2 What You Return

A single `float`. Represents RUL in shift-cycles. The Capacity Agent uses thresholds:

| Your output | What happens downstream |
|---|---|
| RUL > 30 | Machine stays **ONLINE** (full capacity) |
| 15 < RUL ≤ 30 | Machine goes **DEGRADED** (50% capacity) |
| RUL ≤ 15 | Machine goes **OFFLINE** (shutdown, maintenance triggered) |

You don't need to know or care about these thresholds. Just return the most accurate number you can.

---

## 10. Your Day-by-Day Checklist

| Day | Task | Done? |
|---|---|---|
| 1 | Add dataset to Kaggle notebook. Run `load_ncmapss()`. Print keys, shapes, unit IDs, RUL stats. Verify everything matches Section 2. | ☐ |
| 2 | Implement `dataset.py`. Run `build_dataloaders()` end-to-end with `sampling=10`. Verify scaler saves. Verify window counts look reasonable. | ☐ |
| 3 | Implement `model.py`. Run one training epoch on the small-sampled data. Verify loss decreases. Print parameter count. | ☐ |
| 4 | Full training run #1 with `sampling=10`. Monitor loss curves. Checkpoint saves every improvement. | ☐ |
| 5 | Analyze run #1 results. **Run GTX 1650 smoke test** (Section 8). Record PyTorch version on both machines. Try `sampling=5` if memory allows. | ☐ |
| 6 | Training run #2 with tuned hyperparameters. Try `sampling=1` if H100 memory permits. | ☐ |
| 7 | Validate on test units. Compute RMSE and PHM score. Plot predicted vs actual RUL. Assess: is accuracy acceptable? | ☐ |
| 8 | Export `best_model.pt` + `scaler.pkl`. Commit both to `dl_engine/weights/`. | ☐ |
| 9 | Buffer — retrain if RMSE too high (80% accuracy is acceptable, don't over-tune). Test `inference.py` locally with real weights. | ☐ |
| **10** | **INTEGRATION: Team UI swaps dummy oracle → your `inference.py`. Test end-to-end with real agent pipeline.** | ☐ |
| 11–13 | Integration bug fixes, rehearsals. | ☐ |
| 14 | Final bug fixes only. | ☐ |

---

## 11. Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: "object 'W' doesn't exist"` | Using wrong HDF5 key names | Use `f['W_dev']` and `f['W_test']` — keys have `_dev`/`_test` suffixes |
| OOM on Kaggle H100 | Full 1Hz data + large batch size | Use `sampling=10`, reduce `batch_size` to 256 |
| Val loss never decreases | Learning rate too high, or data pipeline bug | Verify scaler is fit on train only. Try `lr=3e-4`. Check windowing respects unit boundaries. |
| Good train RMSE, terrible test RMSE | Overfitting or window boundary corruption | Ensure windowing is per-unit (Section 3.2). Add more dropout. Reduce model complexity. |
| `torch.load()` fails on GTX 1650 | PyTorch version mismatch between Kaggle and local | Pin the same PyTorch version. Always use `map_location='cpu'`. |
| RUL predictions are all ~0 or all the same | Scaler not saved from the same training run, or model never actually converged | Verify `scaler.pkl` matches `best_model.pt`. Check training logs — did val_loss improve at all? |
| Inference returns negative values | Model regression head has no floor | Handled — `predict_rul()` clamps with `max(0.0, rul)` |
| Predicted RUL range doesn't match expected | Model trained on different data scale | This is normal. Check `Y_dev` and `Y_test` ranges on Day 1. Your model should roughly match those ranges. |
| `_var` keys fail to decode | String encoding issue in h5py | Use `[s.decode() for s in np.array(f['W_var'])]` to convert bytes → str |

---

## 12. Package Summary

Complete `requirements.txt` entries for Team DL:

```
# Kaggle (training) — most pre-installed, but pin versions
torch>=2.0.0
numpy>=1.24.0
h5py>=3.8.0
scikit-learn>=1.2.0
joblib>=1.3.0
matplotlib>=3.7.0

# Local (inference only)
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
joblib>=1.3.0
```

---

*You own `/dl_engine/*`. Your only deliverable to other teams is `predict_rul(sensor_tensor) -> float`. Make it work. Make it accurate. Make it load on a GTX 1650. Everything else is your team's internal business.*
