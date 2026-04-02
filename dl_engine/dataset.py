# dl_engine/dataset.py
# PyTorch Dataset + data loading utilities for N-CMAPSS DS02.
# Handles HDF5 loading, per-unit subsampling, MinMax scaling,
# and sliding-window construction without cross-unit boundary bleed.

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


# ── Published DS02 train/test split protocol ──────────────────────────────────
TRAIN_UNITS = [2, 5, 10, 16, 18, 20]
TEST_UNITS  = [11, 14, 15]


def load_h5(h5_path: str) -> dict:
    """
    Load N-CMAPSS DS02 arrays from the HDF5 file.

    The Kaggle dataset (shreyaravi0/aircraft) uses _dev / _test suffixes.
    Returns a dict with keys: W_tr, Xs_tr, Y_tr, A_tr, W_te, Xs_te, Y_te, A_te
    """
    with h5py.File(h5_path, 'r') as f:
        return {
            "W_tr"  : np.array(f['W_dev'],    dtype=np.float32),
            "Xs_tr" : np.array(f['X_s_dev'],  dtype=np.float32),
            "Y_tr"  : np.array(f['Y_dev'],    dtype=np.float32),
            "A_tr"  : np.array(f['A_dev'],    dtype=np.float32),
            "W_te"  : np.array(f['W_test'],   dtype=np.float32),
            "Xs_te" : np.array(f['X_s_test'], dtype=np.float32),
            "Y_te"  : np.array(f['Y_test'],   dtype=np.float32),
            "A_te"  : np.array(f['A_test'],   dtype=np.float32),
        }


def subsample_by_unit(
    W: np.ndarray,
    Xs: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    stride: int,
) -> tuple:
    """
    Subsample arrays at `stride` per unit independently.
    stride=1 is a no-op (returns inputs unchanged).
    Subsampling per-unit prevents boundary bleed at unit edges.
    """
    if stride == 1:
        return W, Xs, Y, A
    ids = np.unique(A[:, 0].astype(int))
    parts: list = [[], [], [], []]
    for uid in ids:
        m = A[:, 0].astype(int) == uid
        parts[0].append(W[m][::stride])
        parts[1].append(Xs[m][::stride])
        parts[2].append(Y[m][::stride])
        parts[3].append(A[m][::stride])
    return tuple(np.concatenate(p) for p in parts)


def build_feature_matrix(W: np.ndarray, Xs: np.ndarray) -> np.ndarray:
    """
    Concatenate operating conditions (W, 4 cols) and
    physical sensors (X_s, 14 cols) → 18-feature matrix.
    """
    return np.concatenate([W, Xs], axis=1)   # (N, 18)


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    """Fit MinMaxScaler on training data and return it."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(X_train)
    return scaler


def apply_scaler(
    scaler: MinMaxScaler,
    X: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """
    Apply a pre-fitted scaler to X.
    clip=True clamps values to [0, 1] to handle test-set outliers.
    """
    scaled = scaler.transform(X)
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    return scaled


class NCMAPSSDataset(Dataset):
    """
    Builds (window_size, n_features) sliding windows PER UNIT
    so window edges never bleed across unit boundaries.

    Parameters
    ----------
    X_scaled : np.ndarray, shape (N, n_features)  — already scaled
    Y        : np.ndarray, shape (N, 1)            — RUL labels
    A        : np.ndarray, shape (N, 5)            — metadata; col 0 = unit_id
    window   : int   — sliding window length (time-steps)
    stride   : int   — window stride (1 = fully overlapping)
    """

    def __init__(
        self,
        X_scaled: np.ndarray,
        Y: np.ndarray,
        A: np.ndarray,
        window: int = 50,
        stride: int = 1,
    ):
        self.samples: list = []
        self.labels: list  = []

        unit_ids = np.unique(A[:, 0].astype(int))
        for uid in unit_ids:
            mask = A[:, 0].astype(int) == uid
            X_u  = X_scaled[mask]
            Y_u  = Y[mask]
            n    = len(X_u)
            for i in range(0, n - window + 1, stride):
                self.samples.append(X_u[i : i + window])
                self.labels.append(Y_u[i + window - 1, 0])

        self.samples_arr = np.array(self.samples, dtype=np.float32)  # (N, W, F)
        self.labels_arr  = np.array(self.labels,  dtype=np.float32)  # (N,)

    def __len__(self) -> int:
        return len(self.samples_arr)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.samples_arr[idx]),
            torch.tensor(self.labels_arr[idx]),
        )


def make_dataloaders(
    train_ds: NCMAPSSDataset,
    test_ds: NCMAPSSDataset,
    batch_size: int = 512,
    num_workers: int = 4,
) -> tuple:
    """Return (train_loader, test_loader)."""
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    return train_loader, test_loader