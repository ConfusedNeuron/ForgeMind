# dl_engine/train.py
# Training loop, evaluation helpers, and checkpointing for the CNN-LSTM model.
# Run this locally for a quick smoke-test. Full training runs on Kaggle (H100).
#
# Usage (from repo root):
#   python -m dl_engine.train --h5 /path/to/N-CMAPSS_DS02-006.h5
#
# All outputs are written to dl_engine/weights/ by default.

import argparse
import math
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    load_h5,
    subsample_by_unit,
    build_feature_matrix,
    fit_scaler,
    apply_scaler,
    NCMAPSSDataset,
    make_dataloaders,
)
from .model import CNNLSTM_RUL


# ── Default config (matches the Optuna-tuned best from the notebook) ──────────
DEFAULT_CONFIG = {
    "sampling"    : 10,    # use 1 for full 1Hz run on H100; 10 for local testing
    "window"      : 50,
    "stride"      : 1,
    "n_features"  : 18,
    "cnn_filters" : 64,
    "lstm_hidden" : 128,
    "lstm_layers" : 2,
    "dropout"     : 0.3,
    "batch_size"  : 512,
    "lr"          : 1e-3,
    "epochs"      : 60,
    "patience"    : 20,
    "lr_patience" : 5,
    "lr_factor"   : 0.5,
}


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def compute_nasa_score(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    NASA asymmetric scoring function for RUL estimation.
    Late predictions (pred > true) are penalised more than early ones.
    Lower is better.
    """
    d = preds - targets
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(score))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Run inference over loader and return (val_mse, val_rmse, nasa_score).
    RUL predictions are clipped to >= 0.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, Y_b in loader:
            pred = model(X_b.to(device)).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(Y_b.numpy())
    preds   = np.clip(np.concatenate(all_preds),  0.0, None)
    targets = np.concatenate(all_targets)
    mse     = float(np.mean((preds - targets) ** 2))
    return mse, compute_rmse(preds, targets), compute_nasa_score(preds, targets)


def evaluate_per_unit(
    model: nn.Module,
    X_test_scaled: np.ndarray,
    Y_te: np.ndarray,
    A_te: np.ndarray,
    window: int,
    stride: int,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    """
    Returns a dict of {unit_id: rmse} for each test unit.
    Useful for diagnosing which unit drives aggregate val noise.
    """
    model.eval()
    results = {}
    for uid in sorted(np.unique(A_te[:, 0].astype(int))):
        mask = A_te[:, 0].astype(int) == uid
        ds   = NCMAPSSDataset(X_test_scaled[mask], Y_te[mask],
                              A_te[mask], window, stride)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        _, rmse, _ = evaluate(model, loader, device)
        results[uid] = rmse
    return results


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    h5_path: str,
    out_dir: str = "dl_engine/weights",
    config: dict = None,
    device: torch.device = None,
):
    cfg     = {**DEFAULT_CONFIG, **(config or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    data = load_h5(h5_path)

    W_tr, Xs_tr, Y_tr, A_tr = subsample_by_unit(
        data["W_tr"], data["Xs_tr"], data["Y_tr"], data["A_tr"], cfg["sampling"])
    W_te, Xs_te, Y_te, A_te = subsample_by_unit(
        data["W_te"], data["Xs_te"], data["Y_te"], data["A_te"], cfg["sampling"])

    X_train_raw = build_feature_matrix(W_tr, Xs_tr)
    X_test_raw  = build_feature_matrix(W_te, Xs_te)

    scaler        = fit_scaler(X_train_raw)
    X_train_scaled = apply_scaler(scaler, X_train_raw)
    X_test_scaled  = apply_scaler(scaler, X_test_raw)

    print(f"Train rows : {len(X_train_raw):,}  |  Test rows : {len(X_test_raw):,}")

    train_ds = NCMAPSSDataset(X_train_scaled, Y_tr, A_tr,
                              cfg["window"], cfg["stride"])
    test_ds  = NCMAPSSDataset(X_test_scaled,  Y_te, A_te,
                              cfg["window"], cfg["stride"])
    train_loader, test_loader = make_dataloaders(
        train_ds, test_ds, cfg["batch_size"])

    print(f"Train windows : {len(train_ds):,}  |  Test windows : {len(test_ds):,}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = CNNLSTM_RUL(
        n_features  = cfg["n_features"],
        window      = cfg["window"],
        cnn_filters = cfg["cnn_filters"],
        lstm_hidden = cfg["lstm_hidden"],
        lstm_layers = cfg["lstm_layers"],
        dropout     = cfg["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        patience = cfg["lr_patience"],
        factor   = cfg["lr_factor"],
    )

    # ── Loop ─────────────────────────────────────────────────────────────────
    history           = {"train_loss": [], "val_rmse": [], "val_score": [], "lr": []}
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    print(f"🚀 Training for up to {cfg['epochs']} epochs  "
          f"(early stop patience={cfg['patience']})")
    print("-" * 70)

    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        running_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, Y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(Y_b)

        train_loss = running_loss / len(train_ds)

        # validate
        val_mse, val_rmse, val_score = evaluate(model, test_loader, device)
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_score"].append(val_score)
        history["lr"].append(current_lr)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:>3d}/{cfg['epochs']}  "
              f"train_loss={train_loss:.4f}  "
              f"val_RMSE={val_rmse:.3f}  "
              f"NASA={val_score:.1f}  "
              f"lr={current_lr:.2e}  "
              f"[{elapsed/60:.1f}m]")

        # checkpoint every epoch (Kaggle session safety)
        ckpt = {
            "epoch"                : epoch,
            "model_state_dict"     : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "val_mse"              : val_mse,
            "val_rmse"             : val_rmse,
            "val_score"            : val_score,
            "config"               : cfg,
            "history"              : history,
        }
        torch.save(ckpt, out_dir / "latest_checkpoint.pt")

        if val_mse < best_val_loss:
            best_val_loss     = val_mse
            epochs_no_improve = 0
            torch.save(ckpt, out_dir / "best_model.pt")
            print(f"          ✅ New best  (val_RMSE={val_rmse:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"\n⏹  Early stopping at epoch {epoch}")
                break

    # ── Save scaler after training (always matches best_model.pt) ────────────
    joblib.dump(scaler, out_dir / "scaler.pkl")

    total_time = time.time() - t_start
    print(f"\n✅ Done in {total_time/60:.1f} minutes")
    print(f"   Best val RMSE : {math.sqrt(best_val_loss):.3f}")
    print(f"   Outputs saved to: {out_dir.resolve()}")

    # ── Per-unit breakdown ────────────────────────────────────────────────────
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    unit_rmse = evaluate_per_unit(
        model, X_test_scaled, Y_te, A_te,
        cfg["window"], cfg["stride"], device,
    )
    print("\n  PER-UNIT RMSE:")
    for uid, rmse in unit_rmse.items():
        print(f"    Unit {uid:>3d} : {rmse:.3f}")

    return model, scaler, history


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM RUL model on N-CMAPSS DS02")
    parser.add_argument("--h5",       required=True,            help="Path to N-CMAPSS_DS02-006.h5")
    parser.add_argument("--out_dir",  default="dl_engine/weights", help="Output directory for weights")
    parser.add_argument("--sampling", type=int, default=10,     help="Subsampling stride (1=full 1Hz)")
    parser.add_argument("--epochs",   type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--dropout",  type=float, default=0.3)
    args = parser.parse_args()

    config_override = {
        "sampling" : args.sampling,
        "epochs"   : args.epochs,
        "patience" : args.patience,
        "lr"       : args.lr,
        "dropout"  : args.dropout,
    }
    train(args.h5, args.out_dir, config_override)
