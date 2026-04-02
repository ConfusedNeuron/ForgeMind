import hashlib, joblib, torch, io

# ── Fingerprint the scaler you downloaded ─────────────────────────────────────
with open("./dl_engine/weights/scaler.pkl", "rb") as f:
    actual_md5 = hashlib.md5(f.read()).hexdigest()

# ── Cross-check against the checkpoint's saved epoch/rmse ────────────────────
ckpt = torch.load("./dl_engine/weights/best_model.pt", map_location="cpu")

print(f"scaler.pkl  MD5  : {actual_md5}")
print(f"best epoch       : {ckpt['epoch']}")
print(f"val_RMSE         : {ckpt['val_rmse']:.3f}")
print(f"dropout in ckpt  : {ckpt['config'].get('dropout')}")
print(f"lr in ckpt       : {ckpt['config'].get('lr')}")