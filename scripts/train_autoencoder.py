"""
Run this script to train the GRU sequence autoencoder.

Usage:
    python scripts/train_autoencoder.py

Requires:
    data/processed/features.csv   (from scripts/build_features.py)
    data/ground_truth/labels.csv  (from scripts/generate_data.py)

Outputs:
    models/autoencoder.keras
    models/autoencoder_scaler.pkl
    models/autoencoder_meta.json   (threshold, feature columns, seq length)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from config import settings as cfg

np.random.seed(cfg.RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(cfg.RANDOM_SEED)

from src.models.autoencoder import (
    chronological_split, build_sequences, apply_log1p,
    build_autoencoder, reconstruction_errors, save_artifacts,
)


def main():
    features = pd.read_csv(cfg.PROCESSED_DATA_DIR + "/features.csv")
    labels = pd.read_csv(cfg.GROUND_TRUTH_PATH)
    assert "is_anomaly" not in features.columns, "LEAKAGE: features.csv already has labels!"

    df = features.merge(labels, on="event_id", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_df, val_df, test_df, (train_cutoff, val_cutoff) = chronological_split(df)
    print("=" * 70)
    print("SENTINELAI — GRU AUTOENCODER TRAINING")
    print("=" * 70)
    print(f"Train: {len(train_df)} events (up to {train_cutoff})")
    print(f"Val  : {len(val_df)} events (up to {val_cutoff})")
    print(f"Test : {len(test_df)} events")

    # ---- Split integrity: no time overlap between splits, ever ----
    if len(train_df) and len(val_df):
        assert train_df["timestamp"].max() <= val_df["timestamp"].min(), \
            "LEAKAGE: train/val splits overlap in time!"
    if len(val_df) and len(test_df):
        assert val_df["timestamp"].max() <= test_df["timestamp"].min(), \
            "LEAKAGE: val/test splits overlap in time!"
    print("[OK] Train/val/test splits do not overlap in time.")

    feature_cols = cfg.MODEL_FEATURE_COLUMNS
    train_df = apply_log1p(train_df, cfg.LOG1P_COLUMNS)
    val_df = apply_log1p(val_df, cfg.LOG1P_COLUMNS)
    test_df = apply_log1p(test_df, cfg.LOG1P_COLUMNS)

    # ---- Build raw (unscaled) sequences first, per split ----
    X_train_raw, y_train, _, _ = build_sequences(train_df, feature_cols)
    X_val_raw, y_val, val_ids, _ = build_sequences(val_df, feature_cols)
    X_test_raw, y_test, test_ids, _ = build_sequences(test_df, feature_cols)

    print(f"\nSequences  -> train: {len(X_train_raw)} (label counts {np.bincount(y_train)}), "
          f"val: {len(X_val_raw)} (label counts {np.bincount(y_val)}), "
          f"test: {len(X_test_raw)} (label counts {np.bincount(y_test)})")

    # ---- Fit scaler on TRAIN sequences only (flatten across timesteps) ----
    n_features = len(feature_cols)
    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, n_features))

    def scale(X):
        if len(X) == 0:
            return X
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_features)).reshape(shape).astype(np.float32)

    X_train = scale(X_train_raw)
    X_val = scale(X_val_raw)
    X_test = scale(X_test_raw)

    # ---- Train ONLY on windows that are entirely normal ----
    X_train_normal = X_train[y_train == 0]
    print(f"\nTraining on {len(X_train_normal)} purely-normal sequences "
          f"(excluded {len(X_train) - len(X_train_normal)} windows touching an anomaly).")

    model = build_autoencoder(seq_len=cfg.SEQUENCE_LENGTH, n_features=n_features)
    model.summary()

    # Monitor on NORMAL validation sequences only, so early stopping reflects
    # "are we still reconstructing normal behavior well" rather than being
    # confused by anomalous validation windows.
    X_val_normal = X_val[y_val == 0]

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train_normal, X_train_normal,
        validation_data=(X_val_normal, X_val_normal) if len(X_val_normal) > 0 else None,
        epochs=cfg.AE_EPOCHS,
        batch_size=cfg.AE_BATCH_SIZE,
        callbacks=[early_stop] if len(X_val_normal) > 0 else [],
        verbose=2,
    )

    # ---- Threshold selection: percentile of NORMAL validation reconstruction error ----
    val_errors = reconstruction_errors(model, X_val)
    val_normal_errors = val_errors[y_val == 0]
    threshold = float(np.percentile(val_normal_errors, cfg.AE_THRESHOLD_PERCENTILE))
    print(f"\nThreshold (P{cfg.AE_THRESHOLD_PERCENTILE} of normal VAL reconstruction error): {threshold:.6f}")

    # ---- Sanity-check on VALIDATION (not the final reported metric -- that's Stage 8) ----
    val_pred = (val_errors > threshold).astype(int)
    if len(np.unique(y_val)) > 1:
        p = precision_score(y_val, val_pred, zero_division=0)
        r = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        print(f"Validation sanity check -> precision={p:.3f} recall={r:.3f} f1={f1:.3f} "
              f"(flagged {val_pred.sum()}/{len(val_pred)})")

    # ---- Persist everything ----
    save_artifacts(model, scaler, threshold, feature_cols, cfg.SEQUENCE_LENGTH)

    print("\n" + "=" * 70)
    print(f"[OK] Model saved to {cfg.AUTOENCODER_MODEL_PATH}")
    print(f"[OK] Scaler saved to {cfg.AUTOENCODER_SCALER_PATH}")
    print(f"[OK] Threshold/metadata saved to {cfg.AUTOENCODER_META_PATH}")
    print("=" * 70)

    # Quick separation check on TEST (informational only, full eval in Stage 8)
    if len(X_test) > 0:
        test_errors = reconstruction_errors(model, X_test)
        print("\nReconstruction error by label on TEST split (informational):")
        print(f"  normal    (n={int((y_test==0).sum())}): mean={test_errors[y_test==0].mean():.4f}")
        if (y_test == 1).sum() > 0:
            print(f"  anomalous (n={int((y_test==1).sum())}): mean={test_errors[y_test==1].mean():.4f}")


if __name__ == "__main__":
    main()
