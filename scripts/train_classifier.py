"""
Run this script to train the XGBoost attack classifier.

Usage:
    python scripts/train_classifier.py

Requires:
    data/processed/features.csv   (from scripts/build_features.py)
    data/ground_truth/labels.csv  (from scripts/generate_data.py)

Outputs:
    models/xgb_classifier.json
    models/xgb_label_encoder.pkl
    models/xgb_meta.json   (feature columns, class list, test metrics)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import settings as cfg

np.random.seed(cfg.RANDOM_SEED)

from src.models.autoencoder import chronological_split
from src.models.classifier import (
    prepare_labels, train_classifier, evaluate_classifier, save_artifacts,
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
    print("SENTINELAI — XGBOOST ATTACK CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Train: {len(train_df)} events (up to {train_cutoff})")
    print(f"Val  : {len(val_df)} events (up to {val_cutoff})")
    print(f"Test : {len(test_df)} events")

    if len(train_df) and len(val_df):
        assert train_df["timestamp"].max() <= val_df["timestamp"].min(), \
            "LEAKAGE: train/val splits overlap in time!"
    if len(val_df) and len(test_df):
        assert val_df["timestamp"].max() <= test_df["timestamp"].min(), \
            "LEAKAGE: val/test splits overlap in time!"
    print("[OK] Train/val/test splits do not overlap in time.")

    feature_cols = cfg.MODEL_FEATURE_COLUMNS
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train, encoder = prepare_labels(train_df)
    y_val = encoder.transform(val_df["attack_type"].values)
    y_test = encoder.transform(test_df["attack_type"].values)

    print(f"\nClasses ({len(encoder.classes_)}): {list(encoder.classes_)}")
    print("\nTraining label distribution:")
    train_dist = train_df["attack_type"].value_counts()
    print(train_dist.to_string())

    model = train_classifier(X_train, y_train, num_classes=len(encoder.classes_))

    print("\n--- VALIDATION performance (sanity check only) ---")
    val_metrics = evaluate_classifier(model, X_val, y_val, encoder)
    print(val_metrics["report_str"])
    print(f"Macro F1: {val_metrics['macro_f1']:.3f} | Weighted F1: {val_metrics['weighted_f1']:.3f}")

    print("\n--- TEST performance (held out, unbiased) ---")
    test_metrics = evaluate_classifier(model, X_test, y_test, encoder)
    print(test_metrics["report_str"])
    print(f"Macro F1: {test_metrics['macro_f1']:.3f} | Weighted F1: {test_metrics['weighted_f1']:.3f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=test_metrics["confusion_matrix_labels"],
        columns=test_metrics["confusion_matrix_labels"],
    )
    print(cm_df.to_string())

    save_artifacts(model, encoder, feature_cols, test_metrics)

    print("\n" + "=" * 70)
    print(f"[OK] Model saved to {cfg.XGB_MODEL_PATH}")
    print(f"[OK] Label encoder saved to {cfg.XGB_LABEL_ENCODER_PATH}")
    print(f"[OK] Metadata saved to {cfg.XGB_META_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
