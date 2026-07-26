"""
Run this script to produce the final, unbiased evaluation on the
held-out TEST split only.

Usage:
    python scripts/evaluate.py

Requires:
    data/processed/features.csv
    data/processed/risk_scores.csv
    data/ground_truth/labels.csv
    models/xgb_*

Outputs:
    reports/evaluation_report.json
    reports/alert_budget_curve.csv   (for the Stage 12 dashboard chart)
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import settings as cfg
from src.models.autoencoder import chronological_split
from src.models.classifier import load_artifacts as load_xgb_artifacts, evaluate_classifier
from src.evaluation.metrics import evaluate_anomaly_detection
from src.evaluation.alert_budget import alert_budget_table, alert_budget_curve, STANDARD_BUDGETS


def main():
    features = pd.read_csv(cfg.PROCESSED_DATA_DIR + "/features.csv")
    labels = pd.read_csv(cfg.GROUND_TRUTH_PATH)
    risk_scores = pd.read_csv(cfg.RISK_SCORES_PATH)

    df = risk_scores.merge(labels, on="event_id", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Recompute the SAME chronological split used for training, so
    # evaluation is confined strictly to held-out TEST data.
    _, _, test_df, (train_cutoff, val_cutoff) = chronological_split(df)
    print("=" * 70)
    print("SENTINELAI — FINAL EVALUATION (TEST SPLIT ONLY)")
    print("=" * 70)
    print(f"Test split: {len(test_df)} events (after {val_cutoff})")
    print(f"Actual anomalies in test  : {int(test_df['is_anomaly'].sum())} "
          f"({100 * test_df['is_anomaly'].mean():.2f}%)")

    y_true = test_df["is_anomaly"].values
    risk = test_df["risk_score"].values

    # ============================================================
    # A. ANOMALY DETECTION (binary, threshold-based + ranking-based)
    # ============================================================
    print("\n" + "-" * 70)
    print("A. ANOMALY DETECTION")
    print("-" * 70)
    anomaly_metrics = evaluate_anomaly_detection(y_true, risk)
    print(f"Operating threshold (risk_score >= {anomaly_metrics['operating_threshold']}, i.e. HIGH/CRITICAL):")
    print(f"  Precision: {anomaly_metrics['precision']:.3f}  Recall: {anomaly_metrics['recall']:.3f}  "
          f"F1: {anomaly_metrics['f1']:.3f}  FPR: {anomaly_metrics['false_positive_rate']:.4f}")
    print(f"  Confusion matrix: {anomaly_metrics['confusion_matrix']}")
    print(f"  PR-AUC (ranking-based): {anomaly_metrics['pr_auc']:.3f}")
    print(f"  ROC-AUC (ranking-based): {anomaly_metrics['roc_auc']:.3f}")
    print("  NOTE: risk_score is used here purely as a RANKING signal for PR-AUC/ROC-AUC; "
          "it is not a calibrated probability.")

    # ============================================================
    # B. ATTACK CLASSIFICATION (multiclass, fresh eval on test split)
    # ============================================================
    print("\n" + "-" * 70)
    print("B. ATTACK CLASSIFICATION")
    print("-" * 70)
    xgb_booster, xgb_encoder, xgb_meta = load_xgb_artifacts()
    feature_cols = xgb_meta["feature_cols"]

    test_features = features.merge(labels, on="event_id", how="inner")
    test_features["timestamp"] = pd.to_datetime(test_features["timestamp"])
    test_features = test_features[test_features["event_id"].isin(test_df["event_id"])]

    X_test = test_features[feature_cols].values
    y_test_clf = xgb_encoder.transform(test_features["attack_type"].values)
    clf_metrics = evaluate_classifier(xgb_booster, X_test, y_test_clf, xgb_encoder)
    print(clf_metrics["report_str"])
    print(f"Macro F1: {clf_metrics['macro_f1']:.3f}  |  Weighted F1: {clf_metrics['weighted_f1']:.3f}")
    print("(Accuracy is intentionally not reported as a headline metric -- with ~98% normal "
          "events, accuracy would be misleadingly high regardless of attack detection quality.)")

    # ============================================================
    # C. TOP-K ANALYST ALERT BUDGET  (the headline evaluation feature)
    # ============================================================
    print("\n" + "-" * 70)
    print("C. ANALYST ALERT BUDGET (Precision@K / Recall@K)")
    print("-" * 70)
    budget_table = alert_budget_table(y_true, risk, STANDARD_BUDGETS)
    print(budget_table.to_string(index=False))

    budget_curve = alert_budget_curve(y_true, risk)
    curve_path = os.path.join(cfg.REPORTS_DIR, "alert_budget_curve.csv")
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    budget_curve.to_csv(curve_path, index=False)

    # ============================================================
    # Save full report
    # ============================================================
    report = {
        "test_split_size": len(test_df),
        "test_actual_anomalies": int(test_df["is_anomaly"].sum()),
        "anomaly_detection": anomaly_metrics,
        "attack_classification": {
            "macro_f1": clf_metrics["macro_f1"],
            "weighted_f1": clf_metrics["weighted_f1"],
            "per_class": clf_metrics["per_class"],
            "confusion_matrix": clf_metrics["confusion_matrix"],
            "confusion_matrix_labels": clf_metrics["confusion_matrix_labels"],
        },
        "alert_budget": budget_table.to_dict(orient="records"),
    }
    report_path = os.path.join(cfg.REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ---------------- Sanity checks ----------------
    assert 0 <= anomaly_metrics["precision"] <= 1
    assert 0 <= anomaly_metrics["recall"] <= 1
    for row in budget_table.to_dict(orient="records"):
        assert 0 <= row["precision_at_k"] <= 1
        assert 0 <= row["recall_at_k"] <= 1
    # Recall must be non-decreasing as budget grows (monotonicity sanity check)
    recalls = budget_table["recall_at_k"].values
    assert np.all(np.diff(recalls) >= -1e-9), "Recall@K should never decrease as budget grows!"

    print("\n[OK] All metrics bounded in [0, 1].")
    print("[OK] Recall@K is monotonically non-decreasing as budget increases.")
    print("=" * 70)
    print(f"\nWrote: {report_path}")
    print(f"Wrote: {curve_path}")


if __name__ == "__main__":
    main()
