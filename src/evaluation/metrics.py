"""
SentinelAI — Standard evaluation metrics.

Two separate evaluation tracks, per spec section 24:

A. ANOMALY DETECTION (binary: is_anomaly vs risk_score)
   - risk_score is used as a continuous RANKING score for PR-AUC/ROC-AUC
     (these only require an ordering, not a calibrated probability --
     risk scores are explicitly not probabilities, but they are a valid
     ranking signal).
   - A binary "flagged" decision uses the risk_level HIGH/CRITICAL
     boundary (risk_score >= 60) as the operating point, since that's
     already the natural SOC escalation threshold established in Stage 6.

B. ATTACK CLASSIFICATION (multiclass: attack_type vs predicted_attack_type)
   - Reuses the same per-class precision/recall/F1 machinery from
     Stage 5, evaluated fresh here on the test split for a
     self-contained final report.

Because the dataset is highly imbalanced, accuracy is deliberately never
reported as a headline metric.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score, roc_auc_score,
)

FLAGGED_THRESHOLD = 60.0  # risk_score >= this => HIGH or CRITICAL => "flagged"


def evaluate_anomaly_detection(y_true: np.ndarray, risk_scores: np.ndarray,
                                threshold: float = FLAGGED_THRESHOLD) -> dict:
    """y_true: 0/1 ground-truth is_anomaly. risk_scores: continuous 0-100 scores."""
    y_pred = (risk_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Threshold-independent ranking metrics (risk_score used purely as
    # an ordering signal here, not as a calibrated probability).
    pr_auc = average_precision_score(y_true, risk_scores) if len(np.unique(y_true)) > 1 else float("nan")
    roc_auc = roc_auc_score(y_true, risk_scores) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "operating_threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_flagged": int(y_pred.sum()),
        "n_total": int(len(y_true)),
        "n_actual_anomalies": int(y_true.sum()),
    }
