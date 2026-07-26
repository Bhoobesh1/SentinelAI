"""
SentinelAI — Analyst alert-budget evaluation.

SOC analysts cannot investigate every alert. This module answers:
"If analysts can only look at the top K% of events ranked by risk_score,
how many real attacks do they catch, and how much of their time is
wasted on false positives?"

This is deliberately separate from the threshold-based metrics in
metrics.py -- alert budgets are about a fixed INVESTIGATION CAPACITY
(e.g. "we can only review 1% of events per day"), not a fixed risk
threshold.
"""

import numpy as np
import pandas as pd

STANDARD_BUDGETS = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5%


def precision_recall_at_budget(y_true: np.ndarray, risk_scores: np.ndarray, budget_frac: float) -> dict:
    """Rank all events by risk_score descending; take the top `budget_frac`
    fraction; report precision/recall/TP/FP within that slice."""
    n_total = len(y_true)
    k = max(1, int(np.ceil(n_total * budget_frac)))

    order = np.argsort(-risk_scores)
    top_k_idx = order[:k]
    y_top_k = y_true[top_k_idx]

    tp = int(y_top_k.sum())
    fp = int(k - tp)
    total_anomalies = int(y_true.sum())

    precision = tp / k if k > 0 else 0.0
    recall = tp / total_anomalies if total_anomalies > 0 else 0.0

    return {
        "budget_pct": budget_frac * 100,
        "k_events": k,
        "true_positives": tp,
        "false_positives": fp,
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "total_anomalies_in_test": total_anomalies,
    }


def alert_budget_table(y_true: np.ndarray, risk_scores: np.ndarray,
                        budgets: list = STANDARD_BUDGETS) -> pd.DataFrame:
    rows = [precision_recall_at_budget(y_true, risk_scores, b) for b in budgets]
    return pd.DataFrame(rows)


def alert_budget_curve(y_true: np.ndarray, risk_scores: np.ndarray,
                        budget_range=None) -> pd.DataFrame:
    """Finer-grained curve (for plotting 'Alert Budget vs Attack Recall')
    than the four standard checkpoints -- e.g. for a Plotly chart in the
    dashboard (Stage 12)."""
    if budget_range is None:
        budget_range = np.concatenate([
            np.arange(0.001, 0.01, 0.001),
            np.arange(0.01, 0.05, 0.005),
            np.arange(0.05, 0.21, 0.01),
        ])
    rows = [precision_recall_at_budget(y_true, risk_scores, b) for b in budget_range]
    return pd.DataFrame(rows)
