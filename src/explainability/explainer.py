"""
SentinelAI — Explainability layer.

Two complementary explanations per alert:

1. CLASSIFIER EVIDENCE (SHAP): for the XGBoost prediction, which features
   pushed the model toward (or away from) the predicted attack type, and
   by how much. SHAP values are attributions, NOT probabilities -- we
   never describe them as such anywhere in this module's output.

2. SEQUENCE EVIDENCE: for events with autoencoder context, which specific
   behavioral features had the largest per-feature reconstruction error
   within the trailing window -- i.e. which signals the model found
   hardest to reconstruct as "normal."

All raw feature names are translated into analyst-friendly phrasing via
FEATURE_DISPLAY_NAMES before being shown to a human.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg
from src.models.autoencoder import apply_log1p

# ---------------------------------------------------------------------------
# Analyst-friendly feature translations
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    "hour_of_day": "Hour of day of this event",
    "day_of_week": "Day of the week",
    "weekend": "Occurred on a weekend",
    "time_since_last_event": "Time since this entity's previous event",
    "events_last_10m": "Number of this entity's events in the last 10 minutes",
    "events_last_1h": "Number of this entity's events in the last hour",
    "auth_failure_rate": "This entity's overall authentication failure rate",
    "failed_attempts_last_10m": "Failed login attempts in the last 10 minutes",
    "unusual_auth_method": "Authentication method rarely used by this entity",
    "distance_from_previous_login": "Geographic distance from the previous login",
    "geo_velocity": "Implied travel speed between consecutive logins",
    "unusual_location": "Login from a location not previously seen for this entity",
    "new_device": "Login from a device never seen before for this entity",
    "device_change_frequency": "How often this entity switches between different devices",
    "device_consistency_score": "How consistently this entity uses its most common device",
    "resource_novelty": "How rarely (or never) this entity has accessed this resource",
    "resource_access_frequency": "How often this entity normally accesses this resource",
    "sensitive_resource_access": "Access to a highly sensitive resource",
    "session_duration_deviation": "Session length compared to this entity's typical pattern",
    "command_novelty": "Use of commands not previously seen for this entity",
    "command_frequency": "How common these commands are for this entity",
    "deviation_from_entity_mean": "Overall deviation from this entity's typical behavior",
    "deviation_from_entity_std": "Recent volatility in this entity's session lengths",
    "entity_history_count": "Number of prior events on record for this entity",
    "cold_start_weight": "Confidence in this entity's behavioral baseline (low = still new)",
}


def display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# 1. Classifier evidence (SHAP)
# ---------------------------------------------------------------------------
def build_explainer(booster):
    import shap
    return shap.TreeExplainer(booster)


def explain_classifier_prediction(explainer, x_row: np.ndarray, feature_cols: list,
                                   class_idx: int, top_k: int = 5) -> list:
    """Returns the top_k features (by |SHAP value|) driving the model toward
    (or away from) `class_idx` for this single event. x_row must be shape
    (1, n_features)."""
    shap_out = explainer(x_row)
    # shap_out.values shape: (1, n_features, n_classes)
    values = np.array(shap_out.values)[0, :, class_idx]

    order = np.argsort(-np.abs(values))[:top_k]
    results = []
    for i in order:
        feature = feature_cols[i]
        shap_val = float(values[i])
        results.append({
            "feature": feature,
            "display_name": display_name(feature),
            "feature_value": float(x_row[0, i]),
            "shap_contribution": round(shap_val, 4),
            "direction": "increases" if shap_val > 0 else "decreases",
        })
    return results


# ---------------------------------------------------------------------------
# 2. Sequence evidence (per-feature autoencoder reconstruction error)
# ---------------------------------------------------------------------------
def get_window_for_event(features_df: pd.DataFrame, entity_id: str, anchor_event_id: str,
                          ae_meta: dict):
    """Rebuild the raw (log1p-applied, unscaled) trailing window ending at
    `anchor_event_id` for `entity_id`. Returns None if insufficient history."""
    seq_len = ae_meta["sequence_length"]
    feature_cols = ae_meta["feature_cols"]

    g = features_df[features_df["entity_id"] == entity_id].sort_values("timestamp").reset_index(drop=True)
    g = apply_log1p(g, ae_meta["log1p_columns"])

    if anchor_event_id not in set(g["event_id"]):
        return None
    idx = g.index[g["event_id"] == anchor_event_id][0]
    if idx < seq_len - 1:
        return None
    window = g.loc[idx - seq_len + 1: idx, feature_cols].values.astype(np.float32)
    event_ids = g.loc[idx - seq_len + 1: idx, "event_id"].values
    return window, event_ids


def explain_sequence_reconstruction(ae_model, ae_scaler, window_raw: np.ndarray,
                                     feature_cols: list, top_k: int = 3) -> list:
    """Per-feature reconstruction error (mean squared error across the
    window's timesteps) -- identifies which specific behavioral signals
    the autoencoder found hardest to reconstruct as 'normal'."""
    n_features = len(feature_cols)
    X = window_raw.reshape(1, *window_raw.shape)
    X_scaled = ae_scaler.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)

    recon = ae_model.predict(X_scaled, verbose=0)
    per_feature_error = np.mean(np.square(X_scaled - recon), axis=1)[0]  # (n_features,)

    order = np.argsort(-per_feature_error)[:top_k]
    results = []
    for i in order:
        feature = feature_cols[i]
        results.append({
            "feature": feature,
            "display_name": display_name(feature),
            "reconstruction_error": round(float(per_feature_error[i]), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Full alert explanation assembly
# ---------------------------------------------------------------------------
def explain_alert(event_id: str, features_df: pd.DataFrame, risk_row: pd.Series,
                   explainer, xgb_meta: dict, encoder,
                   ae_model=None, ae_scaler=None, ae_meta=None) -> dict:
    feature_cols = xgb_meta["feature_cols"]
    row = features_df[features_df["event_id"] == event_id].iloc[0]
    x_row = row[feature_cols].values.reshape(1, -1).astype(np.float32)

    predicted_class = risk_row["predicted_attack_type"]
    class_idx = list(encoder.classes_).index(predicted_class)

    classifier_evidence = explain_classifier_prediction(explainer, x_row, feature_cols, class_idx)

    sequence_evidence = None
    if ae_model is not None and bool(risk_row.get("has_sequence_context", False)):
        window_result = get_window_for_event(features_df, row["entity_id"], event_id, ae_meta)
        if window_result is not None:
            window_raw, _ = window_result
            sequence_evidence = explain_sequence_reconstruction(
                ae_model, ae_scaler, window_raw, ae_meta["feature_cols"]
            )

    return {
        "event_id": event_id,
        "entity_id": row["entity_id"],
        "risk_score": risk_row["risk_score"],
        "risk_level": risk_row["risk_level"],
        "predicted_attack_type": predicted_class,
        "attack_confidence": round(float(risk_row["attack_confidence"]), 3),
        "classifier_evidence": classifier_evidence,
        "sequence_evidence": sequence_evidence,
        "note": ("SHAP values are feature attributions for this classification, "
                 "not probabilities. Reconstruction error highlights which behavioral "
                 "signals looked least 'normal' in the recent event sequence."),
    }
