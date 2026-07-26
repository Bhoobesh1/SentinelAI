"""
SentinelAI — Hybrid anomaly score + dynamic risk engine.

Two distinct outputs per event, computed by `score_events`:

1. hybrid_anomaly_score (0-1): a PURELY STATISTICAL/ML combination of
   sequence reconstruction error + entity behavioral deviation +
   novelty signals. The classifier never participates here -- this is
   "how unusual is this event," independent of what we think it is.
   (Spec section 13.)

2. risk_score (0-100, bounded by construction): the analyst-facing
   score, broken into 5 named, independently-capped components so the
   total can never exceed 100 regardless of how extreme any individual
   signal gets. (Spec section 15.) Components:

       sequence_anomaly_component    (max 30) <- autoencoder error vs threshold
       behavior_deviation_component  (max 25) <- session z-score, resource novelty, unusual location
       attack_confidence_component   (max 20) <- XGBoost 1 - P(normal)
       device_novelty_component      (max 15) <- new_device, device_change_frequency
       historical_context_component (max 10) <- cold-start uncertainty + geo-velocity severity
                                                 (reserves room for attack-chain evidence, Stage 9)

Risk scores are NOT probabilities -- they are a weighted, capped
priority signal for analyst triage.
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg
from src.models.autoencoder import apply_log1p, reconstruction_errors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel_ai.risk_engine")


# ---------------------------------------------------------------------------
# Sequence reconstruction error for EVERY event (not just a train/val/test split)
# ---------------------------------------------------------------------------
def compute_sequence_errors(features_df: pd.DataFrame, ae_model, ae_scaler, ae_meta: dict) -> pd.Series:
    """Returns a pd.Series of reconstruction error indexed by event_id.
    Events without enough prior history for a full window (an entity's
    first `sequence_length - 1` events) get NaN -- there simply isn't a
    sequence to reconstruct yet."""
    seq_len = ae_meta["sequence_length"]
    feature_cols = ae_meta["feature_cols"]

    df = apply_log1p(features_df, ae_meta["log1p_columns"])
    df = df.sort_values("timestamp")

    errors_by_event = {}
    for entity_id, g in df.groupby("entity_id"):
        g = g.sort_values("timestamp")
        if len(g) < seq_len:
            continue
        feats = g[feature_cols].values.astype(np.float32)
        event_ids = g["event_id"].values
        windows = []
        anchors = []
        for i in range(seq_len - 1, len(g)):
            windows.append(feats[i - seq_len + 1: i + 1])
            anchors.append(event_ids[i])
        X = np.stack(windows)
        X_scaled = ae_scaler.transform(X.reshape(-1, len(feature_cols))).reshape(X.shape).astype(np.float32)
        errs = reconstruction_errors(ae_model, X_scaled)
        for eid, err in zip(anchors, errs):
            errors_by_event[eid] = float(err)

    result = pd.Series(errors_by_event, name="sequence_reconstruction_error")
    result.index.name = "event_id"
    return result


# ---------------------------------------------------------------------------
# XGBoost classifier probabilities for every event
# ---------------------------------------------------------------------------
def compute_classifier_predictions(features_df: pd.DataFrame, booster, encoder, meta: dict) -> pd.DataFrame:
    import xgboost as xgb

    feature_cols = meta["feature_cols"]
    X = features_df[feature_cols].values
    dmatrix = xgb.DMatrix(X)
    proba = booster.predict(dmatrix)  # (n, num_class)

    classes = list(encoder.classes_)
    normal_idx = classes.index("normal")
    predicted_idx = np.argmax(proba, axis=1)
    predicted_class = encoder.inverse_transform(predicted_idx)
    attack_confidence = 1.0 - proba[:, normal_idx]  # P(any attack) = 1 - P(normal)

    out = pd.DataFrame({
        "event_id": features_df["event_id"].values,
        "predicted_attack_type": predicted_class,
        "attack_confidence": attack_confidence,
        "p_normal": proba[:, normal_idx],
    })
    return out


# ---------------------------------------------------------------------------
# Hybrid anomaly score (0-1, spec section 13)
# ---------------------------------------------------------------------------
def compute_hybrid_anomaly_score(row: pd.Series) -> float:
    # Sequence signal: 0 if no context yet, else ratio of error to threshold, capped at 1
    if pd.isna(row.get("sequence_reconstruction_error")):
        seq_norm = 0.0
    else:
        seq_norm = min(1.0, row["sequence_reconstruction_error"] / max(row["_ae_threshold"], 1e-9))

    behavior_raw = np.mean([
        min(1.0, abs(row["session_duration_deviation"]) / cfg.SESSION_Z_CAP),
        min(1.0, row["resource_novelty"]),
        float(row["unusual_location"]),
    ])

    novelty_raw = np.mean([
        float(row["new_device"]),
        float(row["unusual_location"]),
        min(1.0, row["resource_novelty"]),
    ])

    score = (cfg.HYBRID_WEIGHT_SEQUENCE * seq_norm +
             cfg.HYBRID_WEIGHT_BEHAVIOR * behavior_raw +
             cfg.HYBRID_WEIGHT_NOVELTY * novelty_raw)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Dynamic risk score (0-100, spec section 15)
# ---------------------------------------------------------------------------
def compute_risk_components(row: pd.Series) -> dict:
    # --- Sequence anomaly (max 30) ---
    if pd.isna(row.get("sequence_reconstruction_error")):
        seq_component = 0.0
        has_sequence_context = False
    else:
        ratio = row["sequence_reconstruction_error"] / max(row["_ae_threshold"], 1e-9)
        # first 20 pts fill as error approaches the threshold (ratio -> 1);
        # remaining 10 pts fill as error grows from 1x to SEQ_ERROR_RATIO_CAP-x the threshold
        below = min(1.0, ratio)
        above = min(1.0, max(0.0, ratio - 1.0) / (cfg.SEQ_ERROR_RATIO_CAP - 1.0))
        seq_component = (below * (cfg.RISK_MAX_SEQUENCE_ANOMALY * 2 / 3) +
                         above * (cfg.RISK_MAX_SEQUENCE_ANOMALY * 1 / 3))
        has_sequence_context = True

    # --- Behavior deviation (max 25) ---
    behavior_raw = np.mean([
        min(1.0, abs(row["session_duration_deviation"]) / cfg.SESSION_Z_CAP),
        min(1.0, row["resource_novelty"]),
        float(row["unusual_location"]),
    ])
    behavior_component = behavior_raw * cfg.RISK_MAX_BEHAVIOR_DEVIATION

    # --- Attack confidence (max 20) ---
    attack_component = float(row["attack_confidence"]) * cfg.RISK_MAX_ATTACK_CONFIDENCE

    # --- Device novelty (max 15) ---
    device_raw = 0.7 * float(row["new_device"]) + 0.3 * min(1.0, row["device_change_frequency"] * 3)
    device_component = min(1.0, device_raw) * cfg.RISK_MAX_DEVICE_NOVELTY

    # --- Historical context (max 10) --- cold-start uncertainty + geo-velocity severity.
    # Reserves room for attack-chain evidence to be blended in here in Stage 9.
    historical_raw = np.mean([
        1.0 - float(row["cold_start_weight"]),
        min(1.0, row["geo_velocity"] / cfg.GEO_VELOCITY_NORM_KMH),
    ])
    historical_component = historical_raw * cfg.RISK_MAX_HISTORICAL_CONTEXT

    seq_component = round(seq_component, 2)
    behavior_component = round(behavior_component, 2)
    attack_component = round(attack_component, 2)
    device_component = round(device_component, 2)
    historical_component = round(historical_component, 2)

    total = seq_component + behavior_component + attack_component + device_component + historical_component
    total = round(float(np.clip(total, 0, 100)), 2)

    return {
        "sequence_anomaly_component": seq_component,
        "behavior_deviation_component": behavior_component,
        "attack_confidence_component": attack_component,
        "device_novelty_component": device_component,
        "historical_context_component": historical_component,
        "risk_score": total,
        "has_sequence_context": has_sequence_context,
    }


def risk_level(score: float) -> str:
    for low, high, label in cfg.RISK_LEVELS:
        if low <= score <= high:
            return label
    return "CRITICAL" if score > 100 else "LOW"


# ---------------------------------------------------------------------------
# Full pipeline: score every event in the dataset
# ---------------------------------------------------------------------------
def score_events(features_df: pd.DataFrame, ae_model, ae_scaler, ae_meta: dict,
                  xgb_booster, xgb_encoder, xgb_meta: dict) -> pd.DataFrame:
    logger.info("Computing sequence reconstruction errors for all events...")
    seq_errors = compute_sequence_errors(features_df, ae_model, ae_scaler, ae_meta)

    logger.info("Computing classifier predictions for all events...")
    clf_preds = compute_classifier_predictions(features_df, xgb_booster, xgb_encoder, xgb_meta)

    df = features_df.merge(clf_preds, on="event_id", how="left")
    df = df.merge(seq_errors.reset_index(), on="event_id", how="left")
    df["_ae_threshold"] = ae_meta["threshold"]

    logger.info("Computing hybrid anomaly scores and risk breakdowns...")
    hybrid_scores = df.apply(compute_hybrid_anomaly_score, axis=1)
    risk_rows = df.apply(compute_risk_components, axis=1, result_type="expand")

    result = pd.concat([
        df[["event_id", "entity_id", "entity_type", "timestamp",
            "predicted_attack_type", "attack_confidence",
            "sequence_reconstruction_error"]],
        risk_rows,
    ], axis=1)
    result["hybrid_anomaly_score"] = hybrid_scores
    result["risk_level"] = result["risk_score"].apply(risk_level)

    return result
