"""
Run this script to compute hybrid anomaly scores and dynamic risk scores
for every event in the dataset.

Usage:
    python scripts/score_events.py

Requires:
    data/processed/features.csv
    data/ground_truth/labels.csv   (used only for a sanity-check summary,
                                     NEVER as a scoring input)
    models/autoencoder.* , models/xgb_*  (from Stages 4 and 5)

Outputs:
    data/processed/risk_scores.csv
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import settings as cfg
from src.models.autoencoder import load_artifacts as load_ae_artifacts
from src.models.classifier import load_artifacts as load_xgb_artifacts
from src.models.risk_engine import score_events


def main():
    features = pd.read_csv(cfg.PROCESSED_DATA_DIR + "/features.csv")
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    labels = pd.read_csv(cfg.GROUND_TRUTH_PATH)

    print("=" * 70)
    print("SENTINELAI — HYBRID ANOMALY + RISK ENGINE")
    print("=" * 70)
    print("Loading trained model artifacts...")
    ae_model, ae_scaler, ae_meta = load_ae_artifacts()
    xgb_booster, xgb_encoder, xgb_meta = load_xgb_artifacts()

    risk_df = score_events(features, ae_model, ae_scaler, ae_meta, xgb_booster, xgb_encoder, xgb_meta)

    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    risk_df.to_csv(cfg.RISK_SCORES_PATH, index=False)

    # ---------------- Validation / sanity summary (uses labels for a
    # SANITY CHECK PRINTOUT ONLY -- ground truth is never fed into scoring) ----------------
    merged = risk_df.merge(labels, on="event_id", how="inner")

    print(f"\nTotal events scored       : {len(risk_df)}")
    print(f"Events with sequence context: {int(risk_df['has_sequence_context'].sum())} "
          f"({100*risk_df['has_sequence_context'].mean():.1f}%)")

    print("\nRisk level distribution:")
    print(risk_df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).to_string())

    print("\nMean risk_score by ground-truth label (sanity check -- higher for anomalies is good):")
    print(merged.groupby("is_anomaly")["risk_score"].agg(["mean", "median", "count"]).to_string())

    print("\nMean risk_score by attack_type:")
    print(merged.groupby("attack_type")["risk_score"].mean().sort_values(ascending=False).round(2).to_string())

    print("\nSample CRITICAL alert (highest risk_score):")
    top = risk_df.sort_values("risk_score", ascending=False).iloc[0]
    print(f"  event_id: {top['event_id']}  entity: {top['entity_id']}")
    print(f"  risk_score: {top['risk_score']} ({top['risk_level']})")
    print(f"  predicted_attack_type: {top['predicted_attack_type']} (confidence={top['attack_confidence']:.3f})")
    print(f"  components -> sequence={top['sequence_anomaly_component']}, "
          f"behavior={top['behavior_deviation_component']}, attack={top['attack_confidence_component']}, "
          f"device={top['device_novelty_component']}, historical={top['historical_context_component']}")

    # Sanity assertions
    assert risk_df["risk_score"].between(0, 100).all(), "risk_score out of [0, 100] bounds!"
    assert risk_df["hybrid_anomaly_score"].between(0, 1).all(), "hybrid_anomaly_score out of [0, 1] bounds!"
    component_sum = (risk_df["sequence_anomaly_component"] + risk_df["behavior_deviation_component"] +
                      risk_df["attack_confidence_component"] + risk_df["device_novelty_component"] +
                      risk_df["historical_context_component"])
    assert np.allclose(component_sum, risk_df["risk_score"], atol=0.02), "Components don't sum to risk_score!"

    print("\n[OK] risk_score bounded in [0, 100].")
    print("[OK] hybrid_anomaly_score bounded in [0, 1].")
    print("[OK] Risk components sum exactly to risk_score for every event.")
    print("=" * 70)
    print(f"\nWrote: {cfg.RISK_SCORES_PATH}")


if __name__ == "__main__":
    main()
