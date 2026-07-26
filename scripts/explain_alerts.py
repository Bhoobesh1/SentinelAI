"""
Run this script to generate SHAP-based explanations for the highest-risk
alerts (analyst-facing "why was this flagged" evidence).

Usage:
    python scripts/explain_alerts.py

Requires:
    data/processed/features.csv
    data/processed/risk_scores.csv
    models/xgb_*, models/autoencoder_*

Outputs:
    reports/alert_explanations.json
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import settings as cfg
from src.models.classifier import load_artifacts as load_xgb_artifacts
from src.models.autoencoder import load_artifacts as load_ae_artifacts
from src.explainability.explainer import build_explainer, explain_alert, display_name

TOP_N_ALERTS = 15
OUTPUT_PATH = os.path.join(cfg.REPORTS_DIR, "alert_explanations.json")


def print_human_readable(explanation: dict):
    print(f"\nALERT {explanation['event_id']}  |  Entity: {explanation['entity_id']}")
    print(f"Risk: {explanation['risk_score']}/100 ({explanation['risk_level']})  "
          f"|  Likely pattern: {explanation['predicted_attack_type']} "
          f"(confidence={explanation['attack_confidence']})")
    print("Top classifier evidence (SHAP -- attributions, not probabilities):")
    for item in explanation["classifier_evidence"]:
        sign = "+" if item["shap_contribution"] >= 0 else ""
        print(f"    {item['display_name']:<55s} {sign}{item['shap_contribution']:<8} "
              f"(value={item['feature_value']:.3f}, {item['direction']} likelihood)")
    if explanation["sequence_evidence"]:
        print("Sequence context (features hardest to reconstruct as 'normal'):")
        for item in explanation["sequence_evidence"]:
            print(f"    {item['display_name']:<55s} error={item['reconstruction_error']}")
    else:
        print("Sequence context: insufficient history for this entity yet.")


def main():
    features = pd.read_csv(cfg.PROCESSED_DATA_DIR + "/features.csv")
    risk_scores = pd.read_csv(cfg.RISK_SCORES_PATH)

    print("=" * 70)
    print("SENTINELAI — SHAP + SEQUENCE EXPLAINABILITY")
    print("=" * 70)
    print("Loading model artifacts...")
    xgb_booster, xgb_encoder, xgb_meta = load_xgb_artifacts()
    ae_model, ae_scaler, ae_meta = load_ae_artifacts()
    explainer = build_explainer(xgb_booster)

    top_alerts = risk_scores.sort_values("risk_score", ascending=False).head(TOP_N_ALERTS)
    print(f"\nExplaining top {len(top_alerts)} highest-risk alerts...\n")

    explanations = []
    for _, risk_row in top_alerts.iterrows():
        explanation = explain_alert(
            event_id=risk_row["event_id"], features_df=features, risk_row=risk_row,
            explainer=explainer, xgb_meta=xgb_meta, encoder=xgb_encoder,
            ae_model=ae_model, ae_scaler=ae_scaler, ae_meta=ae_meta,
        )
        explanations.append(explanation)

    # Print the single highest-risk alert in full, human-readable form
    print_human_readable(explanations[0])

    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(explanations, f, indent=2, default=str)

    # ---------------- Sanity checks ----------------
    for exp in explanations:
        assert len(exp["classifier_evidence"]) > 0, "No SHAP evidence produced for an alert!"
        for item in exp["classifier_evidence"]:
            assert -1000 < item["shap_contribution"] < 1000, "SHAP value looks unreasonable!"

    print(f"\n[OK] Generated explanations for {len(explanations)} alerts.")
    print("[OK] Every alert has non-empty classifier evidence.")
    print("=" * 70)
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
