"""
Run this script to build the behavioral feature table from raw events.

Usage:
    python scripts/build_features.py

Requires:
    data/raw/events.csv  (produced by scripts/generate_data.py)

Outputs:
    data/processed/features.csv
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd

from config import settings as cfg
from src.features.feature_engineering import build_features

FEATURES_PATH = os.path.join(cfg.PROCESSED_DATA_DIR, "features.csv")
ENTITY_BASELINES_PATH = os.path.join(cfg.PROCESSED_DATA_DIR, "entity_baselines.json")


def main():
    if not os.path.exists(cfg.RAW_EVENTS_PATH):
        raise FileNotFoundError(
            f"{cfg.RAW_EVENTS_PATH} not found — run scripts/generate_data.py first."
        )

    events = pd.read_csv(cfg.RAW_EVENTS_PATH)
    assert "is_anomaly" not in events.columns, "LEAKAGE: raw events file already contains labels!"

    features, profiler, entity_states = build_features(events)
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)

    # Export human-readable per-entity baseline summaries (post-full-sweep;
    # for dashboard / SOC-copilot display only — NEVER fed back as a feature).
    baselines = {eid: state.summarize() for eid, state in entity_states.items()}
    with open(ENTITY_BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)

    # ---------------- Validation summary ----------------
    print("\n" + "=" * 70)
    print("SENTINELAI — FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Rows                    : {len(features)}")
    print(f"Feature columns         : {len(features.columns) - 4}")  # minus id/entity/type/timestamp
    print(f"Entities covered        : {features['entity_id'].nunique()}")

    print("\nbaseline_source distribution (entity / entity_type / global):")
    print(features["baseline_source"].value_counts().to_string())

    print("\nSample of engineered features (first 5 rows):")
    show_cols = ["event_id", "entity_id", "hour_of_day", "time_since_last_event",
                 "new_device", "unusual_location", "geo_velocity",
                 "resource_novelty", "session_duration_deviation",
                 "cold_start_weight", "baseline_source"]
    print(features[show_cols].head(5).to_string())

    # Sanity checks
    assert "is_anomaly" not in features.columns, "LEAKAGE: is_anomaly leaked into features!"
    assert "attack_type" not in features.columns, "LEAKAGE: attack_type leaked into features!"
    assert features["event_id"].is_unique, "Duplicate event_id in features table!"
    assert not features.isnull().values.any(), "Unexpected NaNs in feature table!"
    assert features["cold_start_weight"].between(0, 1).all(), "cold_start_weight out of [0,1] range!"

    print("\n[OK] No ground-truth leakage into feature columns.")
    print("[OK] event_id is unique.")
    print("[OK] No NaNs present.")
    print("[OK] cold_start_weight within [0, 1].")
    print("=" * 70)
    print(f"\nWrote: {FEATURES_PATH}")
    print(f"Wrote: {ENTITY_BASELINES_PATH}")


if __name__ == "__main__":
    main()
