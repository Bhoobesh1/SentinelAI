"""
Run this script to generate the SentinelAI synthetic dataset.

Usage:
    python scripts/generate_data.py

Outputs:
    data/raw/events.csv              (no ground-truth columns)
    data/ground_truth/labels.csv     (event_id, is_anomaly, attack_type)
    data/processed/entity_profiles.json
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as cfg
from src.data.generator import SyntheticDataGenerator


def main():
    os.makedirs(cfg.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(cfg.GROUND_TRUTH_DIR, exist_ok=True)
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)

    gen = SyntheticDataGenerator(seed=cfg.RANDOM_SEED)
    raw_events, ground_truth = gen.generate()

    raw_events.to_csv(cfg.RAW_EVENTS_PATH, index=False)
    ground_truth.to_csv(cfg.GROUND_TRUTH_PATH, index=False)
    gen.export_entity_profiles(cfg.ENTITY_PROFILES_PATH)

    # ---------------- Validation summary ----------------
    print("\n" + "=" * 70)
    print("SENTINELAI — DATA GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total events generated       : {len(raw_events)}")
    print(f"Total entities                : {raw_events['entity_id'].nunique()}")
    print(f"Date range                    : {raw_events['timestamp'].min()} -> {raw_events['timestamp'].max()}")

    n_anom = int(ground_truth["is_anomaly"].sum())
    pct_anom = 100 * n_anom / len(ground_truth)
    print(f"\nAnomalous events              : {n_anom} ({pct_anom:.2f}% of total)")
    print("\nAttack type distribution (labeled anomalies only):")
    print(ground_truth[ground_truth.is_anomaly == 1]["attack_type"].value_counts().to_string())

    # Leakage sanity checks
    assert "is_anomaly" not in raw_events.columns, "LEAKAGE: is_anomaly found in raw events!"
    assert "attack_type" not in raw_events.columns, "LEAKAGE: attack_type found in raw events!"
    assert set(raw_events["event_id"]) == set(ground_truth["event_id"]), "event_id mismatch between files!"
    assert raw_events["timestamp"].is_monotonic_increasing, "Events are not chronologically sorted!"

    print("\n[OK] No ground-truth leakage into raw events.")
    print("[OK] event_id sets match between raw events and ground truth.")
    print("[OK] Events are chronologically sorted (required for later leakage-safe splitting).")
    print("=" * 70)
    print(f"\nWrote: {cfg.RAW_EVENTS_PATH}")
    print(f"Wrote: {cfg.GROUND_TRUTH_PATH}")
    print(f"Wrote: {cfg.ENTITY_PROFILES_PATH}")


if __name__ == "__main__":
    main()
