"""
Run this script to simulate a live event stream being scored in
near-real-time, one event at a time -- proving the architecture is
streaming-capable, not just batch-only.

Usage:
    python scripts/simulate_streaming.py [--limit N] [--speed X]

    --limit N   only replay the first N events (default: all)
    --speed X   simulated playback speed multiplier for the printed
                "elapsed simulated time" (does not affect actual
                processing time, which is measured for real)

Requires:
    data/raw/events.csv
    models/autoencoder.*, models/xgb_*
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import settings as cfg
from src.models.autoencoder import load_artifacts as load_ae_artifacts
from src.models.classifier import load_artifacts as load_xgb_artifacts
from src.streaming.scorer import StreamingScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only replay the first N events")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulated playback speed multiplier")
    args = parser.parse_args()

    print("=" * 70)
    print("SENTINELAI — LIVE STREAMING SIMULATION")
    print("=" * 70)
    print("Loading trained model artifacts...")
    ae_model, ae_scaler, ae_meta = load_ae_artifacts()
    xgb_booster, xgb_encoder, xgb_meta = load_xgb_artifacts()

    events = pd.read_csv(cfg.RAW_EVENTS_PATH)
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events = events.sort_values("timestamp").reset_index(drop=True)
    if args.limit:
        events = events.head(args.limit)

    print(f"Replaying {len(events)} events chronologically, one at a time "
          f"(as if consumed from a live Kafka topic)...\n")

    scorer = StreamingScorer(ae_model, ae_scaler, ae_meta, xgb_booster, xgb_encoder, xgb_meta)

    alerts_raised = 0
    for _, event in events.iterrows():
        result = scorer.score_event(event.to_dict())

        if result["risk_level"] in ("HIGH", "CRITICAL"):
            alerts_raised += 1
            print(f"[ALERT] t={result['timestamp']} | {result['event_id']} | {result['entity_id']} | "
                  f"risk={result['risk_score']:.1f} ({result['risk_level']}) | "
                  f"{result['predicted_attack_type']} | "
                  f"scored in {result['_processing_latency_ms']:.2f}ms")

        if scorer.events_processed % 5000 == 0:
            print(f"  ... {scorer.events_processed} events processed so far ...")

    summary = scorer.latency_summary()
    print("\n" + "=" * 70)
    print("STREAMING PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Events processed      : {summary['events_processed']}")
    print(f"Alerts raised         : {alerts_raised}")
    print(f"Mean latency/event    : {summary['mean_ms']:.2f} ms")
    print(f"P50 latency           : {summary['p50_ms']:.2f} ms")
    print(f"P95 latency           : {summary['p95_ms']:.2f} ms")
    print(f"P99 latency           : {summary['p99_ms']:.2f} ms")
    print(f"Max latency           : {summary['max_ms']:.2f} ms")
    print(f"Single-core throughput: {summary['throughput_events_per_sec_single_core']:.1f} events/sec")
    print("=" * 70)
    print("\nNote: this is a single Python process, single CPU core, using Keras'")
    print("eager-mode inference (no dedicated model-serving layer, no batching,")
    print("no horizontal scaling). See reports/real_time_architecture.md for how")
    print("this maps to a production streaming deployment and its scalability.")


if __name__ == "__main__":
    main()
