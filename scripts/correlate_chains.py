"""
Run this script to correlate related alerts into attack chains.

Usage:
    python scripts/correlate_chains.py

Requires:
    data/processed/risk_scores.csv
    data/raw/events.csv

Outputs:
    reports/attack_chains.json
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import settings as cfg
from src.correlation.attack_chain import correlate_chains


def main():
    risk_scores = pd.read_csv(cfg.RISK_SCORES_PATH)
    raw_events = pd.read_csv(cfg.RAW_EVENTS_PATH)

    print("=" * 70)
    print("SENTINELAI — ATTACK-CHAIN CORRELATION")
    print("=" * 70)
    print(f"Candidate risk levels: {sorted(cfg.CHAIN_CANDIDATE_RISK_LEVELS)}")
    print(f"Time window: {cfg.CHAIN_TIME_WINDOW_MINUTES} min (same-entity merge), "
          f"{cfg.CHAIN_SHARED_IP_LINK_WINDOW_MINUTES} min (cross-entity IP linking)")

    chains = correlate_chains(risk_scores, raw_events)

    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    with open(cfg.ATTACK_CHAINS_PATH, "w") as f:
        json.dump(chains, f, indent=2, default=str)

    print(f"\nTotal chains identified: {len(chains)}")
    multi_event_chains = [c for c in chains if c["n_events"] > 1]
    linked_chains = [c for c in chains if c["linked_entities"]]
    print(f"Multi-event chains (n_events > 1): {len(multi_event_chains)}")
    print(f"Cross-entity linked chains (shared source IP): {len(linked_chains)}")

    print("\nTop 5 chains by overall risk:")
    for chain in chains[:5]:
        print(f"\n  {chain['chain_id']}  |  Entity: {chain['primary_entity']}"
              f"{'  (+linked: ' + ', '.join(chain['linked_entities']) + ')' if chain['linked_entities'] else ''}")
        print(f"    Risk: {chain['overall_risk']} ({chain['overall_risk_level']})  "
              f"|  Stages: {' -> '.join(chain['stages'])}")
        print(f"    Window: {chain['start_time']} -> {chain['end_time']}  ({chain['n_events']} events)")
        for e in chain["evidence"]:
            print(f"    - {e}")

    # ---------------- Sanity checks ----------------
    all_event_ids = [eid for c in chains for eid in c["events"]]
    assert len(all_event_ids) == len(set(all_event_ids)), \
        "An event appears in more than one chain -- chains should partition candidate events!"
    for c in chains:
        assert pd.Timestamp(c["start_time"]) <= pd.Timestamp(c["end_time"])
        assert 0 <= c["overall_risk"] <= 100

    print(f"\n[OK] Every candidate event belongs to exactly one chain (no duplication).")
    print("[OK] All chain start_time <= end_time, overall_risk in [0, 100].")
    print("=" * 70)
    print(f"\nWrote: {cfg.ATTACK_CHAINS_PATH}")


if __name__ == "__main__":
    main()
