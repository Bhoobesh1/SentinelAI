"""
Run this script to detect sustained concept drift per entity.

Usage:
    python scripts/detect_drift.py

Requires:
    data/raw/events.csv
    data/ground_truth/labels.csv  (used ONLY for a validation printout --
                                    never as an input to the drift computation)

Outputs:
    data/processed/drift_status.csv   (full per-entity per-day timeline)
    reports/drift_summary.json        (per-entity final status)
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import settings as cfg
from src.profiling.drift import detect_drift_for_all_entities


def main():
    raw_events = pd.read_csv(cfg.RAW_EVENTS_PATH)
    raw_events["timestamp"] = pd.to_datetime(raw_events["timestamp"])
    labels = pd.read_csv(cfg.GROUND_TRUTH_PATH)

    print("=" * 70)
    print("SENTINELAI — CONCEPT DRIFT DETECTION")
    print("=" * 70)
    print(f"Reference baseline window : first {cfg.DRIFT_REFERENCE_DAYS} active days per entity")
    print(f"Recent comparison window  : last {cfg.DRIFT_RECENT_WINDOW_DAYS} days")
    print(f"EWMA alpha                : {cfg.DRIFT_EWMA_ALPHA}")
    print(f"Sustained-drift threshold : EWMA > {cfg.DRIFT_SCORE_THRESHOLD} for "
          f"{cfg.DRIFT_MIN_CONSECUTIVE_DAYS}+ consecutive days")

    timeline, summary = detect_drift_for_all_entities(raw_events)

    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    timeline.to_csv(cfg.DRIFT_STATUS_PATH, index=False)
    with open(cfg.DRIFT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    n_sustained = sum(1 for s in summary.values() if s["status"] == "sustained_drift_detected")
    n_no_drift = sum(1 for s in summary.values() if s["status"] == "no_sustained_drift")
    n_insufficient = sum(1 for s in summary.values() if s["status"] == "insufficient_history")

    print(f"\nEntities with sustained drift detected : {n_sustained}")
    print(f"Entities with no sustained drift        : {n_no_drift}")
    print(f"Entities with insufficient history       : {n_insufficient}")

    print("\nEntities flagged for sustained drift:")
    for eid, s in summary.items():
        if s["status"] == "sustained_drift_detected":
            print(f"  {eid}: first confirmed on {s['first_confirmed_day']}, "
                  f"final EWMA score {s['final_ewma_score']:.3f}")

    # ---------------- Validation against ground truth (informational only) ----------------
    print("\n" + "-" * 70)
    print("VALIDATION (informational -- ground truth used only to check ourselves, "
          "never as a drift-detection input)")
    print("-" * 70)
    print("Note: this module detects SUSTAINED behavioral change from any cause, not")
    print("specifically the 'insider_drift' attack label (that narrower pattern-matching")
    print("job belongs to the Stage 5 classifier). The appropriate validation question is")
    print("'does a sustained-drift flag correlate with genuine recent anomalous activity")
    print("of ANY kind for that entity' -- not just insider_drift specifically.\n")

    validation_rows = []
    for eid, s in summary.items():
        if s["status"] == "insufficient_history":
            continue
        g = labels.merge(raw_events[["event_id", "entity_id", "timestamp"]], on="event_id")
        g = g[g["entity_id"] == eid].sort_values("timestamp")
        last_ts = g["timestamp"].max()
        recent_window = g[g["timestamp"] >= last_ts - pd.Timedelta(days=7)]
        had_recent_anomaly = bool((recent_window["attack_type"] != "normal").any())
        validation_rows.append({
            "entity_id": eid,
            "sustained_flagged": s["status"] == "sustained_drift_detected",
            "had_recent_anomaly": had_recent_anomaly,
        })
    vdf = pd.DataFrame(validation_rows)

    flagged = vdf[vdf["sustained_flagged"]]
    precision = flagged["had_recent_anomaly"].mean() if len(flagged) > 0 else float("nan")
    print(f"Entities flagged as sustained drift: {len(flagged)}")
    print(f"  Of those, how many had genuine recent anomalous activity: "
          f"{int(flagged['had_recent_anomaly'].sum())}/{len(flagged)} "
          f"(precision = {precision:.1%})" if len(flagged) > 0 else "  (none flagged)")

    not_flagged = vdf[~vdf["sustained_flagged"]]
    false_negatives = not_flagged["had_recent_anomaly"].sum()
    print(f"Entities NOT flagged but with genuine recent anomalous activity "
          f"(missed -- expected, since this is a deliberately conservative/high-precision signal): "
          f"{int(false_negatives)}/{len(not_flagged)}")

    # Narrower check specifically against insider_drift (informational context only --
    # this module isn't designed to specialize in this one attack type)
    insider_drift_entities = set(
        labels.merge(raw_events[["event_id", "entity_id"]], on="event_id")
        .query("attack_type == 'insider_drift'")["entity_id"]
        .unique()
    )
    flagged_entities = set(flagged["entity_id"])
    print(f"\n(Narrower context: of {len(insider_drift_entities)} entities with a true "
          f"insider_drift incident, {len(insider_drift_entities & flagged_entities)} were "
          f"also flagged sustained -- insider_drift is deliberately the subtlest pattern, "
          f"so this being a partial overlap rather than 100% is expected.)")

    # ---------------- Sanity checks ----------------
    if len(timeline) > 0:
        assert timeline["ewma_score"].between(0, 1).all(), "ewma_score out of [0, 1] bounds!"
        assert timeline["composite_score"].between(0, 1).all(), "composite_score out of [0, 1] bounds!"

    print("\n[OK] All drift scores bounded in [0, 1].")
    print("=" * 70)
    print(f"\nWrote: {cfg.DRIFT_STATUS_PATH}")
    print(f"Wrote: {cfg.DRIFT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
