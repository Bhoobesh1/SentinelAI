"""
SentinelAI — Concept drift detection.

Deliberately separate from attack classification. `insider_drift` in
Stage 5 is a CLASSIFIED ATTACK TYPE (the model recognizes the pattern of
gradual privilege/resource creep as resembling a known attack shape).
THIS module answers a different question: "is this entity's behavioral
baseline genuinely, sustainedly shifting over time, regardless of
whether that shift turns out to be malicious?" It exists to give an
analyst false-positive-tuning context -- e.g. "this looks like a role
change, not an attack" -- without silently suppressing real threats
that happen to resemble drift.

Method (day-level, causal, no ML model):
  1. An entity's first DRIFT_REFERENCE_DAYS of activity establishes a
     frozen REFERENCE baseline (typical hour, device set, resource-
     sensitivity mix, mean session duration).
  2. Each subsequent day, a rolling DRIFT_RECENT_WINDOW_DAYS window is
     compared against that reference across the same four dimensions,
     producing a composite drift score in [0, 1].
  3. The daily composite score is EWMA-smoothed over calendar days, so
     ONE unusual day never triggers a drift flag on its own -- per the
     spec requirement to never update a profile off a single event.
  4. "Sustained drift" is only confirmed once the smoothed score has
     stayed above DRIFT_SCORE_THRESHOLD for DRIFT_MIN_CONSECUTIVE_DAYS
     consecutive days.

Everything here is computed causally: a day's drift score only ever
uses events at or before that day.
"""

import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg

RESOURCE_SENSITIVITY_SCORE = {"low": 0.0, "medium": 0.5, "high": 1.0}


def _circular_mean_hour(hours: pd.Series) -> float:
    if len(hours) == 0:
        return 0.0
    angles = hours.values / 24 * 2 * math.pi
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return (math.atan2(s, c) / (2 * math.pi) * 24) % 24


def _circular_hour_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2) % 24
    return min(diff, 24 - diff)


def _day_aggregate(day_events: pd.DataFrame) -> dict:
    hours = day_events["timestamp"].dt.hour + day_events["timestamp"].dt.minute / 60.0
    sensitivity = day_events["resource_accessed"].map(
        lambda r: RESOURCE_SENSITIVITY_SCORE.get(cfg.RESOURCES.get(r, "low"), 0.0)
    )
    return {
        "typical_hour": _circular_mean_hour(hours),
        "devices": set(day_events["device_fingerprint"].unique()),
        "mean_sensitivity": float(sensitivity.mean()) if len(sensitivity) else 0.0,
        "mean_session_duration": float(day_events["session_duration"].mean()),
        "n_events": len(day_events),
    }


def _aggregate_window(day_aggregates: list) -> dict:
    """Combine several daily aggregates into one window-level aggregate,
    weighted by each day's event count."""
    total_events = sum(d["n_events"] for d in day_aggregates) or 1
    all_devices = set()
    for d in day_aggregates:
        all_devices |= d["devices"]

    # Weighted circular mean of hours across days
    sin_sum = sum(math.sin(d["typical_hour"] / 24 * 2 * math.pi) * d["n_events"] for d in day_aggregates)
    cos_sum = sum(math.cos(d["typical_hour"] / 24 * 2 * math.pi) * d["n_events"] for d in day_aggregates)
    typical_hour = (math.atan2(sin_sum, cos_sum) / (2 * math.pi) * 24) % 24 if total_events else 0.0

    mean_sensitivity = sum(d["mean_sensitivity"] * d["n_events"] for d in day_aggregates) / total_events
    mean_session = sum(d["mean_session_duration"] * d["n_events"] for d in day_aggregates) / total_events

    return {
        "typical_hour": typical_hour,
        "devices": all_devices,
        "mean_sensitivity": mean_sensitivity,
        "mean_session_duration": mean_session,
        "n_events": total_events,
    }


SHRINKAGE_K = 10  # equivalent "prior sample size" pulling individual variance estimates toward the pooled entity-type average


def _reference_daily_series(reference_daily_aggregates: list) -> dict:
    """Per-day values during the reference period, used to estimate each
    entity's OWN natural day-to-day variability -- the yardstick a later
    shift gets measured against, rather than a fixed raw threshold."""
    hours = [d["typical_hour"] for d in reference_daily_aggregates]
    sensitivities = [d["mean_sensitivity"] for d in reference_daily_aggregates]
    sessions = [d["mean_session_duration"] for d in reference_daily_aggregates]
    return {"hours": hours, "sensitivities": sensitivities, "sessions": sessions}


def _hour_dispersion(hours: list, mean_hour: float) -> float:
    """Approximate circular standard deviation (in hours) of a list of
    hour-of-day values around a circular mean."""
    if len(hours) < 2:
        return None
    diffs = [_circular_hour_distance(h, mean_hour) for h in hours]
    return float(np.std(diffs))


def compute_pooled_type_variances(raw_events: pd.DataFrame) -> dict:
    """First pass over all entities: compute each entity's individual
    reference-period variance for hour/sensitivity/session, then pool by
    entity_type (average variance across entities of that type). A single
    entity's 10-day reference sample is a noisy variance estimate --
    by chance, roughly half of entities will have their TRUE variance
    underestimated from such a small sample, which would otherwise cause
    permanent spurious 'drift' later purely from that bad luck. Pooling
    across all entities of the same type gives a far more stable prior,
    mirroring the same entity -> entity_type hierarchy used for cold-start
    baselines in Stage 3."""
    individual_stats = {}  # entity_id -> {"entity_type":.., "hour_std":.., "sensitivity_std":.., "session_std":.., "reference_mean_session":..}

    for entity_id, g in raw_events.groupby("entity_id"):
        g = g.sort_values("timestamp").copy()
        g["day"] = g["timestamp"].dt.date
        sorted_days = sorted(g["day"].unique())
        if len(sorted_days) < cfg.DRIFT_REFERENCE_DAYS + 1:
            continue
        reference_days = sorted_days[:cfg.DRIFT_REFERENCE_DAYS]
        daily_aggs = [_day_aggregate(g[g["day"] == d]) for d in reference_days]
        series = _reference_daily_series(daily_aggs)
        ref_window = _aggregate_window(daily_aggs)

        hour_std = _hour_dispersion(series["hours"], ref_window["typical_hour"])
        sensitivity_std = float(np.std(series["sensitivities"])) if len(series["sensitivities"]) > 1 else None
        session_std = float(np.std(series["sessions"])) if len(series["sessions"]) > 1 else None

        individual_stats[entity_id] = {
            "entity_type": g["entity_type"].iloc[0],
            "hour_std": hour_std, "sensitivity_std": sensitivity_std, "session_std": session_std,
            "reference_mean_session": ref_window["mean_session_duration"],
        }

    # Pool by entity_type (mean of individual stds across that type, ignoring None)
    pooled = {}
    for etype in set(s["entity_type"] for s in individual_stats.values()):
        type_stats = [s for s in individual_stats.values() if s["entity_type"] == etype]
        pooled[etype] = {
            "hour_std": float(np.mean([s["hour_std"] for s in type_stats if s["hour_std"] is not None]) or 1.0),
            "sensitivity_std": float(np.mean([s["sensitivity_std"] for s in type_stats if s["sensitivity_std"] is not None]) or 0.05),
            "session_std_ratio": float(np.mean([
                s["session_std"] / max(s["reference_mean_session"], 1e-6)
                for s in type_stats if s["session_std"] is not None
            ]) or 0.15),
        }

    return {"individual": individual_stats, "pooled": pooled}


def _shrunk_std(individual_std, pooled_std, n_ref_days: int, k: int = SHRINKAGE_K) -> float:
    """Blend an entity's own (noisy, small-sample) std toward the more
    stable entity-type pooled std, weighted by how much individual data
    we actually have vs. the shrinkage prior strength k."""
    if individual_std is None:
        return pooled_std
    return (n_ref_days * individual_std + k * pooled_std) / (n_ref_days + k)


def _composite_drift_score(reference: dict, recent: dict, stds: dict) -> dict:
    """Each continuous component is expressed as a z-score using the
    (shrinkage-stabilized) std values in `stds` -- see
    compute_pooled_type_variances / _shrunk_std for why raw per-entity
    reference stds alone are unreliable at this sample size."""
    Z_CAP = 3.0  # z-scores are clipped here before normalizing to [0, 1]

    raw_hour_diff = _circular_hour_distance(reference["typical_hour"], recent["typical_hour"])
    raw_resource_diff = abs(recent["mean_sensitivity"] - reference["mean_sensitivity"])
    raw_session_diff = abs(recent["mean_session_duration"] - reference["mean_session_duration"])

    hour_component = min(Z_CAP, raw_hour_diff / max(stds["hour_std"], 0.5)) / Z_CAP
    resource_component = min(Z_CAP, raw_resource_diff / max(stds["sensitivity_std"], 0.02)) / Z_CAP
    session_component = min(Z_CAP, raw_session_diff / max(stds["session_std"], 1e-6)) / Z_CAP

    if len(recent["devices"]) == 0:
        device_component = 0.0
    else:
        novel_devices = recent["devices"] - reference["devices"]
        device_component = len(novel_devices) / len(recent["devices"])

    w = cfg.DRIFT_COMPONENT_WEIGHTS
    composite = (w["hour"] * hour_component + w["device"] * device_component +
                 w["resource"] * resource_component + w["session"] * session_component)

    return {
        "composite": float(np.clip(composite, 0, 1)),
        "hour_component": round(hour_component, 4),
        "device_component": round(device_component, 4),
        "resource_component": round(resource_component, 4),
        "session_component": round(session_component, 4),
    }


def compute_entity_drift_timeline(entity_events: pd.DataFrame, pooled_stats: dict = None) -> pd.DataFrame:
    """Causal, day-by-day drift timeline for ONE entity's events (already
    filtered to a single entity_id). Returns a DataFrame with one row per
    active day: day, composite score, component breakdown, ewma score,
    and sustained_drift flag.

    `pooled_stats` (from compute_pooled_type_variances) is used to shrink
    this entity's own noisy small-sample reference variance toward a more
    stable entity-type-level estimate. If not provided, falls back to
    fixed floor values (less reliable, but keeps this function usable
    standalone)."""
    entity_events = entity_events.sort_values("timestamp").copy()
    entity_events["day"] = entity_events["timestamp"].dt.date
    entity_id = entity_events["entity_id"].iloc[0]
    entity_type = entity_events["entity_type"].iloc[0]

    daily_groups = {day: g for day, g in entity_events.groupby("day")}
    sorted_days = sorted(daily_groups.keys())
    if len(sorted_days) < cfg.DRIFT_REFERENCE_DAYS + 1:
        return pd.DataFrame()  # not enough history to establish a reference baseline yet

    daily_aggregates = {day: _day_aggregate(daily_groups[day]) for day in sorted_days}

    reference_days = sorted_days[:cfg.DRIFT_REFERENCE_DAYS]
    reference = _aggregate_window([daily_aggregates[d] for d in reference_days])
    reference_series = _reference_daily_series([daily_aggregates[d] for d in reference_days])

    if pooled_stats is not None and entity_type in pooled_stats["pooled"]:
        individual = pooled_stats["individual"].get(entity_id, {})
        pooled = pooled_stats["pooled"][entity_type]
        n_ref = cfg.DRIFT_REFERENCE_DAYS
        stds = {
            "hour_std": _shrunk_std(individual.get("hour_std"), pooled["hour_std"], n_ref),
            "sensitivity_std": _shrunk_std(individual.get("sensitivity_std"), pooled["sensitivity_std"], n_ref),
            "session_std": _shrunk_std(
                individual.get("session_std"),
                pooled["session_std_ratio"] * max(reference["mean_session_duration"], 1e-6),
                n_ref,
            ),
        }
    else:
        stds = {"hour_std": 0.75, "sensitivity_std": 0.04,
                "session_std": 0.15 * max(reference["mean_session_duration"], 1e-6)}

    rows = []
    ewma = None
    consecutive_elevated = 0
    for i, day in enumerate(sorted_days[cfg.DRIFT_REFERENCE_DAYS:], start=cfg.DRIFT_REFERENCE_DAYS):
        window_days = sorted_days[max(cfg.DRIFT_REFERENCE_DAYS, i - cfg.DRIFT_RECENT_WINDOW_DAYS + 1): i + 1]
        recent = _aggregate_window([daily_aggregates[d] for d in window_days])

        scores = _composite_drift_score(reference, recent, stds)
        composite = scores["composite"]

        ewma = composite if ewma is None else cfg.DRIFT_EWMA_ALPHA * composite + (1 - cfg.DRIFT_EWMA_ALPHA) * ewma
        consecutive_elevated = consecutive_elevated + 1 if ewma > cfg.DRIFT_SCORE_THRESHOLD else 0
        sustained = consecutive_elevated >= cfg.DRIFT_MIN_CONSECUTIVE_DAYS

        rows.append({
            "day": str(day),
            "composite_score": round(composite, 4),
            "ewma_score": round(ewma, 4),
            "consecutive_elevated_days": consecutive_elevated,
            "sustained_drift_detected": sustained,
            **{k: v for k, v in scores.items() if k != "composite"},
        })

    return pd.DataFrame(rows)


def detect_drift_for_all_entities(raw_events: pd.DataFrame) -> tuple:
    """Runs the drift timeline for every entity. Two passes: (1) compute
    pooled entity-type variance estimates across the whole population,
    (2) compute each entity's timeline using those shrunk std estimates.
    Returns (full_timeline_df, summary_dict) where summary lists which
    entities show sustained drift and on what day it was first confirmed."""
    pooled_stats = compute_pooled_type_variances(raw_events)

    all_rows = []
    summary = {}
    for entity_id, g in raw_events.groupby("entity_id"):
        timeline = compute_entity_drift_timeline(g, pooled_stats)
        if len(timeline) == 0:
            summary[entity_id] = {"status": "insufficient_history"}
            continue
        timeline.insert(0, "entity_id", entity_id)
        all_rows.append(timeline)

        sustained_days = timeline[timeline["sustained_drift_detected"]]
        if len(sustained_days) > 0:
            first_day = sustained_days.iloc[0]["day"]
            summary[entity_id] = {
                "status": "sustained_drift_detected",
                "first_confirmed_day": first_day,
                "final_ewma_score": float(timeline.iloc[-1]["ewma_score"]),
            }
        else:
            summary[entity_id] = {
                "status": "no_sustained_drift",
                "final_ewma_score": float(timeline.iloc[-1]["ewma_score"]),
            }

    full_timeline = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return full_timeline, summary
