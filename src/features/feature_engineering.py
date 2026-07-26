"""
SentinelAI — Behavioral feature engineering.

CRITICAL INVARIANT: every feature for event T is computed using only
events strictly BEFORE T (globally, across all entities). Running state
is read via `snapshot`/direct lookups and only updated AFTER a row's
features have been computed, so an event can never see itself or the
future.

Ground-truth labels (is_anomaly / attack_type) are never touched here —
this module only ever reads data/raw/events.csv.

Baselines with a sensible hierarchical fallback (session duration,
typical login hour, resource-access frequency) are delegated to
`src.profiling.entity_profiler.EntityProfiler`, which blends the
entity's own history with entity-type and global baselines depending on
how much history the entity has (cold-start handling). Baselines with
no meaningful cross-entity fallback (known devices, known locations,
auth-method habits, command vocabulary) stay entity-only via
`EntityRunningState`.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg
from src.data.generator import haversine_km
from src.profiling.entity_profiler import EntityProfiler, EntityRunningState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel_ai.features")

NO_PRIOR_EVENT_MINUTES = 10_080.0  # 1 week sentinel for "no previous event yet"


def compute_event_features(entity_states: dict, profiler: EntityProfiler,
                            entity_id, entity_type, ts, geo, device, method,
                            auth_success: bool, resource, duration: float, cmds: list) -> dict:
    """Compute the full causal feature set for ONE event, given the
    current (pre-this-event) entity_states dict and profiler, then commit
    this event's values into both AFTER reading -- preserving the exact
    same causality guarantee as the batch loop in build_features().

    This is the single shared implementation used by both the batch
    pipeline (build_features, looping over a static CSV) and the
    streaming demo (src/streaming/scorer.py, processing one live event
    at a time) -- there is only one feature-computation code path,
    exercised identically by both, so batch results and streaming
    results are guaranteed consistent by construction.
    """
    state = entity_states.setdefault(entity_id, EntityRunningState())
    ts = pd.Timestamp(ts)

    features = {}

    # ---- Temporal ----
    if state.last_timestamp is None:
        time_since_last = NO_PRIOR_EVENT_MINUTES
    else:
        time_since_last = (ts - state.last_timestamp).total_seconds() / 60.0
    features["time_since_last_event"] = time_since_last

    state.purge_windows(ts)
    features["events_last_10m"] = state.events_last_10m(ts)
    features["events_last_1h"] = state.events_last_1h(ts)

    # ---- Authentication ----
    features["auth_failure_rate"] = (state.n_auth_fail / state.n_events) if state.n_events else 0.0
    features["failed_attempts_last_10m"] = state.failed_attempts_last_10m(ts)
    method_freq = state.auth_method_frequency(method)
    features["unusual_auth_method"] = int(method_freq < 0.15)

    # ---- Geography ----
    if state.last_geo_location is not None:
        dist = haversine_km(state.last_geo_location, geo) if geo != state.last_geo_location else 0.0
    else:
        dist = 0.0
    features["distance_from_previous_login"] = dist
    hours_gap = max(time_since_last / 60.0, 1e-3)
    features["geo_velocity"] = dist / hours_gap
    features["unusual_location"] = int(geo not in state.seen_locations)

    # ---- Device ----
    features["new_device"] = int(device not in state.seen_devices)
    n_distinct_devices = len(state.seen_devices)
    features["device_change_frequency"] = (n_distinct_devices / state.n_events) if state.n_events else 0.0
    features["device_consistency_score"] = state.most_common_device_share()

    # ---- Hierarchical baselines (session duration, hour, resource freq) ----
    current_hour = ts.hour + ts.minute / 60.0
    snap = profiler.snapshot(entity_id, entity_type, resource, current_hour)

    features["resource_access_frequency"] = snap["resource_freq"]
    features["resource_novelty"] = 1.0 - snap["resource_freq"]
    features["sensitive_resource_access"] = 1 if cfg.RESOURCES.get(resource, "low") == "high" else 0

    if snap["session_mean"] is None or snap["session_std"] is None or snap["session_std"] < 1e-6:
        z = 0.0
    else:
        z = (duration - snap["session_mean"]) / snap["session_std"]
    features["session_duration_deviation"] = z

    features["deviation_from_entity_mean"] = 0.5 * abs(z) + 0.5 * snap["hour_deviation_norm"]
    features["cold_start_weight"] = snap["weight_entity"]
    features["entity_history_count"] = snap["entity_history_count"]
    features["baseline_source"] = snap["baseline_source"]

    # ---- Commands ----
    if cmds:
        novel = sum(1 for c in cmds if c not in state.command_vocab)
        features["command_novelty"] = novel / len(cmds)
        freqs = [state.command_frequency(c) for c in cmds]
        features["command_frequency"] = float(np.mean(freqs)) if freqs else 0.0
    else:
        features["command_novelty"] = 0.0
        features["command_frequency"] = 0.0

    # ---- Local session-duration volatility ----
    if len(state.recent_session_durations) >= 2:
        features["deviation_from_entity_std"] = float(np.std(list(state.recent_session_durations)))
    else:
        features["deviation_from_entity_std"] = 0.0

    # ---- Static (self-referential only, no cross-row dependency) ----
    features["hour_of_day"] = ts.hour
    features["day_of_week"] = ts.dayofweek
    features["weekend"] = int(ts.dayofweek >= 5)

    # ---- Commit updates AFTER reading (preserve causality) ----
    state.update(ts=ts, geo_location=geo, device=device, auth_method=method,
                 auth_success=auth_success, resource=resource, commands=cmds)
    state.recent_session_durations.append(duration)
    profiler.update(entity_id, entity_type, resource, current_hour, duration)

    return features


def build_features(events: pd.DataFrame, profiler: EntityProfiler = None):
    """Compute the full causal feature set for every event.

    Parameters
    ----------
    events : DataFrame with the raw schema written by generate_data.py
             (NO ground-truth columns present).
    profiler : optionally pass in an EntityProfiler to reuse/extend.
               If None, a fresh one is created and swept through `events`.

    Returns
    -------
    (features_df, profiler, entity_states) : the engineered feature table,
    the EntityProfiler holding final accumulated baselines, and the raw
    per-entity running states (devices/locations/resources/etc.) --
    useful for exporting human-readable entity summaries.
    """
    df = events.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["command_list"] = df["command_sequence"].fillna("").apply(
        lambda s: [c for c in s.split("|") if c]
    )

    n = len(df)
    logger.info(f"Computing causal features for {n} events across {df['entity_id'].nunique()} entities "
                f"(single global chronological pass)...")

    entity_states: dict = {}
    profiler = profiler if profiler is not None else EntityProfiler()

    ts_col = df["timestamp"].values
    entity_id_col = df["entity_id"].values
    entity_type_col = df["entity_type"].values
    geo_col = df["geo_location"].values
    device_col = df["device_fingerprint"].values
    auth_method_col = df["auth_method"].values
    auth_success_col = df["auth_success"].values
    resource_col = df["resource_accessed"].values
    duration_col = df["session_duration"].values
    commands_col = df["command_list"].values

    all_feature_rows = []
    for i in range(n):
        row_features = compute_event_features(
            entity_states=entity_states, profiler=profiler,
            entity_id=entity_id_col[i], entity_type=entity_type_col[i], ts=ts_col[i],
            geo=geo_col[i], device=device_col[i], method=auth_method_col[i],
            auth_success=bool(auth_success_col[i]), resource=resource_col[i],
            duration=float(duration_col[i]), cmds=commands_col[i],
        )
        all_feature_rows.append(row_features)

    feature_df = pd.DataFrame(all_feature_rows)
    df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    feature_cols = [
        "event_id", "entity_id", "entity_type", "timestamp",
        "hour_of_day", "day_of_week", "weekend",
        "time_since_last_event", "events_last_10m", "events_last_1h",
        "auth_failure_rate", "failed_attempts_last_10m", "unusual_auth_method",
        "distance_from_previous_login", "geo_velocity", "unusual_location",
        "new_device", "device_change_frequency", "device_consistency_score",
        "resource_novelty", "resource_access_frequency", "sensitive_resource_access",
        "session_duration_deviation",
        "command_novelty", "command_frequency",
        "deviation_from_entity_mean", "deviation_from_entity_std",
        "entity_history_count", "cold_start_weight", "baseline_source",
    ]
    result = df[feature_cols].copy()

    logger.info("Feature engineering complete.")
    logger.info(f"baseline_source distribution: {result['baseline_source'].value_counts().to_dict()}")

    return result, profiler, entity_states
