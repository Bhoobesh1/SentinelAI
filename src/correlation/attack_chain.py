"""
SentinelAI — Attack-chain correlation.

Deliberately deterministic (NO ML model here) -- purely time-window and
entity/resource relationship logic, per spec section 17:

    1. Within a single entity: consecutive elevated-risk events are
       merged into one chain if the gap between them is under
       CHAIN_TIME_WINDOW_MINUTES (a simple sequential merge, like
       interval-merging).
    2. Across entities: two different entities' chains are linked into
       a shared "incident_group" if they share a source_ip within
       CHAIN_SHARED_IP_LINK_WINDOW_MINUTES of each other -- the classic
       signature of one attacker (or one compromised IP) touching
       multiple accounts (credential stuffing, coordinated brute force).

Produces exactly the fields the spec asks for: chain_id, entities,
events, start_time, end_time, stages, overall_risk, evidence.
"""

import sys
import os

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg


def _build_candidates(risk_scores: pd.DataFrame, raw_events: pd.DataFrame) -> pd.DataFrame:
    """Join risk scores with the raw event context (source_ip, device,
    resource, auth_success) needed to write evidence strings, and filter
    down to events worth correlating (MEDIUM risk or above)."""
    df = risk_scores.merge(
        raw_events[["event_id", "source_ip", "device_fingerprint", "resource_accessed", "auth_success"]],
        on="event_id", how="left",
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["risk_level"].isin(cfg.CHAIN_CANDIDATE_RISK_LEVELS)].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def _merge_into_chains(entity_events: pd.DataFrame, window_minutes: int) -> list:
    """Sequential interval-merge: start a new chain whenever the gap since
    the previous candidate event (for this SAME entity) exceeds the
    window. Returns a list of DataFrames, one per chain."""
    chains = []
    current_chain_rows = []
    prev_ts = None

    for _, row in entity_events.iterrows():
        if prev_ts is not None and (row["timestamp"] - prev_ts).total_seconds() / 60.0 > window_minutes:
            chains.append(pd.DataFrame(current_chain_rows))
            current_chain_rows = []
        current_chain_rows.append(row)
        prev_ts = row["timestamp"]

    if current_chain_rows:
        chains.append(pd.DataFrame(current_chain_rows))
    return chains


def _make_evidence(chain_df: pd.DataFrame) -> list:
    """Short, deterministic evidence strings built from the raw signals
    present in the chain -- no free-text generation, no model involved."""
    evidence = []

    n_failed = int((~chain_df["auth_success"].astype(bool)).sum())
    if n_failed >= 3:
        span_seconds = (chain_df["timestamp"].max() - chain_df["timestamp"].min()).total_seconds()
        evidence.append(f"{n_failed} failed authentication attempts within "
                         f"{max(1, int(span_seconds // 60))} minute(s)")

    n_distinct_devices = chain_df["device_fingerprint"].nunique()
    if n_distinct_devices > 1:
        evidence.append(f"{n_distinct_devices} distinct devices used during this incident")

    n_distinct_resources = chain_df["resource_accessed"].nunique()
    if n_distinct_resources >= 3:
        evidence.append(f"Access spanned {n_distinct_resources} different resources in quick succession")

    n_distinct_ips = chain_df["source_ip"].nunique()
    if n_distinct_ips > 1:
        evidence.append(f"{n_distinct_ips} distinct source IPs observed within this incident")

    max_risk_row = chain_df.loc[chain_df["risk_score"].idxmax()]
    evidence.append(f"Peak risk {max_risk_row['risk_score']} ({max_risk_row['risk_level']}) "
                     f"classified as {max_risk_row['predicted_attack_type']}")

    return evidence


def _stage_sequence(chain_df: pd.DataFrame) -> list:
    """Deduplicated, chronologically-ordered sequence of predicted attack
    types -- e.g. ['brute_force', 'lateral_movement'] tells the analyst
    the incident's overall shape without listing every single event.
    'normal' entries are dropped from the narrative (they're often
    lower-confidence cold-start noise pulled in only by time proximity),
    UNLESS the entire chain turns out to be normal, which would be an
    edge case worth surfacing rather than hiding."""
    ordered_types = chain_df.sort_values("timestamp")["predicted_attack_type"].tolist()
    non_normal = [t for t in ordered_types if t != "normal"]
    sequence_source = non_normal if non_normal else ordered_types

    stages = []
    for attack_type in sequence_source:
        if not stages or stages[-1] != attack_type:
            stages.append(attack_type)
    return stages


def correlate_chains(risk_scores: pd.DataFrame, raw_events: pd.DataFrame) -> list:
    candidates = _build_candidates(risk_scores, raw_events)

    raw_chains = []
    for entity_id, g in candidates.groupby("entity_id"):
        for chain_df in _merge_into_chains(g, cfg.CHAIN_TIME_WINDOW_MINUTES):
            if len(chain_df) == 0:
                continue
            raw_chains.append({"entity_id": entity_id, "df": chain_df})

    # ---- Cross-entity linking via shared source_ip within a nearby window ----
    # Build (chain_index -> set of linked chain_indices)
    link_window = pd.Timedelta(minutes=cfg.CHAIN_SHARED_IP_LINK_WINDOW_MINUTES)
    linked_groups = {i: {i} for i in range(len(raw_chains))}

    ip_to_chains = {}
    for i, chain in enumerate(raw_chains):
        for ip in chain["df"]["source_ip"].unique():
            ip_to_chains.setdefault(ip, []).append(i)

    for ip, chain_indices in ip_to_chains.items():
        if len(chain_indices) < 2:
            continue
        for a in range(len(chain_indices)):
            for b in range(a + 1, len(chain_indices)):
                i, j = chain_indices[a], chain_indices[b]
                if raw_chains[i]["entity_id"] == raw_chains[j]["entity_id"]:
                    continue  # same-entity linking already handled by the time-merge step
                start_i, end_i = raw_chains[i]["df"]["timestamp"].min(), raw_chains[i]["df"]["timestamp"].max()
                start_j, end_j = raw_chains[j]["df"]["timestamp"].min(), raw_chains[j]["df"]["timestamp"].max()
                gap = max(start_i, start_j) - min(end_i, end_j)
                if gap <= link_window:
                    linked_groups[i].add(j)
                    linked_groups[j].add(i)

    # ---- Assemble final chain records ----
    results = []
    for i, chain in enumerate(raw_chains):
        chain_df = chain["df"]
        linked_entities = sorted({raw_chains[j]["entity_id"] for j in linked_groups[i]})
        result = {
            "chain_id": f"CHAIN-{i + 1:04d}",
            "primary_entity": chain["entity_id"],
            "linked_entities": [e for e in linked_entities if e != chain["entity_id"]],
            "events": chain_df["event_id"].tolist(),
            "start_time": str(chain_df["timestamp"].min()),
            "end_time": str(chain_df["timestamp"].max()),
            "stages": _stage_sequence(chain_df),
            "overall_risk": float(chain_df["risk_score"].max()),
            "overall_risk_level": chain_df.loc[chain_df["risk_score"].idxmax(), "risk_level"],
            "evidence": _make_evidence(chain_df),
            "n_events": len(chain_df),
        }
        results.append(result)

    return sorted(results, key=lambda r: r["overall_risk"], reverse=True)
