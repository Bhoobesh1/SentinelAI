"""
SentinelAI — SOC Copilot investigation tools.

CRITICAL: every tool here is a deterministic Python function reading from
precomputed artifacts on disk (risk_scores.csv, features.csv, raw
events.csv, entity_baselines.json, attack_chains.json, the RAG index).
None of them ever read data/ground_truth/labels.csv -- an investigation
tool that could see the answer key would defeat the entire point of
having an investigation agent. The LLM decides WHEN to call these and
HOW to interpret the results; it never computes anomaly/risk scores
itself.

All data loading is done once via the module-level DataStore and cached,
since tools may be called many times within a single conversation.
"""

import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg
from langchain_core.tools import tool


class DataStore:
    """Lazy-loaded, cached access to all precomputed artifacts the
    Copilot's tools need. Loaded once per process."""
    _instance = None

    def __init__(self):
        self.raw_events = pd.read_csv(cfg.RAW_EVENTS_PATH)
        self.raw_events["timestamp"] = pd.to_datetime(self.raw_events["timestamp"])

        self.features = pd.read_csv(cfg.PROCESSED_DATA_DIR + "/features.csv")
        self.features["timestamp"] = pd.to_datetime(self.features["timestamp"])

        self.risk_scores = pd.read_csv(cfg.RISK_SCORES_PATH)
        self.risk_scores["timestamp"] = pd.to_datetime(self.risk_scores["timestamp"])

        with open(cfg.PROCESSED_DATA_DIR + "/entity_baselines.json") as f:
            self.entity_baselines = json.load(f)

        with open(cfg.ATTACK_CHAINS_PATH) as f:
            self.attack_chains = json.load(f)

        self.drift_summary = {}
        if os.path.exists(cfg.DRIFT_SUMMARY_PATH):
            with open(cfg.DRIFT_SUMMARY_PATH) as f:
                self.drift_summary = json.load(f)

        # ML artifacts, loaded lazily (only needed for get_model_evidence)
        self._xgb_booster = None
        self._xgb_encoder = None
        self._xgb_meta = None
        self._ae_model = None
        self._ae_scaler = None
        self._ae_meta = None
        self._shap_explainer = None

        # RAG artifacts, loaded lazily (only needed for search_security_knowledge)
        self._rag_chunks = None
        self._rag_vectorizer = None
        self._rag_index = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_ml_artifacts(self):
        if self._xgb_booster is None:
            from src.models.classifier import load_artifacts as load_xgb
            from src.models.autoencoder import load_artifacts as load_ae
            from src.explainability.explainer import build_explainer
            self._xgb_booster, self._xgb_encoder, self._xgb_meta = load_xgb()
            self._ae_model, self._ae_scaler, self._ae_meta = load_ae()
            self._shap_explainer = build_explainer(self._xgb_booster)

    def _ensure_rag_artifacts(self):
        if self._rag_index is None:
            from src.rag.retriever import load_knowledge_base
            self._rag_chunks, self._rag_vectorizer, self._rag_index = load_knowledge_base()


def _entity_not_found(entity_id: str) -> dict:
    return {"error": f"No profile found for entity '{entity_id}'. Check the entity ID is correct."}


def _event_not_found(event_id: str) -> dict:
    return {"error": f"No event found with ID '{event_id}'. Check the event ID is correct."}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def get_entity_profile(entity_id: str) -> dict:
    """Get an entity's behavioral BASELINE: known devices, typical locations,
    typical resources, and overall authentication failure rate, built from
    their full historical activity. Use this to understand what 'normal'
    looks like for this entity before judging whether a specific event is
    unusual."""
    ds = DataStore.instance()
    profile = ds.entity_baselines.get(entity_id)
    if profile is None:
        return _entity_not_found(entity_id)
    return {"entity_id": entity_id, **profile}


@tool
def get_top_alerts_for_entity(entity_id: str, n: int = 5) -> dict:
    """Get an entity's highest-RISK alerts, sorted by risk_score descending
    (NOT by recency). Use this when the analyst asks broadly why an entity
    was flagged or what their worst/most notable alerts are -- the most
    recent event for an entity is often not the most significant one, so
    prefer this tool over get_entity_history for open-ended 'why was this
    entity flagged' questions."""
    ds = DataStore.instance()
    g = ds.risk_scores[ds.risk_scores["entity_id"] == entity_id].sort_values("risk_score", ascending=False).head(n)
    if len(g) == 0:
        return _entity_not_found(entity_id)
    records = g[["event_id", "timestamp", "risk_score", "risk_level",
                 "predicted_attack_type", "attack_confidence"]].copy()
    records["timestamp"] = records["timestamp"].astype(str)
    return {"entity_id": entity_id, "top_alerts": records.to_dict(orient="records")}


@tool
def get_entity_history(entity_id: str, n: int = 5) -> dict:
    """Get an entity's N most recent events (default 5), each with
    timestamp, source IP, location, resource accessed, device, auth
    result, and its risk score if one was computed. Use this to answer
    questions like 'show their recent logins' or 'what did they do
    after this event'."""
    ds = DataStore.instance()
    g = ds.raw_events[ds.raw_events["entity_id"] == entity_id].sort_values("timestamp", ascending=False).head(n)
    if len(g) == 0:
        return _entity_not_found(entity_id)

    risk_lookup = ds.risk_scores.set_index("event_id")[["risk_score", "risk_level", "predicted_attack_type"]]
    records = []
    for _, row in g.iterrows():
        record = {
            "event_id": row["event_id"],
            "timestamp": str(row["timestamp"]),
            "source_ip": row["source_ip"],
            "geo_location": row["geo_location"],
            "resource_accessed": row["resource_accessed"],
            "device_fingerprint": row["device_fingerprint"],
            "auth_method": row["auth_method"],
            "auth_success": bool(row["auth_success"]),
        }
        if row["event_id"] in risk_lookup.index:
            r = risk_lookup.loc[row["event_id"]]
            record.update({
                "risk_score": float(r["risk_score"]),
                "risk_level": r["risk_level"],
                "predicted_attack_type": r["predicted_attack_type"],
            })
        records.append(record)
    return {"entity_id": entity_id, "events": records}


@tool
def get_alert_details(event_id: str) -> dict:
    """Get the full risk assessment for a specific event: risk score,
    risk level, predicted attack type, classifier confidence, and the
    5-component risk breakdown (sequence anomaly, behavior deviation,
    attack confidence, device novelty, historical context). Use this
    first when an analyst asks about a specific alert/event ID."""
    ds = DataStore.instance()
    row = ds.risk_scores[ds.risk_scores["event_id"] == event_id]
    if len(row) == 0:
        return _event_not_found(event_id)
    row = row.iloc[0]

    raw = ds.raw_events[ds.raw_events["event_id"] == event_id]
    raw_info = {}
    if len(raw) > 0:
        raw = raw.iloc[0]
        raw_info = {
            "source_ip": raw["source_ip"], "geo_location": raw["geo_location"],
            "resource_accessed": raw["resource_accessed"], "device_fingerprint": raw["device_fingerprint"],
            "auth_success": bool(raw["auth_success"]), "session_duration": float(raw["session_duration"]),
        }

    return {
        "event_id": event_id,
        "entity_id": row["entity_id"],
        "timestamp": str(row["timestamp"]),
        "risk_score": float(row["risk_score"]),
        "risk_level": row["risk_level"],
        "predicted_attack_type": row["predicted_attack_type"],
        "attack_confidence": float(row["attack_confidence"]),
        "hybrid_anomaly_score": float(row["hybrid_anomaly_score"]),
        "component_breakdown": {
            "sequence_anomaly": float(row["sequence_anomaly_component"]),
            "behavior_deviation": float(row["behavior_deviation_component"]),
            "attack_confidence": float(row["attack_confidence_component"]),
            "device_novelty": float(row["device_novelty_component"]),
            "historical_context": float(row["historical_context_component"]),
        },
        **raw_info,
    }


@tool
def get_model_evidence(event_id: str) -> dict:
    """Get the detailed ML EVIDENCE behind an alert: the top SHAP feature
    attributions driving the classifier's prediction (with direction of
    influence), and, if sequence context is available, which behavioral
    features the autoencoder found hardest to reconstruct as normal. Use
    this when the analyst asks WHY an alert was flagged, or asks for
    specific evidence/proof rather than just the risk score."""
    ds = DataStore.instance()
    ds._ensure_ml_artifacts()

    row = ds.risk_scores[ds.risk_scores["event_id"] == event_id]
    if len(row) == 0:
        return _event_not_found(event_id)
    row = row.iloc[0]

    from src.explainability.explainer import explain_alert
    explanation = explain_alert(
        event_id=event_id, features_df=ds.features, risk_row=row,
        explainer=ds._shap_explainer, xgb_meta=ds._xgb_meta, encoder=ds._xgb_encoder,
        ae_model=ds._ae_model, ae_scaler=ds._ae_scaler, ae_meta=ds._ae_meta,
    )
    return explanation


@tool
def get_device_history(entity_id: str, device_fingerprint: Optional[str] = None) -> dict:
    """Check whether a device has been used before by this entity, and
    whether the SAME device fingerprint has also appeared on OTHER
    entities (a strong signal of device spoofing / shared attacker
    tooling if so). If device_fingerprint is omitted, returns the
    entity's known devices with usage counts instead."""
    ds = DataStore.instance()
    entity_events = ds.raw_events[ds.raw_events["entity_id"] == entity_id]
    if len(entity_events) == 0:
        return _entity_not_found(entity_id)

    if device_fingerprint is None:
        counts = entity_events["device_fingerprint"].value_counts()
        return {"entity_id": entity_id, "known_devices": counts.to_dict()}

    entity_uses = int((entity_events["device_fingerprint"] == device_fingerprint).sum())
    other_entities = ds.raw_events[
        (ds.raw_events["device_fingerprint"] == device_fingerprint) &
        (ds.raw_events["entity_id"] != entity_id)
    ]["entity_id"].unique().tolist()

    return {
        "entity_id": entity_id,
        "device_fingerprint": device_fingerprint,
        "times_used_by_this_entity": entity_uses,
        "previously_seen_for_this_entity": entity_uses > 0,
        "also_seen_on_other_entities": other_entities,
    }


@tool
def get_related_alerts(event_id: str) -> dict:
    """Find other alerts related to this event -- events in the same
    correlated attack chain (same entity, close in time, or linked via a
    shared source IP across entities). Use this to answer 'are there
    related suspicious events' or 'what happened after/before this'."""
    ds = DataStore.instance()
    for chain in ds.attack_chains:
        if event_id in chain["events"]:
            related = [e for e in chain["events"] if e != event_id]
            return {
                "event_id": event_id, "chain_id": chain["chain_id"],
                "related_events": related, "stages": chain["stages"],
                "linked_entities": chain["linked_entities"],
            }
    return {"event_id": event_id, "related_events": [], "note": "No correlated chain found for this event."}


@tool
def get_attack_chain(chain_id: Optional[str] = None, event_id: Optional[str] = None) -> dict:
    """Get the full correlated attack-chain record (all events, stages,
    overall risk, evidence, linked entities) by chain_id, or by any
    event_id known to belong to a chain. Use this when the analyst asks
    to understand a whole incident, not just a single alert."""
    ds = DataStore.instance()
    if chain_id is not None:
        for chain in ds.attack_chains:
            if chain["chain_id"] == chain_id:
                return chain
        return {"error": f"No chain found with ID '{chain_id}'."}
    if event_id is not None:
        for chain in ds.attack_chains:
            if event_id in chain["events"]:
                return chain
        return {"error": f"No chain found containing event '{event_id}'."}
    return {"error": "Provide either chain_id or event_id."}


@tool
def get_entity_drift_status(entity_id: str) -> dict:
    """Check whether an entity's behavioral BASELINE is showing sustained
    concept drift -- a gradual, multi-day shift in login hours, devices,
    or resource access, as opposed to a single anomalous event. This is a
    SEPARATE signal from risk scores/alerts: drift can indicate either a
    legitimate change (role change, new work pattern) or a slow-moving
    threat -- use this when the analyst asks about behavioral drift,
    baseline changes, or whether something might be a legitimate evolving
    pattern rather than an attack."""
    ds = DataStore.instance()
    status = ds.drift_summary.get(entity_id)
    if status is None:
        return {"entity_id": entity_id, "note": "No drift analysis available for this entity."}
    return {"entity_id": entity_id, **status}


@tool
def search_security_knowledge(query: str) -> list:
    """Search the general cybersecurity knowledge base for attack
    descriptions, indicators, investigation questions, and defensive
    recommendations relevant to the query. This returns GENERAL security
    knowledge, NOT telemetry about any specific entity or event -- never
    treat these results as evidence about what actually happened."""
    ds = DataStore.instance()
    ds._ensure_rag_artifacts()
    from src.rag.retriever import search_security_knowledge as rag_search
    return rag_search(query, ds._rag_chunks, ds._rag_vectorizer, ds._rag_index)


@tool
def generate_response_playbook(event_id: str) -> dict:
    """Generate the recommended RESPONSE PLAYBOOK for an alert: a
    deterministic, attack-type-specific list of containment/remediation
    actions (e.g., suspend account, block IP, isolate device, open an
    incident ticket), plus an illustrative estimated business impact in
    dollars if left unaddressed. Use this when the analyst asks 'what
    should I do about this', 'how do I respond to this incident', or
    asks for a remediation plan. This does NOT execute anything -- it
    only recommends; use the dashboard's Response Center to simulate
    (dry-run) any action with explicit approval."""
    from src.response.playbook import get_playbook
    from src.response.business_impact import estimate_business_impact

    ds = DataStore.instance()
    row = ds.risk_scores[ds.risk_scores["event_id"] == event_id]
    if len(row) == 0:
        return _event_not_found(event_id)
    row = row.iloc[0]

    playbook = get_playbook(row["predicted_attack_type"], row["risk_level"])

    # Gather resources touched -- from the attack chain if this event
    # belongs to one, else just this event's own resource.
    resources_touched = set()
    for chain in ds.attack_chains:
        if event_id in chain["events"]:
            chain_event_ids = set(chain["events"])
            chain_resources = ds.raw_events[ds.raw_events["event_id"].isin(chain_event_ids)]["resource_accessed"]
            resources_touched.update(chain_resources.unique())
            break
    if not resources_touched:
        raw = ds.raw_events[ds.raw_events["event_id"] == event_id]
        if len(raw) > 0:
            resources_touched.add(raw.iloc[0]["resource_accessed"])

    impact = estimate_business_impact(list(resources_touched), hours_unaddressed=0)

    return {
        "event_id": event_id,
        "entity_id": row["entity_id"],
        "predicted_attack_type": row["predicted_attack_type"],
        "risk_level": row["risk_level"],
        "recommended_actions": playbook,
        "estimated_business_impact": impact,
        "note": ("These actions are DETERMINISTIC recommendations based on attack type, "
                 "not an LLM decision. Nothing is executed automatically -- every action "
                 "requires explicit human approval via the dashboard's Response Center."),
    }


ALL_TOOLS = [
    get_entity_profile, get_top_alerts_for_entity, get_entity_history, get_alert_details,
    get_model_evidence, get_device_history, get_related_alerts, get_attack_chain,
    get_entity_drift_status, generate_response_playbook, search_security_knowledge,
]
