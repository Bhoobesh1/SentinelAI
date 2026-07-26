"""
SentinelAI Dashboard — cached data and model loading.

Streamlit re-runs the whole script on every interaction, so everything
here is wrapped in st.cache_data (for DataFrames/JSON) or st.cache_resource
(for models/agents that shouldn't be re-loaded or re-instantiated).
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg


@st.cache_data
def load_risk_scores() -> pd.DataFrame:
    df = pd.read_csv(cfg.RISK_SCORES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_raw_events() -> pd.DataFrame:
    df = pd.read_csv(cfg.RAW_EVENTS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_features() -> pd.DataFrame:
    df = pd.read_csv(cfg.PROCESSED_DATA_DIR + "/features.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_entity_baselines() -> dict:
    with open(cfg.PROCESSED_DATA_DIR + "/entity_baselines.json") as f:
        return json.load(f)


@st.cache_data
def load_attack_chains() -> list:
    with open(cfg.ATTACK_CHAINS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_evaluation_report() -> dict:
    path = os.path.join(cfg.REPORTS_DIR, "evaluation_report.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_alert_budget_curve() -> pd.DataFrame:
    path = os.path.join(cfg.REPORTS_DIR, "alert_budget_curve.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_drift_status() -> pd.DataFrame:
    path = cfg.DRIFT_STATUS_PATH
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_drift_summary() -> dict:
    if not os.path.exists(cfg.DRIFT_SUMMARY_PATH):
        return {}
    with open(cfg.DRIFT_SUMMARY_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_ml_artifacts():
    """Loads the autoencoder + XGBoost + SHAP explainer once per session."""
    from src.models.classifier import load_artifacts as load_xgb
    from src.models.autoencoder import load_artifacts as load_ae
    from src.explainability.explainer import build_explainer

    xgb_booster, xgb_encoder, xgb_meta = load_xgb()
    ae_model, ae_scaler, ae_meta = load_ae()
    explainer = build_explainer(xgb_booster)
    return {
        "xgb_booster": xgb_booster, "xgb_encoder": xgb_encoder, "xgb_meta": xgb_meta,
        "ae_model": ae_model, "ae_scaler": ae_scaler, "ae_meta": ae_meta,
        "explainer": explainer,
    }


@st.cache_resource
def load_copilot_graph():
    """Builds the LangGraph SOC Copilot once per session. Requires
    OPENAI_API_KEY to already be set (via .env, loaded in app.py)."""
    from src.agents.graph import build_graph
    return build_graph(model_name="gpt-4o-mini")


def artifacts_exist() -> dict:
    """Check which pipeline artifacts are present, so the dashboard can
    show a clear setup message instead of crashing if earlier stages
    haven't been run yet."""
    return {
        "raw_events": os.path.exists(cfg.RAW_EVENTS_PATH),
        "features": os.path.exists(cfg.PROCESSED_DATA_DIR + "/features.csv"),
        "risk_scores": os.path.exists(cfg.RISK_SCORES_PATH),
        "entity_baselines": os.path.exists(cfg.PROCESSED_DATA_DIR + "/entity_baselines.json"),
        "attack_chains": os.path.exists(cfg.ATTACK_CHAINS_PATH),
        "evaluation_report": os.path.exists(os.path.join(cfg.REPORTS_DIR, "evaluation_report.json")),
        "xgb_model": os.path.exists(cfg.XGB_MODEL_PATH),
        "autoencoder_model": os.path.exists(cfg.AUTOENCODER_MODEL_PATH),
        "rag_index": os.path.exists(cfg.RAG_INDEX_PATH),
        "drift_status": os.path.exists(cfg.DRIFT_STATUS_PATH),
    }
