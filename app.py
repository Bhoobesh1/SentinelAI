"""
SentinelAI — Streamlit Dashboard (main entry point).

Usage:
    streamlit run app.py

Requires all prior pipeline stages to have been run at least once
(data generation through RAG indexing). The SOC Copilot tab additionally
requires OPENAI_API_KEY to be set (via .env).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="SentinelAI — SOC Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.data_loader import artifacts_exist


def render_setup_warning(missing: list):
    st.error("Some pipeline artifacts are missing. Run the setup scripts in order first:")
    st.code(
        "python scripts/generate_data.py\n"
        "python scripts/build_features.py\n"
        "python scripts/train_autoencoder.py\n"
        "python scripts/train_classifier.py\n"
        "python scripts/score_events.py\n"
        "python scripts/explain_alerts.py\n"
        "python scripts/evaluate.py\n"
        "python scripts/correlate_chains.py\n"
        "python scripts/build_rag_index.py",
        language="bash",
    )
    st.write("Missing artifacts:", ", ".join(missing))


def main():
    st.sidebar.title("🛡️ SentinelAI")
    st.sidebar.caption("Adaptive Agentic Behavioral Threat Detection & SOC Copilot")

    status = artifacts_exist()
    missing = [k for k, v in status.items() if not v]

    view = st.sidebar.radio(
        "Navigate",
        ["SOC Overview", "Alert Investigation", "Entity Investigation", "SOC Copilot",
         "Model Performance", "Response Center"],
    )

    st.sidebar.divider()
    with st.sidebar.expander("Pipeline status"):
        for k, v in status.items():
            st.write(("✅ " if v else "❌ ") + k)

    # Core artifacts needed by nearly every view
    core_required = ["raw_events", "features", "risk_scores", "entity_baselines"]
    if any(k in missing for k in core_required):
        render_setup_warning(missing)
        return

    if view == "SOC Overview":
        from src.dashboard import overview
        overview.render()
    elif view == "Alert Investigation":
        from src.dashboard import alert_investigation
        alert_investigation.render()
    elif view == "Entity Investigation":
        from src.dashboard import entity_investigation
        entity_investigation.render()
    elif view == "SOC Copilot":
        from src.dashboard import copilot
        copilot.render()
    elif view == "Model Performance":
        from src.dashboard import model_performance
        model_performance.render()
    elif view == "Response Center":
        from src.dashboard import response_center
        response_center.render()


if __name__ == "__main__":
    main()
