"""SentinelAI Dashboard — Entity Investigation view."""

import plotly.express as px
import pandas as pd
import streamlit as st

from config import settings as cfg
from src.dashboard.data_loader import (
    load_risk_scores, load_raw_events, load_entity_baselines,
    load_drift_status, load_drift_summary,
)


def render():
    st.header("Entity Investigation")

    risk_df = load_risk_scores()
    raw_events = load_raw_events()
    baselines = load_entity_baselines()

    entity_ids = sorted(raw_events["entity_id"].unique())
    entity_id = st.selectbox("Select an entity", options=entity_ids)

    profile = baselines.get(entity_id, {})
    entity_events = raw_events[raw_events["entity_id"] == entity_id].sort_values("timestamp", ascending=False)
    entity_risk = risk_df[risk_df["entity_id"] == entity_id].sort_values("timestamp")

    st.subheader(f"{entity_id} — Behavioral Profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", profile.get("total_events", len(entity_events)))
    c2.metric("Known Devices", len(profile.get("known_devices", [])))
    c3.metric("Typical Locations", len(profile.get("typical_locations", [])))
    c4.metric("Auth Failure Rate", f"{profile.get('auth_failure_rate', 0):.1%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Known devices:** " + ", ".join(profile.get("known_devices", ["n/a"])))
        st.markdown("**Typical locations:** " + ", ".join(profile.get("typical_locations", ["n/a"])))
        st.markdown("**Typical resources:** " + ", ".join(profile.get("typical_resources", ["n/a"])))
    with col_b:
        st.markdown("**Normal login hours (distribution)**")
        fig = px.histogram(entity_events, x=entity_events["timestamp"].dt.hour, nbins=24,
                            labels={"x": "Hour of day"})
        fig.update_layout(height=250, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Risk Timeline (behavioral changes over time)")
    if len(entity_risk) > 0:
        fig = px.scatter(
            entity_risk, x="timestamp", y="risk_score", color="risk_level",
            color_discrete_map={"LOW": "#4CAF50", "MEDIUM": "#FFC107", "HIGH": "#FF9800", "CRITICAL": "#F44336"},
            hover_data=["event_id", "predicted_attack_type"],
        )
        fig.add_hline(y=60, line_dash="dash", line_color="gray", annotation_text="HIGH threshold")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk history available for this entity.")

    st.divider()
    st.subheader("Concept Drift Status")
    st.caption("Is this entity's behavioral baseline SUSTAINEDLY shifting over time (any cause -- "
               "legitimate role change or a slow-moving threat), as opposed to a single anomalous event? "
               "This is a separate signal from the risk score above -- see the note below.")

    drift_summary = load_drift_summary()
    drift_status_df = load_drift_status()
    entity_drift = drift_status_df[drift_status_df["entity_id"] == entity_id] if len(drift_status_df) > 0 else pd.DataFrame()

    if entity_id not in drift_summary:
        st.info("No drift analysis available. Run `python scripts/detect_drift.py` first.")
    elif drift_summary[entity_id]["status"] == "insufficient_history":
        st.info("Not enough history yet to establish a drift reference baseline for this entity "
                f"(needs {cfg.DRIFT_REFERENCE_DAYS}+ active days).")
    else:
        status = drift_summary[entity_id]
        is_sustained = status["status"] == "sustained_drift_detected"

        d1, d2, d3 = st.columns(3)
        d1.metric("Drift Status", "⚠️ Sustained drift detected" if is_sustained else "✅ Stable")
        d2.metric("Final EWMA Score", f"{status['final_ewma_score']:.3f}")
        if is_sustained:
            d3.metric("First Confirmed", status["first_confirmed_day"])

        if is_sustained:
            st.warning(
                "**Potential behavioral drift detected.** This entity's baseline has been "
                "sustainedly different from its established reference for multiple consecutive "
                "days -- not just a single unusual event. This could mean a legitimate role/work "
                "pattern change, OR a slow-moving threat. Cross-check with recent alerts and "
                "confirm with the entity's manager/HR before deciding which."
            )

        if len(entity_drift) > 0:
            fig = px.line(
                entity_drift, x="day", y="ewma_score",
                labels={"day": "Day", "ewma_score": "EWMA Drift Score"},
            )
            fig.add_hline(y=cfg.DRIFT_SCORE_THRESHOLD, line_dash="dash", line_color="orange",
                          annotation_text="Sustained-drift threshold")
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Component breakdown (hour / device / resource / session)"):
                comp_cols = ["day", "hour_component", "device_component",
                             "resource_component", "session_component"]
                st.dataframe(entity_drift[comp_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Recent Activity")
    display_cols = ["event_id", "timestamp", "source_ip", "geo_location",
                     "resource_accessed", "device_fingerprint", "auth_success"]
    st.dataframe(entity_events[display_cols].head(15), use_container_width=True, hide_index=True)
