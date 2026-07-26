"""SentinelAI Dashboard — SOC Overview view."""

import plotly.express as px
import streamlit as st

from src.dashboard.data_loader import load_risk_scores, load_drift_summary


def render():
    st.header("SOC Overview")

    risk_df = load_risk_scores()
    total_events = len(risk_df)
    flagged = risk_df[risk_df["risk_level"].isin(["HIGH", "CRITICAL"])]
    critical = risk_df[risk_df["risk_level"] == "CRITICAL"]
    high = risk_df[risk_df["risk_level"] == "HIGH"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Events", f"{total_events:,}")
    col2.metric("Flagged (HIGH/CRITICAL)", f"{len(flagged):,}")
    col3.metric("Critical Alerts", f"{len(critical):,}")
    col4.metric("High Alerts", f"{len(high):,}")
    col5.metric("Flagged Rate", f"{100 * len(flagged) / total_events:.2f}%")

    st.caption("\"Flagged\" = risk_level HIGH or CRITICAL, i.e. events the system is "
               "recommending an analyst actually look at -- not a ground-truth label.")

    drift_summary = load_drift_summary()
    if drift_summary:
        sustained = [eid for eid, s in drift_summary.items() if s["status"] == "sustained_drift_detected"]
        if sustained:
            st.warning(f"⚠️ **{len(sustained)} entities** show sustained behavioral drift "
                       f"(separate from event-level alerts -- see Entity Investigation for details): "
                       f"{', '.join(sustained)}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Attack Type Distribution")
        attack_dist = risk_df[risk_df["predicted_attack_type"] != "normal"]["predicted_attack_type"].value_counts()
        if len(attack_dist) > 0:
            fig = px.bar(
                x=attack_dist.values, y=attack_dist.index, orientation="h",
                labels={"x": "Count", "y": "Predicted Attack Type"},
                color=attack_dist.values, color_continuous_scale="Reds",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No non-normal predictions found.")

    with col_right:
        st.subheader("Risk Level Distribution")
        risk_dist = risk_df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        colors = {"LOW": "#4CAF50", "MEDIUM": "#FFC107", "HIGH": "#FF9800", "CRITICAL": "#F44336"}
        fig = px.bar(
            x=risk_dist.index, y=risk_dist.values,
            labels={"x": "Risk Level", "y": "Count"},
            color=risk_dist.index, color_discrete_map=colors,
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Highest-Risk Recent Alerts")
    top_alerts = risk_df.sort_values("risk_score", ascending=False).head(20)
    display_cols = ["event_id", "entity_id", "timestamp", "risk_score", "risk_level",
                     "predicted_attack_type", "attack_confidence"]
    st.dataframe(top_alerts[display_cols], use_container_width=True, hide_index=True)
