"""SentinelAI Dashboard — Alert Investigation view."""

import streamlit as st

from src.dashboard.data_loader import (
    load_risk_scores, load_raw_events, load_features,
    load_entity_baselines, load_attack_chains, load_ml_artifacts,
)
from src.explainability.explainer import explain_alert


def _risk_color(level: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(level, "⚪")


def render():
    st.header("Alert Investigation")

    risk_df = load_risk_scores()
    raw_events = load_raw_events()
    baselines = load_entity_baselines()
    chains = load_attack_chains()

    # Default to sorting by risk so the interesting alerts are easy to find
    sorted_alerts = risk_df.sort_values("risk_score", ascending=False)
    options = sorted_alerts["event_id"].tolist()
    # Precompute ONCE -- calling .set_index() inside format_func for every
    # one of 20,000+ options would be an O(n^2) blowup (found via testing).
    display_lookup = sorted_alerts.set_index("event_id")[["risk_score", "risk_level"]].to_dict("index")

    col1, col2 = st.columns([2, 1])
    with col1:
        event_id = st.selectbox(
            "Select an alert (sorted by risk, highest first)",
            options=options,
            format_func=lambda eid: f"{eid} — risk {display_lookup[eid]['risk_score']:.1f} "
                                     f"({display_lookup[eid]['risk_level']})",
        )
    with col2:
        manual_id = st.text_input("...or type an exact event_id")
        if manual_id:
            event_id = manual_id

    row = risk_df[risk_df["event_id"] == event_id]
    if len(row) == 0:
        st.error(f"No alert found with ID '{event_id}'.")
        return
    row = row.iloc[0]
    raw_row = raw_events[raw_events["event_id"] == event_id]
    raw_row = raw_row.iloc[0] if len(raw_row) > 0 else None

    st.markdown(f"### {_risk_color(row['risk_level'])} {event_id} — "
                f"Risk {row['risk_score']:.1f}/100 ({row['risk_level']})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entity", row["entity_id"])
    c2.metric("Classification", row["predicted_attack_type"])
    c3.metric("Attack Confidence", f"{row['attack_confidence']:.1%}")
    c4.metric("Hybrid Anomaly Score", f"{row['hybrid_anomaly_score']:.3f}")

    st.subheader("Risk Component Breakdown")
    components = {
        "Sequence Anomaly (max 30)": row["sequence_anomaly_component"],
        "Behavior Deviation (max 25)": row["behavior_deviation_component"],
        "Attack Confidence (max 20)": row["attack_confidence_component"],
        "Device Novelty (max 15)": row["device_novelty_component"],
        "Historical Context (max 10)": row["historical_context_component"],
    }
    st.bar_chart(components)

    if raw_row is not None:
        st.subheader("Raw Event Details")
        cols = st.columns(3)
        cols[0].write(f"**Source IP:** {raw_row['source_ip']}")
        cols[0].write(f"**Geo Location:** {raw_row['geo_location']}")
        cols[1].write(f"**Device:** {raw_row['device_fingerprint']}")
        cols[1].write(f"**Resource:** {raw_row['resource_accessed']}")
        cols[2].write(f"**Auth Success:** {raw_row['auth_success']}")
        cols[2].write(f"**Session Duration:** {raw_row['session_duration']:.2f} min")

    st.subheader("SHAP Evidence + Sequence Context")
    with st.spinner("Loading ML models (first time only, ~1-2 minutes; cached afterward)..."):
        artifacts = load_ml_artifacts()
    features_df = load_features()
    with st.spinner("Computing SHAP explanation..."):
        explanation = explain_alert(
            event_id=event_id, features_df=features_df, risk_row=row,
            explainer=artifacts["explainer"], xgb_meta=artifacts["xgb_meta"], encoder=artifacts["xgb_encoder"],
            ae_model=artifacts["ae_model"], ae_scaler=artifacts["ae_scaler"], ae_meta=artifacts["ae_meta"],
        )
    col_shap, col_seq = st.columns(2)
    with col_shap:
        st.markdown("**Top classifier evidence (SHAP attributions, not probabilities):**")
        for item in explanation["classifier_evidence"]:
            sign = "🔺" if item["shap_contribution"] >= 0 else "🔻"
            st.write(f"{sign} **{item['display_name']}** — {item['shap_contribution']:+.3f}")
    with col_seq:
        st.markdown("**Sequence reconstruction (hardest-to-reconstruct features):**")
        if explanation["sequence_evidence"]:
            for item in explanation["sequence_evidence"]:
                st.write(f"⚠️ **{item['display_name']}** — error {item['reconstruction_error']:.2f}")
        else:
            st.info("Insufficient history for sequence context (entity too new).")

    st.subheader("Baseline Comparison")
    profile = baselines.get(row["entity_id"])
    if profile and raw_row is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Entity's typical baseline:**")
            st.write(f"Known devices: {', '.join(profile['known_devices'])}")
            st.write(f"Typical locations: {', '.join(profile['typical_locations'])}")
            st.write(f"Typical resources: {', '.join(profile['typical_resources'])}")
        with col_b:
            st.markdown("**This event:**")
            device_flag = "🆕 NEW" if raw_row["device_fingerprint"] not in profile["known_devices"] else "known"
            loc_flag = "🆕 NEW" if raw_row["geo_location"] not in profile["typical_locations"] else "known"
            res_flag = "🆕 NEW" if raw_row["resource_accessed"] not in profile["typical_resources"] else "known"
            st.write(f"Device: {raw_row['device_fingerprint']} ({device_flag})")
            st.write(f"Location: {raw_row['geo_location']} ({loc_flag})")
            st.write(f"Resource: {raw_row['resource_accessed']} ({res_flag})")
    else:
        st.info("No baseline profile available for this entity yet.")

    st.subheader("Related Events & Attack Chain")
    chain = next((c for c in chains if event_id in c["events"]), None)
    if chain:
        st.write(f"Part of **{chain['chain_id']}** ({chain['n_events']} events, "
                 f"stages: {' → '.join(chain['stages']) if chain['stages'] else 'n/a'})")
        st.write(f"Window: {chain['start_time']} → {chain['end_time']}")
        for e in chain["evidence"]:
            st.write(f"- {e}")
        if chain["linked_entities"]:
            st.warning(f"Linked to other entities via shared source IP: {', '.join(chain['linked_entities'])}")
    else:
        st.info("No correlated chain found for this event -- appears isolated.")
