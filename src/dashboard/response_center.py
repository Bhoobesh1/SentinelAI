"""SentinelAI Dashboard — Response Center view (Automated Threat Response Orchestrator).

Shows the deterministic recommended playbook for an alert, an illustrative
business-impact estimate, and lets an analyst simulate (dry-run) any action
with explicit approval. NOTHING here executes against a real system -- see
src/response/simulator.py's module docstring for why that's a deliberate
design choice, not a limitation.
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings as cfg
from src.dashboard.data_loader import load_risk_scores, load_raw_events, load_attack_chains
from src.response.playbook import get_playbook
from src.response.business_impact import estimate_business_impact, cost_of_delay_curve
from src.response.simulator import ActionSimulator

ACTION_ICONS = {
    "suspend_account": "🔒", "force_password_reset": "🔑", "block_ip": "🚫",
    "isolate_device": "🖥️", "revoke_resource_access": "⛔",
    "notify_manager": "📧", "open_incident_ticket": "🎫",
}


def render():
    st.header("Response Center")
    st.caption("Automated Threat Response Orchestrator (SOAR-lite) — deterministic playbook "
               "recommendations + illustrative business impact translation. **All actions are "
               "simulated dry-runs requiring explicit analyst approval; nothing here ever "
               "contacts a real system.**")

    risk_df = load_risk_scores()
    raw_events = load_raw_events()
    chains = load_attack_chains()

    sorted_alerts = risk_df.sort_values("risk_score", ascending=False)
    # Precompute ONCE -- see the identical fix/explanation in alert_investigation.py.
    display_lookup = sorted_alerts.set_index("event_id")[["risk_score", "risk_level", "predicted_attack_type"]].to_dict("index")
    event_id = st.selectbox(
        "Select an alert to generate a response plan for",
        options=sorted_alerts["event_id"].tolist(),
        format_func=lambda eid: f"{eid} — risk {display_lookup[eid]['risk_score']:.1f} "
                                 f"({display_lookup[eid]['risk_level']}, "
                                 f"{display_lookup[eid]['predicted_attack_type']})",
    )

    row = risk_df[risk_df["event_id"] == event_id].iloc[0]
    st.markdown(f"### {event_id} — {row['entity_id']} — {row['predicted_attack_type']} "
                f"(risk {row['risk_score']:.1f}, {row['risk_level']})")

    # ---- Recommended playbook (deterministic) ----
    st.subheader("Recommended Response Playbook")
    st.caption("Deterministic, attack-type-based recommendation — NOT an LLM decision. "
               "Same lookup-table philosophy as attack-chain correlation (Stage 9).")
    playbook = get_playbook(row["predicted_attack_type"], row["risk_level"])

    if not playbook:
        st.info("No response actions recommended (normal activity).")
    else:
        for action in playbook:
            icon = ACTION_ICONS.get(action["action_type"], "▶️")
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(f"{icon} **{action['action_type'].replace('_', ' ').title()}**")
                st.caption(action["target"])
            with col2:
                st.write(f"Urgency: `{action['urgency']}`")
            with col3:
                approver = st.text_input("Approved by (analyst name)", key=f"approver_{event_id}_{action['action_type']}",
                                          placeholder="e.g. jdoe")
                if st.button(f"Simulate (Dry Run)", key=f"sim_{event_id}_{action['action_type']}"):
                    if not approver:
                        st.error("Enter an approver name first -- no action can be simulated without explicit approval.")
                    else:
                        sim = ActionSimulator()
                        raw_row = raw_events[raw_events["event_id"] == event_id]
                        raw_row = raw_row.iloc[0] if len(raw_row) > 0 else None
                        result = sim.simulate_and_log(
                            action["action_type"], approved_by=approver,
                            entity_id=row["entity_id"],
                            ip=raw_row["source_ip"] if raw_row is not None else None,
                            device_id=raw_row["device_fingerprint"] if raw_row is not None else None,
                            resource=raw_row["resource_accessed"] if raw_row is not None else None,
                        )
                        st.success(f"Simulated (dry run only, not executed): {result['simulated_request']['method']} "
                                   f"{result['simulated_request']['url']}")
                        with st.expander("Full simulated request payload"):
                            st.json(result["simulated_request"])

    # ---- Business impact ----
    st.divider()
    st.subheader("Estimated Business Impact")
    st.caption("⚠️ ILLUSTRATIVE ESTIMATE using placeholder cost benchmarks in config/settings.py -- "
               "calibrate to your organization's real breach-cost data before relying on this operationally.")

    resources_touched = set()
    chain = next((c for c in chains if event_id in c["events"]), None)
    if chain:
        chain_events = set(chain["events"])
        resources_touched.update(raw_events[raw_events["event_id"].isin(chain_events)]["resource_accessed"].unique())
    else:
        raw_row = raw_events[raw_events["event_id"] == event_id]
        if len(raw_row) > 0:
            resources_touched.add(raw_row.iloc[0]["resource_accessed"])

    impact = estimate_business_impact(list(resources_touched), hours_unaddressed=0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Resources Touched", len(resources_touched))
    c2.metric("Est. Records at Risk", f"{impact['records_at_risk_estimate']:,}")
    c3.metric("Est. Impact if Unaddressed Now", f"${impact['immediate_impact_usd']:,.0f}")

    st.markdown("**Cost of delay** -- illustrative impact growth if response is postponed:")
    curve = cost_of_delay_curve(list(resources_touched), max_hours=24, step_hours=2)
    curve_df = pd.DataFrame(curve)
    fig = px.line(curve_df, x="hours_unaddressed", y="projected_impact_usd",
                  labels={"hours_unaddressed": "Hours Unaddressed", "projected_impact_usd": "Projected Impact (USD)"})
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Audit log ----
    st.divider()
    st.subheader("Response Audit Trail")
    st.caption("Every simulated action (never a real one) is logged here, demonstrating the "
               "audit pattern a real SOAR integration would require.")
    if os.path.exists(cfg.RESPONSE_AUDIT_LOG_PATH):
        with open(cfg.RESPONSE_AUDIT_LOG_PATH) as f:
            audit_log = json.load(f)
        if audit_log:
            audit_df = pd.DataFrame(audit_log)[["timestamp", "action_type", "approved_by", "status"]]
            st.dataframe(audit_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No actions simulated yet.")
    else:
        st.info("No actions simulated yet.")
