"""SentinelAI Dashboard — SOC Copilot chat view."""

import os

import streamlit as st

from src.dashboard.data_loader import load_risk_scores, load_copilot_graph


def render():
    st.header("SOC Copilot")

    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is not set. Copy `.env.example` to `.env`, add your real "
            "OpenAI API key, and restart the dashboard to use the Copilot."
        )
        return

    risk_df = load_risk_scores()

    with st.expander("Optionally select an alert or entity to start the conversation about"):
        col1, col2 = st.columns(2)
        with col1:
            top_alerts = risk_df.sort_values("risk_score", ascending=False).head(30)
            selected_alert = st.selectbox(
                "Alert", options=[""] + top_alerts["event_id"].tolist(),
            )
        with col2:
            entities = sorted(risk_df["entity_id"].unique())
            selected_entity = st.selectbox("Entity", options=[""] + entities)

        if st.button("Start conversation about this"):
            if selected_alert:
                starter = f"Explain alert {selected_alert}."
            elif selected_entity:
                starter = f"Why was {selected_entity} flagged?"
            else:
                starter = None
            if starter:
                st.session_state.setdefault("copilot_messages", [])
                st.session_state["copilot_pending_input"] = starter

    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = []
    if "copilot_graph_state" not in st.session_state:
        st.session_state["copilot_graph_state"] = {"messages": [], "route": ""}

    for msg in st.session_state["copilot_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("copilot_pending_input", None)
    user_input = st.chat_input("Ask about an alert, entity, or request an incident report...")
    if pending and not user_input:
        user_input = pending

    if user_input:
        st.session_state["copilot_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        from langchain_core.messages import HumanMessage

        with st.chat_message("assistant"):
            with st.spinner("Investigating..."):
                try:
                    graph = load_copilot_graph()
                    state = st.session_state["copilot_graph_state"]
                    state["messages"].append(HumanMessage(content=user_input))
                    state = graph.invoke(state)
                    st.session_state["copilot_graph_state"] = state
                    response_text = state["messages"][-1].content
                except Exception as e:
                    response_text = f"Error contacting the Copilot: {e}"
            st.markdown(response_text)
        st.session_state["copilot_messages"].append({"role": "assistant", "content": response_text})

    if st.button("Clear conversation"):
        st.session_state["copilot_messages"] = []
        st.session_state["copilot_graph_state"] = {"messages": [], "route": ""}
        st.rerun()
