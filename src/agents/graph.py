"""
SentinelAI — LangGraph SOC Copilot.

Three reasoning roles, per spec section 19:

1. ROUTER      -- classifies analyst intent (report generation vs. any
                  other investigation question) so the right downstream
                  agent handles the turn.
2. INVESTIGATION/EXPLANATION AGENT -- a tool-calling ReAct agent with
                  access to all 8 deterministic tools. It decides WHICH
                  tools to call based on the question -- it does not
                  call every tool for every question.
3. REPORT AGENT -- also tool-calling, but constrained to produce the
                  structured GROUNDED AI RESPONSE format (spec section 22)
                  when the analyst asks for a full incident report.

Both agents are instructed to keep four evidence categories distinct
(ML evidence / historical telemetry / retrieved knowledge / AI
interpretation), to never fabricate telemetry, to never describe SHAP
values or risk scores as probabilities, and to say "Insufficient
evidence available" rather than guess when a tool has nothing to offer.

IMPORTANT: this module requires OPENAI_API_KEY to be set (via .env or
the environment) to actually run -- it cannot be exercised without a
live OpenAI API call.
"""

import os
import sys
from typing import Annotated, TypedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg
from src.agents.tools import ALL_TOOLS

GROUNDING_RULES = """
Grounding rules you must always follow:
- Keep FOUR kinds of evidence clearly distinct in your answer:
    1. ML evidence: autoencoder reconstruction signals and SHAP feature attributions.
       SHAP values are attributions, NOT probabilities. Risk scores (0-100) are a
       priority signal, NOT a probability of compromise.
    2. Historical telemetry: an entity's actual past events, from get_entity_history /
       get_entity_profile / get_device_history. This is real recorded data.
    3. Retrieved security knowledge: general guidance from search_security_knowledge.
       This is GENERAL knowledge about attack patterns, not a finding about this
       specific entity or event -- never present it as if it were telemetry.
    4. AI interpretation: your own synthesis of the above. Always keep this
       clearly attributed to your reasoning, not stated as an independent fact.
- NEVER invent or assume telemetry that a tool did not return. If a tool has
  no relevant data, say "Insufficient evidence available" rather than guessing.
- Only call the tools you actually need for this specific question -- do not
  call every tool on every turn.
- Retrieved security knowledge must never override or contradict ML evidence
  or historical telemetry -- it only adds context for investigation.
"""

ROUTER_PROMPT = (
    "You are the router for a SOC (Security Operations Center) copilot. "
    "Read the analyst's latest message in context of the conversation so far. "
    "Respond with EXACTLY one word and nothing else: "
    "REPORT if they are asking you to generate, write, or produce a full incident report; "
    "INVESTIGATE for any other question (explanations, history, correlation, "
    "security knowledge, false-positive checks, next steps, etc.)."
)

INVESTIGATION_PROMPT = f"""You are SentinelAI's SOC investigation copilot. Analysts will ask you
questions about specific alerts, entities, and incidents. Use the available
tools to gather real evidence before answering -- never answer from
assumption alone when a tool could confirm or refute it.

When an analyst asks an open-ended question like "why was USER_042 flagged"
WITHOUT naming a specific event/alert ID, do not assume they mean the most
recent event -- use get_top_alerts_for_entity first to find their highest-
risk alert, since that is almost always the notable one. Use
get_entity_history instead when they specifically ask about recent activity
or "what happened after/before" a given point in time.

When an analyst asks whether something "could be a false positive" or asks
for a benign/alternative explanation, always call search_security_knowledge
(e.g. with a query like "false positive considerations" or the relevant
attack type) in addition to the ML/historical evidence -- the knowledge base
has specific guidance on ruling out benign explanations (VPN/proxy egress
IPs, legitimate new devices, role changes, etc.) that should ground this
kind of judgment call, not just the risk score alone.

{GROUNDING_RULES}

Support natural follow-up questions using the conversation history (e.g. if
the analyst previously asked about USER_042 and then asks "did they use this
device before", resolve "they" to USER_042 from context).
"""

REPORT_PROMPT = f"""You are SentinelAI's SOC incident report generator. The analyst wants a
full incident report. First use the available tools to gather whatever
evidence you need (alert details, model evidence, entity history, related
alerts/attack chain, relevant security knowledge, and a response playbook).
Then produce the report using EXACTLY this structure:

ALERT SUMMARY
Entity:
Alert:
Risk:
Classification:

ML EVIDENCE
(SHAP attributions and/or sequence reconstruction evidence, clearly labeled
as attributions, not probabilities)

BEHAVIORAL CONTEXT
(how this compares to the entity's own baseline)

HISTORICAL CONTEXT
(relevant recent history, related/correlated alerts if any)

SECURITY KNOWLEDGE
(general guidance retrieved from the knowledge base, clearly marked as
general knowledge, not a specific finding)

ASSESSMENT
(your synthesis -- clearly your interpretation, grounded in the evidence above)

RECOMMENDED RESPONSE PLAYBOOK
(from generate_response_playbook -- list the recommended actions and note
that these are deterministic recommendations, not something you decided,
and that nothing is auto-executed -- every action needs explicit analyst
approval via the dashboard's Response Center)

ESTIMATED BUSINESS IMPACT
(from generate_response_playbook's business impact estimate -- present the
dollar figure but always repeat that it is an ILLUSTRATIVE estimate using
placeholder benchmarks, not a calibrated real number)

RECOMMENDED INVESTIGATION
1.
2.
3.

If evidence for any section is unavailable, write "Insufficient evidence
available" for that section rather than fabricating content.

{GROUNDING_RULES}
"""


class CopilotState(TypedDict):
    messages: Annotated[list, "add_messages"]
    route: str


def build_graph(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Build and compile the 3-role SOC Copilot graph. Requires
    OPENAI_API_KEY to be set in the environment (e.g. via .env +
    python-dotenv, loaded by the caller)."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import create_react_agent

    # Re-declare with the real add_messages reducer (the TypedDict above
    # uses a string placeholder purely for readability/documentation).
    class _State(TypedDict):
        messages: Annotated[list, add_messages]
        route: str

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    investigate_agent = create_react_agent(llm, ALL_TOOLS, prompt=INVESTIGATION_PROMPT)
    report_agent = create_react_agent(llm, ALL_TOOLS, prompt=REPORT_PROMPT)

    def router_node(state: _State) -> dict:
        response = llm.invoke([SystemMessage(content=ROUTER_PROMPT)] + state["messages"])
        route = "report" if "REPORT" in response.content.upper() else "investigate"
        return {"route": route}

    def investigate_node(state: _State) -> dict:
        result = investigate_agent.invoke({"messages": state["messages"]})
        new_messages = result["messages"][len(state["messages"]):]
        return {"messages": new_messages}

    def report_node(state: _State) -> dict:
        result = report_agent.invoke({"messages": state["messages"]})
        new_messages = result["messages"][len(state["messages"]):]
        return {"messages": new_messages}

    graph = StateGraph(_State)
    graph.add_node("router", router_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["route"], {"investigate": "investigate", "report": "report"})
    graph.add_edge("investigate", END)
    graph.add_edge("report", END)

    return graph.compile()
