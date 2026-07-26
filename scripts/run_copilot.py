"""
Interactive CLI for the SentinelAI SOC Copilot.

Usage:
    python scripts/run_copilot.py

Requires:
    OPENAI_API_KEY set in a .env file at the project root (see .env.example),
    or already present in your environment.

    All artifacts from prior stages (features.csv, risk_scores.csv,
    entity_baselines.json, attack_chains.json, models/*, RAG index) must
    already exist -- run the earlier scripts first if you haven't.

Try asking things like:
    Why was USER_038 flagged?
    Explain alert EVT-0000410.
    Show USER_038's previous 5 logins.
    Has device DEV_631 been used before by this entity?
    Are there related suspicious events?
    Could this be a false positive?
    Generate an incident report for EVT-0000410.

Type 'exit' or 'quit' to stop.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not set.")
    print("Copy .env.example to .env and fill in your real key, or export it "
          "in your shell before running this script.")
    sys.exit(1)

from langchain_core.messages import HumanMessage, AIMessage
from src.agents.graph import build_graph


def main():
    print("=" * 70)
    print("SENTINELAI SOC COPILOT")
    print("=" * 70)
    print("Type your question, or 'exit'/'quit' to stop.\n")

    graph = build_graph(model_name="gpt-4o-mini")
    state = {"messages": [], "route": ""}

    while True:
        try:
            user_input = input("Analyst> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))

        try:
            state = graph.invoke(state)
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}\n")
            continue

        last_message = state["messages"][-1]
        print(f"\nCopilot> {last_message.content}\n")


if __name__ == "__main__":
    main()
