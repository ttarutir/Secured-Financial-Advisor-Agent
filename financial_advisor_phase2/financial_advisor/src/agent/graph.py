"""LangGraph agent definition for the baseline Financial Advisor agent.

Pipeline: retrieve_documents → generate_response → finalize_response
No guard node — this is the vulnerable baseline for ASR benchmarking.
"""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve_documents,
    generate_response,
    finalize_response,
)


def build_agent_graph() -> StateGraph:
    """Build and compile the baseline financial advisor LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("finalize_response", finalize_response)

    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", "finalize_response")
    workflow.add_edge("finalize_response", END)

    return workflow.compile()


# Singleton compiled graph
agent = build_agent_graph()
