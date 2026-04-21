"""LangGraph agent definition for the SECURED Financial Advisor agent (Phase 2).

Pipeline with guard node:

  guard_input ──► [injection?] ──YES──► block_response ──► END
                       │
                       NO
                       ▼
              retrieve_documents
                       │
                       ▼
               guard_chunks ──► [injection?] ──YES──► block_response ──► END
                       │
                       NO
                       ▼
         generate_response_hardened
                       │
                       ▼
              finalize_response
                       │
                       ▼
                      END
"""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import retrieve_documents          # unchanged baseline node
from src.agent.nodes_guard import (
    guard_input,
    guard_chunks,
    block_response,
    generate_response_hardened,
    finalize_response,
)


def route_after_input_guard(state: AgentState) -> str:
    """Route to block or retrieval based on input guard result."""
    if state.get("injection_detected"):
        return "block"
    return "retrieve"


def route_after_chunk_guard(state: AgentState) -> str:
    """Route to block or generation based on chunk guard result."""
    if state.get("injection_detected"):
        return "block"
    return "generate"


def build_secured_agent_graph() -> StateGraph:
    """Build and compile the secured financial advisor LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("guard_input",                guard_input)
    workflow.add_node("retrieve_documents",          retrieve_documents)
    workflow.add_node("guard_chunks",               guard_chunks)
    workflow.add_node("block_response",             block_response)
    workflow.add_node("generate_response_hardened", generate_response_hardened)
    workflow.add_node("finalize_response",          finalize_response)

    # Entry point
    workflow.set_entry_point("guard_input")

    # After input guard — branch on injection detected
    workflow.add_conditional_edges(
        "guard_input",
        route_after_input_guard,
        {
            "block":    "block_response",
            "retrieve": "retrieve_documents",
        },
    )

    # After retrieval — inspect chunks
    workflow.add_edge("retrieve_documents", "guard_chunks")

    # After chunk guard — branch on injection detected
    workflow.add_conditional_edges(
        "guard_chunks",
        route_after_chunk_guard,
        {
            "block":    "block_response",
            "generate": "generate_response_hardened",
        },
    )

    # Block → END (no LLM call)
    workflow.add_edge("block_response", END)

    # Clean path → finalize → END
    workflow.add_edge("generate_response_hardened", "finalize_response")
    workflow.add_edge("finalize_response", END)

    return workflow.compile()


# Singleton compiled graph
secured_agent = build_secured_agent_graph()
