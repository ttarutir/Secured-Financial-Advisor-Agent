"""Agent state schema for the Financial Advisor LangGraph agent.

Shared by both the baseline agent and the secured (guard node) agent.
Guard-specific fields are populated only in the secured pipeline.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """State passed through the LangGraph financial advisor agent."""

    # Input
    user_query: str                 # Free-form query including client context

    # Internal processing
    retrieved_chunks: list[dict]    # Retrieved policy/profile/market chunks
    raw_response: str               # LLM response

    # Output
    final_answer: str               # Final recommendation
    cited_documents: list[dict]     # Documents cited in the response
    error: str | None

    # Guard node fields (populated only in the secured agent pipeline)
    injection_detected: bool        # True if any injection was found
    flagged_chunks: list[dict]      # Chunks that triggered the guard
    guard_log: list[str]            # Human-readable log of guard decisions
