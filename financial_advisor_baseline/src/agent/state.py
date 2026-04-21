"""Agent state schema for the Financial Advisor LangGraph agent."""

from typing import TypedDict


class AgentState(TypedDict):
    """State passed through the LangGraph financial advisor agent."""

    # Input
    user_query: str           # Free-form query including client context

    # Internal processing
    retrieved_chunks: list[dict]   # Retrieved policy/profile/market chunks
    raw_response: str              # LLM response

    # Output
    final_answer: str              # Final recommendation
    cited_documents: list[dict]    # Documents cited in the response
    error: str | None
