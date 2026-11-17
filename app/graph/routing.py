
import logging
import re

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

from app.graph.state import State

logger = logging.getLogger(__name__)


def is_unknown_answer(message: AIMessage) -> bool:
    """Return whether the model explicitly declined to answer."""

    content = message.content
    if not isinstance(content, str):
        return False

    return re.match(
        r"^\s*i\s+don['’]t\s+know(?:\b|[.!?])",
        content,
        flags=re.IGNORECASE,
    ) is not None


def route_after_answer(state: State) -> str:
    """Route to tools or terminate the graph."""

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            logger.info(
                "Routing to tools. Number of tool calls: %d",
                len(last_message.tool_calls),
            )
            return "tools"

        has_search_result = any(
            isinstance(message, ToolMessage)
            for message in state["messages"]
        )
        if not has_search_result and is_unknown_answer(last_message):
            logger.info("Model declined to answer. Routing to fallback search.")
            return "fallback_search"

    logger.info("No more tool calls. Ending graph.")

    return END
