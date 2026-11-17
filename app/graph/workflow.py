
import logging

from langgraph.graph import (
    START,
    END,
    StateGraph,
)
from langgraph.prebuilt import ToolNode

from app.graph.nodes import (
    create_answer_node,
    create_fallback_search_node,
)
from app.graph.routing import route_after_answer
from app.graph.state import State

logger = logging.getLogger(__name__)


def create_agent_graph(model, search_tools):
    """Construct and compile the LangGraph workflow."""

    logger.info("Building LangGraph workflow.")

    graph = StateGraph(State)

    # Nodes
    graph.add_node(
        "answer",
        create_answer_node(model, search_tools),
    )

    graph.add_node(
        "tools",
        ToolNode(search_tools),
    )

    graph.add_node(
        "fallback_search",
        create_fallback_search_node(search_tools),
    )

    # Edges
    graph.add_edge(START, "answer")

    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "tools": "tools",
            "fallback_search": "fallback_search",
            END: END,
        },
    )

    graph.add_edge("tools", "answer")
    graph.add_edge("fallback_search", "answer")

    compiled_graph = graph.compile()

    logger.info("LangGraph compiled successfully.")

    return compiled_graph
