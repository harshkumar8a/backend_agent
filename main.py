
import asyncio
import logging
import uuid

from app.core.config import settings
from app.core.logging import setup_logging

from app.services.tavily_service import (
    create_tavily_client,
    discover_search_tools,
     check_tavily_connection,
)

from app.services.llm_service import create_llm

from app.graph.workflow import create_agent_graph


logger = logging.getLogger(__name__)


async def build_agent():
    """Initialize all dependencies and build the agent."""

    logger.info("Initializing %s", settings.app_name)

    # Initialize MCP.
    client = create_tavily_client()

    search_tools = await discover_search_tools(client)

    # Initialize LLM.
    model = create_llm()

    # Build graph.
    agent = create_agent_graph(
        model=model,
        search_tools=search_tools,
    )

    logger.info("Agent initialized successfully.")

    return agent


async def run_agent(agent, query: str):
    """Execute the agent with a user query."""

    request_id = str(uuid.uuid4())

    logger.info(
        "[%s] Processing query.",
        request_id,
    )

    config = {
        "recursion_limit": settings.max_graph_steps,
        "run_name": "Tavily Research Agent",
        "tags": [
            settings.environment,
            "tavily",
            "ollama",
        ],
        "metadata": {
            "request_id": request_id,
            "application": settings.app_name,
        },
    }

    try:
        result = await agent.ainvoke(
            {"input": query, "messages": []},
            config=config,
        )

        answer = result["output"]

        logger.info(
            "[%s] Request completed successfully.",
            request_id,
        )

        return answer

    except Exception:
        logger.exception(
            "[%s] Agent execution failed.",
            request_id,
        )
        raise


async def main():
    """Application entry point."""

    setup_logging()

    logger.info("Starting application.")

    try:
        agent = await build_agent()

        while True:
            query = input("\nAsk a question (or 'exit'): ").strip()

            if query.casefold() in {"exit", "quit"}:
                break

            if not query:
                continue

            response = await run_agent(agent, query)

            print("\nAnswer:")
            print(response)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")

    except Exception:
        logger.exception("Application terminated unexpectedly.")
        raise

    finally:
        logger.info("Application shutting down.")


if __name__ == "__main__":
    asyncio.run(main())
