
import logging
import time

import httpx

from fastmcp.client import Client
from langchain.mcp import MCPAdapter
from langchain_core.tools import StructuredTool

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception_type,
)

from app.core.config import settings


logger = logging.getLogger(__name__)


# MCP CLIENT

def create_tavily_client() -> Client:
    """
    Create an authenticated Tavily MCP client.

    Returns:
        Client: Configured Tavily MCP client.
    """

    logger.info("Initializing Tavily MCP client.")

    return Client(
        "https://mcp.tavily.com/mcp",
        auth=settings.tavily_api_key,
    )


# TOOL INSTRUMENTATION

def instrument_tool(tool):
    """
    Wrap an MCP tool with execution-time logging.

    Measures:
        - Tool execution duration.
        - Successful execution.
        - Failed execution.

    Args:
        tool: Original LangChain MCP tool.

    Returns:
        StructuredTool: Instrumented LangChain tool.
    """

    async def execute_tool(**kwargs):
        """Execute MCP tool and measure execution time."""

        start_time = time.perf_counter()

        logger.info(
            "MCP tool execution started | tool=%s",
            tool.name,
        )

        try:
            result = await tool.ainvoke(kwargs)

            duration = time.perf_counter() - start_time

            logger.info(
                "MCP tool execution completed | "
                "tool=%s | duration=%.3fs",
                tool.name,
                duration,
            )

            # Log Tavily's own response time when available.
            if isinstance(result, dict):
                tavily_time = result.get("response_time")

                if tavily_time is not None:
                    logger.info(
                        "Tavily reported ponse time | "
                        "tool=%s | response_time=%s",
                        tool.name,
                        tavily_time,
                    )

            return result

        except Exception:
            duration = time.perf_counter() - start_time

            logger.exception(
                "MCP tool execution failed | "
                "tool=%s | duration=%.3fs",
                tool.name,
                duration,
            )

            raise

    logger.info(
        "Instrumenting MCP tool | tool=%s",
        tool.name,
    )

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=execute_tool,
        return_direct=getattr(tool, "return_direct", False),
        response_format="content"
    )


# SEARCH TOOL DISCOVERY

@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=10,
    ),
    retry=retry_if_exception_type(
        (
            TimeoutError,
            ConnectionError,
            OSError,
            httpx.TimeoutException,
            httpx.TransportError,
        )
    ),
    before_sleep=before_sleep_log(
        logger,
        logging.WARNING,
    ),
    reraise=True,
)

async def discover_search_tools(client: Client) -> list:
    """
    Connect to Tavily MCP and discover search tools.

    Returns:
        list: Instrumented Tavily search tools.

    Raises:
        RuntimeError: If no search tools are available.
    """

    logger.info("Connecting to Tavily MCP server.")

    start_time = time.perf_counter()

    try:
        async with MCPAdapter(client) as adapter:

            logger.info(
                "MCP connection established. Discovering tools."
            )

            available_tools = await adapter.list_tools()

        duration = time.perf_counter() - start_time

        logger.info(
            "MCP tool discovery completed | "
            "duration=%.3fs | total_tools=%d",
            duration,
            len(available_tools),
        )

        # Use the standard search tool; research is a separate, broader operation.
        search_tools = [
            tool
            for tool in available_tools
            if tool.name.casefold() == "tavily_search"
        ]

        if not search_tools:
            logger.error(
                "No search tools found on Tavily MCP server."
            )

            raise RuntimeError(
                "No search tools available from Tavily MCP."
            )

        logger.info(
            "Discovered %d search tools: %s",
            len(search_tools),
            [tool.name for tool in search_tools],
        )

        # Add execution monitoring to each tool.
        instrumented_tools = [
            instrument_tool(tool)
            for tool in search_tools
        ]

        logger.info(
            "Successfully instrumented %d Tavily tools.",
            len(instrumented_tools),
        )

        return instrumented_tools

    except Exception:
        logger.exception(
            "Failed to discover Tavily search tools."
        )
        raise


# HEALTH CHECK

async def check_tavily_connection() -> bool:
    """
    Check whether Tavily MCP is accessible.

    Returns:
        bool: True if tool discovery succeeds.
    """

    logger.info("Checking Tavily MCP connectivity.")

    try:
        client = create_tavily_client()

        tools = await discover_search_tools(client)

        logger.info(
            "Tavily MCP health check successful | tools=%d",
            len(tools),
        )

        return True

    except Exception:
        logger.exception(
            "Tavily MCP health check failed."
        )

        return False
