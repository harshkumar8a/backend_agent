"""Unit tests for tool schemas, model tool calls, and API boundaries.

Run from the repository root with:
    python -m unittest discover -s test\\unitTest -p test.py -v
"""

import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Application settings require this value during import. It is not sent to any
# service by these unit tests.
os.environ.setdefault("TAVILY_API_KEY", "unit-test-key")

import app.services.tavily_service as tavily_service
from app.core.config import settings
from app.graph.nodes import (
    MAX_TOOL_CONTEXT_CHARS,
    _compact_tool_messages,
    _format_tool_content,
    _is_unknown_tool_name_error,
    create_answer_node,
)
from app.graph.workflow import create_agent_graph
from main import run_agent

from groq import BadRequestError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool


class SearchArguments(BaseModel):
    """Test schema matching the shape expected for Tavily web search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3)
    max_results: int = Field(default=5, ge=1, le=10)


class FakeSearchTool:
    name = "tavily_search"
    description = "Search the web."
    args_schema = SearchArguments
    return_direct = False

    def __init__(self, result=None):
        self.ainvoke = AsyncMock(
            return_value=(
                result
                if result is not None
                else [{"title": "Example", "url": "https://example.test"}]
            )
        )


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bind_calls = []
        self.prompt_messages = []

    def bind_tools(self, tools, **kwargs):
        self.bind_calls.append(
            ([tool.name for tool in tools], kwargs)
        )
        return self

    async def ainvoke(self, messages):
        self.prompt_messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_unknown_tool_error():
    return BadRequestError(
        "invalid tool call",
        response=tavily_service.httpx.Response(
            400,
            request=tavily_service.httpx.Request(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
            ),
        ),
        body={
            "error": {
                "code": "tool_use_failed",
                "message": (
                    "attempted to call tool 'savily_search' "
                    "which was not in request.tools"
                ),
            }
        },
    )


class SchemaAndToolTests(unittest.TestCase):
    def test_search_schema_accepts_valid_arguments(self):
        args = SearchArguments(
            query="latest AI news",
            max_results=5,
        )

        self.assertEqual(args.model_dump(), {
            "query": "latest AI news",
            "max_results": 5,
        })

    def test_search_schema_rejects_invalid_and_unknown_arguments(self):
        with self.assertRaises(ValidationError):
            SearchArguments(query="AI", max_results=5)

        with self.assertRaises(ValidationError):
            SearchArguments(query="latest news", max_results=11)

        with self.assertRaises(ValidationError):
            SearchArguments(
                query="latest news",
                unexpected_parameter=True,
            )

    def test_unknown_tool_error_detection_is_specific(self):
        error = make_unknown_tool_error()

        self.assertTrue(_is_unknown_tool_name_error(error))
        self.assertRegex(
            error.body["error"]["message"],
            re.compile(r"savily_search.*not in request\.tools"),
        )

        unrelated_error = BadRequestError(
            "invalid request",
            response=tavily_service.httpx.Response(
                400,
                request=tavily_service.httpx.Request(
                    "POST",
                    "https://api.groq.com",
                ),
            ),
            body={
                "error": {
                    "code": "invalid_request_error",
                    "message": "invalid request",
                }
            },
        )
        self.assertFalse(_is_unknown_tool_name_error(unrelated_error))

    def test_compacted_tool_messages_obey_context_limit(self):
        messages = [
            ToolMessage(
                content="search result " * 1000,
                tool_call_id="call-1",
            ),
            ToolMessage(
                content="second result",
                tool_call_id="call-2",
            ),
        ]

        compacted = _compact_tool_messages(messages)

        total_content_length = sum(
            len(message.content)
            for message in compacted
            if isinstance(message, ToolMessage)
        )
        self.assertLessEqual(total_content_length, MAX_TOOL_CONTEXT_CHARS)
        self.assertRegex(
            compacted[0].content,
            r"\[Search results truncated to limit context\.\]$",
        )
        self.assertEqual(messages[0].content, "search result " * 1000)

    def test_mcp_text_block_becomes_readable_search_result_text(self):
        mcp_content = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "results": [
                            {
                                "title": "New AI infrastructure report",
                                "url": "https://example.test/ai",
                                "content": "Researchers describe a recent trend.",
                            }
                        ]
                    }
                ),
            }
        ]

        formatted = _format_tool_content(mcp_content)

        self.assertIn("Title: New AI infrastructure report", formatted)
        self.assertIn("URL: https://example.test/ai", formatted)
        self.assertIn("Snippet: Researchers describe a recent trend.", formatted)
        self.assertNotIn('"type": "text"', formatted)


class ToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_uses_input_and_output_state(self):
        model = FakeModel([AIMessage(content="A concise answer.")])
        graph = create_agent_graph(model, [])

        result = await graph.ainvoke({
            "input": "A general question",
            "messages": [],
        })

        self.assertEqual(result["input"], "A general question")
        self.assertEqual(result["output"], "A concise answer.")
        self.assertEqual(
            [message.content for message in result["messages"]],
            ["A general question", "A concise answer."],
        )

    async def test_graph_preserves_state_through_tool_execution(self):
        async def fake_search(query: str, max_results: int = 5):
            return [{"title": query, "url": "https://example.test"}]

        search_tool = StructuredTool(
            name="tavily_search",
            description="Search test results.",
            args_schema=SearchArguments,
            coroutine=fake_search,
            response_format="content",
        )
        model = FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tavily_search",
                            "args": {"query": "latest AI news"},
                            "id": "graph-call-1",
                        }
                    ],
                ),
                AIMessage(content="AI news summary."),
            ]
        )
        graph = create_agent_graph(model, [search_tool])

        result = await graph.ainvoke({
            "input": "latest AI news",
            "messages": [],
        })

        self.assertEqual(result["input"], "latest AI news")
        self.assertEqual(result["output"], "AI news summary.")
        self.assertEqual(
            [message.name for message in result["messages"]
             if isinstance(message, ToolMessage)],
            ["tavily_search"],
        )
        final_prompt = model.prompt_messages[-1]
        self.assertFalse(
            any(isinstance(message, ToolMessage) for message in final_prompt)
        )
        self.assertFalse(
            any(
                isinstance(message, AIMessage) and message.tool_calls
                for message in final_prompt
            )
        )
        self.assertIn("Do not call or request any tools", final_prompt[0].content)
        self.assertRegex(
            final_prompt[-1].content,
            r"(?s)^Search results:\s+.*https://example\.test",
        )

    async def test_model_idk_triggers_tavily_fallback(self):
        searched_queries = []

        async def fake_search(query: str, max_results: int = 5):
            searched_queries.append(query)
            return [{"title": "Recent IT news", "url": "https://example.test"}]

        search_tool = StructuredTool(
            name="tavily_search",
            description="Search the web.",
            args_schema=SearchArguments,
            coroutine=fake_search,
            response_format="content",
        )
        model = FakeModel(
            [
                AIMessage(content="I don't know."),
                AIMessage(content="Here are recent IT industry updates."),
            ]
        )
        graph = create_agent_graph(model, [search_tool])
        query = "Recent IT industry news"

        result = await graph.ainvoke({"input": query, "messages": []})

        self.assertEqual(searched_queries, [query])
        self.assertEqual(
            result["output"],
            "Here are recent IT industry updates.",
        )
        self.assertEqual(len(model.prompt_messages), 2)

    async def test_insufficient_search_results_do_not_trigger_repeat_search(self):
        searched_queries = []

        async def fake_search(query: str, max_results: int = 5):
            searched_queries.append(query)
            return []

        search_tool = StructuredTool(
            name="tavily_search",
            description="Search the web.",
            args_schema=SearchArguments,
            coroutine=fake_search,
            response_format="content",
        )
        model = FakeModel(
            [
                AIMessage(content="I don't know."),
                AIMessage(content="I don't know."),
            ]
        )
        graph = create_agent_graph(model, [search_tool])

        result = await graph.ainvoke({
            "input": "A difficult current question",
            "messages": [],
        })

        self.assertEqual(searched_queries, ["A difficult current question"])
        self.assertEqual(result["output"], "I don't know.")
        self.assertEqual(len(model.prompt_messages), 2)

    async def test_answer_node_binds_registered_search_tool(self):
        tool = FakeSearchTool()
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tavily_search",
                    "args": {
                        "query": "AI news this week",
                        "max_results": 5,
                    },
                    "id": "call-1",
                }
            ],
        )
        model = FakeModel([response])
        answer_node = create_answer_node(model, [tool])

        result = await answer_node(
            {"messages": [HumanMessage(content="AI news this week")]}
        )

        self.assertEqual(
            model.bind_calls,
            [(["tavily_search"], {"tool_choice": "auto"})],
        )
        self.assertEqual(
            result["messages"][0].tool_calls[0]["name"],
            "tavily_search",
        )
        self.assertNotIn("output", result)
        self.assertEqual(
            SearchArguments.model_validate(
                result["messages"][0].tool_calls[0]["args"]
            ).max_results,
            5,
        )

    async def test_unknown_model_tool_call_retries_with_registered_tool(self):
        tool = FakeSearchTool()
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tavily_search",
                    "args": {"query": "AI news this week"},
                    "id": "call-2",
                }
            ],
        )
        model = FakeModel([make_unknown_tool_error(), response])
        answer_node = create_answer_node(model, [tool])

        result = await answer_node(
            {"messages": [HumanMessage(content="AI news this week")]}
        )

        self.assertEqual(len(model.prompt_messages), 2)
        self.assertEqual(
            model.bind_calls,
            [
                (["tavily_search"], {"tool_choice": "auto"}),
                (["tavily_search"], {"tool_choice": "tavily_search"}),
            ],
        )
        self.assertEqual(
            result["messages"][0].tool_calls[0]["name"],
            "tavily_search",
        )

    async def test_unrelated_groq_bad_request_is_not_retried(self):
        tool = FakeSearchTool()
        error = BadRequestError(
            "invalid request",
            response=tavily_service.httpx.Response(
                400,
                request=tavily_service.httpx.Request(
                    "POST",
                    "https://api.groq.com",
                ),
            ),
            body={
                "error": {
                    "code": "invalid_request_error",
                    "message": "invalid request",
                }
            },
        )
        model = FakeModel([error])
        answer_node = create_answer_node(model, [tool])

        with self.assertRaises(BadRequestError):
            await answer_node(
                {"messages": [HumanMessage(content="AI news this week")]}
            )

        self.assertEqual(len(model.prompt_messages), 1)
        self.assertEqual(len(model.bind_calls), 1)


class ApiBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_tavily_client_uses_configured_endpoint_and_auth(self):
        with patch.object(tavily_service, "Client") as client_class:
            client = tavily_service.create_tavily_client()

        self.assertIs(client, client_class.return_value)
        client_class.assert_called_once_with(
            "https://mcp.tavily.com/mcp",
            auth=settings.tavily_api_key,
        )

    async def test_instrumented_tool_validates_and_forwards_api_arguments(self):
        api_result = [{"title": "AI news", "url": "https://example.test/news"}]
        original_tool = FakeSearchTool(result=api_result)
        wrapped_tool = tavily_service.instrument_tool(original_tool)

        validated = wrapped_tool.args_schema.model_validate(
            {"query": "AI news", "max_results": 3}
        )
        result = await wrapped_tool.ainvoke(validated.model_dump())

        original_tool.ainvoke.assert_awaited_once_with({
            "query": "AI news",
            "max_results": 3,
        })
        self.assertEqual(result, api_result)

    async def test_discovery_exposes_only_standard_search_tool(self):
        search_tool = FakeSearchTool()
        research_tool = FakeSearchTool()
        research_tool.name = "tavily_research"
        non_search_tool = FakeSearchTool()
        non_search_tool.name = "tavily_extract"

        class FakeAdapter:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def list_tools(self):
                return [search_tool, research_tool, non_search_tool]

        with patch.object(
            tavily_service,
            "MCPAdapter",
            return_value=FakeAdapter(),
        ):
            tools = await tavily_service.discover_search_tools(object())

        self.assertEqual([tool.name for tool in tools], ["tavily_search"])

    async def test_run_agent_returns_final_answer_from_graph(self):
        graph = type(
            "FakeGraph",
            (),
            {
                "ainvoke": AsyncMock(
                    return_value={
                        "messages": [
                            HumanMessage(content="What is AI?"),
                            AIMessage(content="AI is artificial intelligence."),
                        ],
                        "output": "AI is artificial intelligence.",
                    }
                )
            },
        )()

        answer = await run_agent(graph, "What is AI?")

        self.assertEqual(answer, "AI is artificial intelligence.")
        graph.ainvoke.assert_awaited_once()
        graph_input = graph.ainvoke.await_args.args[0]
        self.assertEqual(
            graph_input,
            {"input": "What is AI?", "messages": []},
        )


if __name__ == "__main__":
    unittest.main()