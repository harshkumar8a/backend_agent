import os
from typing import Literal

from dotenv import load_dotenv
from fastmcp.client import Client
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

SYSTEM_PROMPT = """You answer questions using web search results from Tavily.
Use the search results as your only source of factual information. If they do
not directly and reliably answer the question, or are empty or conflicting,
respond exactly: I don't know.
For supported answers, be concise and cite the source URLs from the results.
Never guess or fill gaps with prior knowledge."""


async def build_agent():
    """Connect to Tavily MCP and return a compiled LangGraph question-answering agent."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY is required to connect to Tavily MCP.")

    client = Client("https://mcp.tavily.com/mcp", auth=tavily_api_key)
    async with MCPAdapter(client) as adapter:
        available_tools = await adapter.list_tools()

    search_tools = [tool for tool in available_tools if "search" in tool.name.casefold()]
    if not search_tools:
        raise RuntimeError("The Tavily MCP server did not provide a search tool.")

    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        temperature=0,
    )
    tool_node = ToolNode(search_tools)
    graph = StateGraph(MessagesState)

    async def answer(state: MessagesState):
        has_search_result = any(
            isinstance(message, ToolMessage) for message in state["messages"]
        )
        model_with_tools = model.bind_tools(
            search_tools,
            tool_choice=None if has_search_result else "any",
        )
        response = await model_with_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    def route_after_answer(state: MessagesState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph.add_node("answer", answer)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "answer")
    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "answer")

    return graph.compile()