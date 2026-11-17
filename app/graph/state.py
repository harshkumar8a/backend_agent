from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    input: str
    output: NotRequired[str]
    messages: Annotated[list[AnyMessage], add_messages]
