from typing import Annotated, Sequence

# Message for providing instructions to LLM
from langchain_core.messages import SystemMessage

# passes data back to LLM after it calls a tool
from langchain_core.messages import ToolMessage

# The foundational class for all message types in LangGraph
from langchain_core.messages import BaseMessage

from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from typing import TypedDict, List, Union

# Lang Graph Imports
from langgraph.graph import StateGraph, START, END

# Langchin Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    # preserve the state by appending. Do not override
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """This tool adds two numbers together."""
    return a + b


@tool
def date_tool() -> str:
    """Returns the current date."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


tools = [add, date_tool]

from aws_llm import llm

llm2 =llm.bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm2.invoke([system_prompt] + state["messages"])
    print(f"Invoking Message======> {[system_prompt] + state["messages"]}")
    print(f"Model Response======> {response.content}")
    ## response will be appended by reducer function no need to append manually
    return {"messages": response}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # if the last message is not a tool call, then end the conversation
    if not last_message.tool_calls:
        return "end"
    # if the last message is a tool call, then continue the conversation
    return "continue"


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


graph = StateGraph(AgentState)
graph.add_node("model_call", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("model_call")

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue": "tool_node",
        "end": END,
    },
)
graph.add_edge("tool_node", "model_call")

app = graph.compile()

inputs = {"messages": [("user", "what is the current date?")]}
print_stream(app.stream(inputs, stream_mode="values"))