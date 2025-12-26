from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or "gemini-1.5-pro", etc.
    temperature=0,  # For more deterministic responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    finalNumber1: int
    number3: int
    number4: int
    finalNumber2: int


def adder(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""
    state["finalNumber1"] = state["number1"] + state["number2"]

    return state

def adder1(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""
    state["finalNumber2"] = state["number3"] + state["number4"]

    return state

def subtractor(state: AgentState) -> AgentState:
    """This node subtracts the 2 numbers"""
    state["finalNumber"] = state["number1"] - state["number2"]
    return state

def subtractor1(state: AgentState) -> AgentState:
    """This node subtracts the 2 numbers"""
    state["finalNumber2"] = state["number3"] - state["number4"]
    return state


def decide_next_node(state: AgentState) -> AgentState:
    """This node will select the next node of the graph"""

    if state["operation"] == "+":
        return "addition_operation"

    elif state["operation"] == "-":
        return "subtraction_operation"


graph = StateGraph(AgentState)

graph.add_node(node="add_node", action=adder)
graph.add_node(node="subtract_node", action=subtractor)
graph.add_node(node="add_node1", action=adder1)
graph.add_node(node="subtract_node1", action=subtractor1)
# pass through function
# input and output will be the same state
# no changing is done to the state
graph.add_node(node="router_node", action=lambda state: state)

graph.add_edge(START, "router_node")
graph.add_conditional_edges(
    source="router_node",
    path=decide_next_node,  # routing function that will decide the next node
    path_map={
        # Edge : Node
        "addition_operation": "add_node",
        "subtraction_operation": "subtract_node",
    },
)
graph.add_edge("add_node", "router_node")
graph.add_edge("subtract_node", "router_node")

graph.add_conditional_edges(
    source="router_node",
    path=decide_next_node,  # routing function that will decide the next node
    path_map={
        # Edge : Node
        "addition_operation": "add_node1",
        "subtraction_operation": "subtract_node1",
    },
)


app = graph.compile()

result = app.invoke(AgentState(number1=10, operation="+", number2=5, finalNumber=0))
print(result)
