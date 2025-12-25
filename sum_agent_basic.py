from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Or "gemini-1.5-pro", etc.
    temperature=0,  # For more deterministic responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# create a form of memory for the agent
class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str


def process_values(state: AgentState) -> AgentState:
    state["result"] = (
        f"Hi There {state['name']}, the sum of your values is {sum(state['values'])}"
    )
    return state


graph = StateGraph(AgentState)
graph.add_node("process_values", process_values)
graph.add_edge(START, "process_values")
graph.add_edge("process_values", END)
agent = graph.compile()
result = agent.invoke(AgentState(values=[1, 2, 3, 4, 5], name="Rifat"))
print(result)
