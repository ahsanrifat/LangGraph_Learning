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
    messages: List[Union[HumanMessage, AIMessage]]


# Defining Node
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print(f"Current State: {state['messages']}")


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    start_state = AgentState(messages=conversation_history)
    result = agent.invoke(start_state)
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
print("Converstation saved in logging.txt file")
