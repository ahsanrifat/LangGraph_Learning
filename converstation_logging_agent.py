import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from aws_llm import llm

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