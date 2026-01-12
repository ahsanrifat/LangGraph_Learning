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

document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Update the document content with the given content."""
    global document_content
    document_content = content
    return f"Document content updated: {document_content}"


@tool
def save(filename: str) -> str:
    """Save the document content to a text file and finishes the process.

    Args:
        filename: The name of the file to save the document content to.
    """
    try:
        if not filename.endswith(".txt"):
            filename += ".txt"
        global document_content
        with open(filename, "w", encoding="utf-8") as f:
            f.write(document_content)
        return f"Document content saved to {filename}"
    except Exception as e:
        return f"Error saving document: {e}"


tools = [update, save]

from aws_llm import llm

llm2 = llm.bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm2.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}



def should_continue(state: AgentState) -> str:
    """Determine if the conversation should continue or end the conversation based on the last message."""
    messages = state["messages"]
    if not messages:
        return "continue"
    # this looks for the most recent tool message and if it is a save tool, then end the conversation
    for message in reversed(messages):
        if (isinstance(message, ToolMessage)) and message.name == "save":
            return "end"
    return "continue"


def print_messages(messages):
    """Print the messages in a readable format."""
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"Tool: {message.name}")
            print(f"Tool Response: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("our_agent", our_agent)
graph.add_node("tool_node", ToolNode(tools=tools))
graph.add_edge("our_agent","tool_node")
graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue": "our_agent",
        "end": END,
    },
)

app = graph.compile()


def run_document_agent():
    print("\n Starting document agent...")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n Document agent finished.")


run_document_agent()
