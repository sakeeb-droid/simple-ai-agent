import os
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import START, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from ToolSet import toolset

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY is None:
    raise RuntimeError("API not set in environment")

llm = ChatGoogleGenerativeAI(
    # model="gemini-2.0-flash",
    model = "gemini-2.5-flash-preview-04-17",
    temperature=0,
    google_api_key = API_KEY
)

llm_with_tools = llm.bind_tools(toolset)

sys_prompt_file = open("system_prompt.txt")
sys_prompt = sys_prompt_file.read()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke([sys_prompt]+state["messages"])],
    }

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(toolset))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge("tools","assistant")

simple_agent = builder.compile()
