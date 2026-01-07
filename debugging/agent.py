from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

from langchain_ollama import OllamaLLM

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from dotenv import load_dotenv
import os

load_dotenv()

# Optional LangSmith tracking
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Another way to initialize model
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="llama3.2:latest", model_provider="ollama")

class State(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def make_tool_graph():
    @tool
    def add(a:float,b:float)->float:
        """ __summary__
        add two numbers
        """
        return a+b

    tools = [add]

    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    # ==================Creating state graph=========================

    ## Node defination

    def call_llm_model(state: State):
        return {
            "messages": llm_with_tools.invoke(state["messages"])
        }


    ## Creating graph

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))

    ## Adding edges

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools","tool_calling_llm")

    ## Compiling the graph and viweing

    graph = builder.compile()

    return graph

tool_agent = make_tool_graph()

"""langgraph dev"""