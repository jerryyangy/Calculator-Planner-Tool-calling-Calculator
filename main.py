
# agent_calculator_graph.py
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---- Tool: calculator ----
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a numeric Python expression safely. Supports + - * / ** and parentheses.
    Example: "12 * (3 + 5) / 2"
    """
    code = compile(expression, "<expr>", "eval")
    if code.co_names:
        return "ERROR: Only numeric expressions (no variables/functions) are allowed."
    try:
        result = eval(code, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"

# ---- State ----
class AgentState(TypedDict):
    messages: List[Any]  # list of HumanMessage/AIMessage/ToolMessage
    final_answer: str | None

# ---- LLM ----
# Replace with your model and API key in environment (OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind tool
llm_with_tools = llm.bind_tools([calculator])

# ---- Nodes ----
def planner_node(state: AgentState) -> AgentState:
    """
    Let the LLM decide: answer directly or call the calculator tool.
    """
    resp = llm_with_tools.invoke(state["messages"])
    # Append assistant message (could contain tool call)
    new_messages = state["messages"] + [resp]
    return {"messages": new_messages, "final_answer": None}

def tool_node(state: AgentState) -> AgentState:
    """
    If the last assistant message requested a tool call, execute it and append ToolMessage.
    """
    last = state["messages"][-1]
    # Check for tool call
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return state

    tool_call = last.tool_calls[0]
    if tool_call["name"] == "calculator":
        expr = tool_call["args"].get("expression", "")
        tool_result = calculator(expr)
        tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
        return {"messages": state["messages"] + [tool_msg], "final_answer": None}
    return state

def finalizer_node(state: AgentState) -> AgentState:
    """
    Ask the LLM to produce a concise final answer based on the conversation + tool result.
    """
    resp = llm.invoke(state["messages"] + [HumanMessage(content="Provide the final numeric answer (and brief steps if helpful).")])
    final = resp.content
    return {"messages": state["messages"] + [resp], "final_answer": final}

# ---- Routing ----
def should_call_tool(state: AgentState) -> str:
    """
    Decide whether to call `tool_node` next or finalize.
    If the last assistant message has a tool call, go to tool; otherwise, finalize.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return "finalize"

# ---- Build graph ----
builder = StateGraph(AgentState)
builder.add_node("plan", planner_node)
builder.add_node("tool", tool_node)
builder.add_node("finalize", finalizer_node)

builder.set_entry_point("plan")
builder.add_conditional_edges("plan", should_call_tool, {"tool": "tool", "finalize": "finalize"})
# After running tool, go to finalize
builder.add_edge("tool", "finalize")
# End after finalize
builder.add_edge("finalize", END)

app = builder.compile()

if __name__ == "__main__":
    # Example query
    user_question = "Compute 12 * (3 + 5) / 2 and subtract 7."
    init_state: AgentState = {"messages": [HumanMessage(content=user_question)], "final_answer": None}
    result = app.invoke(init_state)
    print("Final answer:", result["final_answer"])
    print("\n--- Trace messages ---")
    for m in result["messages"]:
        role = m.type if hasattr(m, "type") else m.__class__.__name__
        print(f"[{role}] {getattr(m, 'content', m)}")
