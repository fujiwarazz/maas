from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END,START
from typing import Dict, Any
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI




def add_tool(a: int, b: int) -> int:
    """a+b

    Args:
        a (int): a
        b (int): b

    Returns:
        int: a+b result
    """
    return a + b

llm = ChatOpenAI(model="qwen-plus",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 api_key="sk-0e349a8dc24443988825b69a56d2b868"
                 )

llm=llm.bind_tools([add_tool])

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


add_tool_node = ToolNode([add_tool])
    
# 简单的llm node，模拟llm调用工具
def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["messages"]
    
    result = llm.invoke(query)
    return {"messages": [result]}
    
    
def test_tool_and_llm_node_integration():
    # 构建 StateGraph
    builder = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    builder.add_node("call_model", llm_node)
    builder.add_node("tools", add_tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")

    graph = builder.compile()
    result = graph.invoke({"messages": [{"role":"user","content":"What is 3 + 5?"}]})
    
    
    print(result['messages'][-1],type(result['messages'][-1]))
    print("\n")
    print(result['messages'])
test_tool_and_llm_node_integration()


