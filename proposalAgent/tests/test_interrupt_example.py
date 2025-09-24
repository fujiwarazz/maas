#!/usr/bin/env python3
"""
示例：如何使用LangGraph的interrupt API进行人类输入

这个文件展示了如何使用interrupt API来获取人类输入，
并演示了如何使用Command(resume=...)来恢复执行。
"""

from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from typing import List

class State(TypedDict):
    """简化的状态定义"""
    messages: List[HumanMessage]
    llm_generated_summary: str
    human_feedback: str
    final_result: str

def generate_summary(state: State):
    """模拟LLM生成摘要"""
    print("=== 生成初始摘要 ===")
    summary = "这是一个关于人工智能项目的初始评估摘要..."
    return {
        "llm_generated_summary": summary
    }

def human_editing(state: State):
    """使用interrupt API获取人类输入"""
    print("=== 等待人类审核 ===")
    
    # 使用interrupt API中断执行，等待人类输入
    result = interrupt(
        # 发送给客户端的中断信息，可以是任何JSON可序列化的值
        {
            "task": "请审查LLM生成的摘要并进行必要的编辑",
            "llm_generated_summary": state["llm_generated_summary"],
            "instructions": "请提供您的反馈或编辑后的文本。如果满意请输入'approved'。"
        }
    )
    
    # 使用人类编辑的文本更新状态
    return {
        "human_feedback": result.get("edited_text", result.get("feedback", ""))
    }

def finalize_result(state: State):
    """最终化结果"""
    print("=== 最终化结果 ===")
    
    feedback = state.get("human_feedback", "")
    original = state.get("llm_generated_summary", "")
    
    if feedback.lower() == "approved":
        final = f"最终结果（已批准）：{original}"
    else:
        final = f"最终结果（已修改）：{feedback}"
    
    return {
        "final_result": final
    }

def create_test_graph():
    """创建测试图"""
    workflow = StateGraph(State)
    
    # 添加节点
    workflow.add_node("generate", generate_summary)
    workflow.add_node("human_editing", human_editing)
    workflow.add_node("finalize", finalize_result)
    
    # 添加边
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "human_editing")
    workflow.add_edge("human_editing", "finalize")
    workflow.add_edge("finalize", END)
    
    # 创建checkpointer - 这是interrupt API工作的关键！
    checkpointer = MemorySaver()
    
    # 编译图，必须包含checkpointer才能使用interrupt API
    return workflow.compile(checkpointer=checkpointer)

def main():
    """主函数演示如何使用interrupt API"""
    print("=== LangGraph Interrupt API 示例 ===\n")
    
    # 创建图
    graph = create_test_graph()
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content="请评估这个AI项目")],
        "llm_generated_summary": "",
        "human_feedback": "",
        "final_result": ""
    }
    
    # 配置线程
    thread_config = {"configurable": {"thread_id": "test_thread_001"}}
    
    print("步骤1: 启动图执行...")
    
    # 第一次调用会运行到interrupt点
    result = graph.invoke(initial_state, config=thread_config)
    
    # 检查是否有中断
    if "__interrupt__" in result and result["__interrupt__"]:
        print("🛑 图在人类审核点中断，等待输入...")
        interrupt_info = result["__interrupt__"][0]
        print(f"中断信息: {interrupt_info.value}")
        print("\n=== 模拟人类输入 ===")
        
        human_input = input()
        # 使用Command.resume恢复执行
        resume_result = graph.invoke(
            Command(resume={"edited_text": human_input}),
            config=thread_config
        )
        
        print("✅ 图执行恢复并完成")
        print(f"最终结果: {resume_result.get('final_result', '无结果')}")
        
    else:
        print("❌ 图执行完成但没有中断")
        print(f"最终结果: {result.get('final_result', '无结果')}")
        print(f"完整结果: {result}")

if __name__ == "__main__":
    main()
