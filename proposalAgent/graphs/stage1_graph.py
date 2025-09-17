#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage1 图结构
集成意图识别、结构分析和输出节点
"""

from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, TypedDict, List
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import sys
import os
from langgraph.graph import MessagesState

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.structure import create_structure_node
from proposalAgent.agents.stage1.output import create_output_node


class Stage1State(MessagesState):
    """Stage1 图状态定义"""
    intention_decision: str
    file_path: str
    research_structure: str
    research_person_info: str
    research_project_team_info: str
    research_project_apply_info: str
    research_report_body_summary: str


def create_stage1_graph():
    """
    创建Stage1图结构
    """
    # 初始化LLM
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868")
    )
    
    # 创建各个节点
    intention_agent = create_intention_agent(llm)
    structure_node = create_structure_node()
    output_node = create_output_node()
    
    # 条件路由函数
    def should_process_structure(state: Stage1State) -> str:
        """
        根据意图决定是否进行结构分析
        """
        intention = state.get("intention_decision", "").lower()
        print(f"意图决策: {intention}")
        
        if "structure" in intention:
            return "structure"
        else:
            return "output"
    
    # 创建图
    workflow = StateGraph(Stage1State)
    
    # 添加节点
    workflow.add_node("intention", intention_agent)
    workflow.add_node("structure", structure_node)
    workflow.add_node("output", output_node)
    
    # 添加边
    workflow.add_edge(START, "intention")
    workflow.add_conditional_edges(
        "intention",
        should_process_structure,
        {
            "structure": "structure",
            "output": "output"
        }
    )
    workflow.add_edge("structure", END)
    workflow.add_edge("output", END)
    
    # 编译图
    graph = workflow.compile()
    return graph


def create_streaming_stage1_graph():
    """
    创建支持流式输出的Stage1图结构
    """
    # 初始化LLM
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868"),
        streaming=True
    )
    
    # 创建各个节点
    intention_agent = create_intention_agent(llm)
    structure_node = create_structure_node()
    output_node = create_output_node()
    
    # 流式输出包装器
    def streaming_wrapper(node_func, node_name):
        """包装节点以支持流式输出"""
        def wrapped_node(state):
            print(f"\n=== 开始执行 {node_name} 节点 ===")
            result = node_func(state)
            print(f"=== {node_name} 节点执行完成 ===\n")
            return result
        return wrapped_node
    
    # 条件路由函数
    def should_process_structure(state: Stage1State) -> str:
        """
        根据意图决定是否进行结构分析
        """
        intention = state.get("intention_decision", "").lower()
        print(f"🤔 意图分析结果: {intention}")
        
        if "structure" in intention:
            print("📄 将进行结构分析...")
            return "structure"
        else:
            print("💬 将进行通用对话...")
            return "output"
    
    # 创建图
    workflow = StateGraph(Stage1State)
    
    # 添加包装后的节点
    workflow.add_node("intention", streaming_wrapper(intention_agent, "意图识别"))
    workflow.add_node("structure", streaming_wrapper(structure_node, "结构分析"))
    workflow.add_node("output", streaming_wrapper(output_node, "输出生成"))
    
    # 添加边
    workflow.add_edge(START, "intention")
    workflow.add_conditional_edges(
        "intention",
        should_process_structure,
        {
            "structure": "structure",
            "output": "output"
        }
    )
    workflow.add_edge("structure", END)
    workflow.add_edge("output", END)
    
    # 编译图
    graph = workflow.compile()
    return graph


if __name__ == "__main__":
    print("=== Stage1 图结构测试 ===\n")
    
    # 创建图
    graph = create_streaming_stage1_graph()
    
    # 测试用例1: 通用对话
    print("测试用例1: 通用对话")
    print("-" * 40)
    state1 = {
        "messages": [("user", "你好，请介绍一下自己")],
        "intention_decision": "",
        "file_path": "",
        "research_structure": "",
        "research_person_info": "",
        "research_project_team_info": "",
        "research_project_apply_info": "",
        "research_report_body_summary": ""
    }
    
    try:
        result1 = graph.invoke(state1)
        print("✅ 测试用例1完成")
        print(f"最终状态: {result1.get('messages', [])[-1] if result1.get('messages') else '无输出'}")
    except Exception as e:
        print(f"❌ 测试用例1失败: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试用例2: 结构分析（需要PDF文件）
    print("测试用例2: 结构分析")
    print("-" * 40)
    state2 = {
        "messages": [("user", "分析这篇申请书的结构")],
        "intention_decision": "",
        "file_path": "/Users/peelsannaw/Desktop/提交版本.pdf",  # 假设存在此文件
        "research_structure": "",
        "research_person_info": "",
        "research_project_team_info": "",
        "research_project_apply_info": "",
        "research_report_body_summary": ""
    }
    
    try:
        # 检查文件是否存在
        import pathlib
        pdf_path = pathlib.Path(state2["file_path"])
        if pdf_path.exists():
            result2 = graph.invoke(state2)
            print("✅ 测试用例2完成")
            print(f"结构分析结果长度: {len(result2.get('research_structure', ''))}")
        else:
            print("⚠️  PDF文件不存在，跳过结构分析测试")
    except Exception as e:
        print(f"❌ 测试用例2失败: {e}")
    
    print("\n=== 测试完成 ===")
