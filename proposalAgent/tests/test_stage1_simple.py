#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage1 简化测试脚本 - 不依赖结构分析模块
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, TypedDict, List
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.output import create_output_node


class SimpleStage1State(TypedDict):
    """简化的Stage1状态"""
    messages: List[Any]
    intention_decision: str


def create_simple_stage1_graph():
    """
    创建简化的Stage1图结构（不包含结构分析）
    """
    # 初始化LLM
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868")
    )
    
    # 创建节点
    intention_agent = create_intention_agent(llm)
    output_node = create_output_node()
    
    # 条件路由函数
    def route_to_output(state: SimpleStage1State) -> str:
        """所有请求都路由到输出节点"""
        intention = state.get("intention_decision", "").lower()
        print(f"🤔 意图分析结果: {intention}")
        
        if "structure" in intention:
            print("📄 检测到结构分析意图，但跳过结构分析，直接输出...")
        else:
            print("💬 路由到输出节点...")
        return "output"
    
    # 流式输出包装器
    def streaming_wrapper(node_func, node_name):
        """包装节点以支持流式输出"""
        def wrapped_node(state):
            print(f"\n=== 开始执行 {node_name} 节点 ===")
            result = node_func(state)
            print(f"=== {node_name} 节点执行完成 ===\n")
            return result
        return wrapped_node
    
    # 创建图
    workflow = StateGraph(SimpleStage1State)
    
    # 添加包装后的节点
    workflow.add_node("intention", streaming_wrapper(intention_agent, "意图识别"))
    workflow.add_node("output", streaming_wrapper(output_node, "输出生成"))
    
    # 添加边
    workflow.add_edge(START, "intention")
    workflow.add_conditional_edges(
        "intention",
        route_to_output,
        {
            "output": "output"
        }
    )
    workflow.add_edge("output", END)
    
    # 编译图
    graph = workflow.compile()
    return graph


def test_simple_graph():
    """测试简化的图结构"""
    print("🚀 开始 Stage1 简化图结构测试")
    print("=" * 60)
    
    graph = create_simple_stage1_graph()
    
    test_cases = [
        {
            "name": "通用问答测试",
            "input": "你好，请介绍一下自己",
        },
        {
            "name": "意图识别测试",
            "input": "请分析这篇申请书的结构",
        },
        {
            "name": "技术问题测试",
            "input": "什么是机器学习？",
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {case['name']}")
        print(f"输入: {case['input']}")
        print("-" * 50)
        
        state = {
            "messages": [("user", case['input'])],
            "intention_decision": "",
        }
        
        try:
            result = graph.invoke(state)
            
            # 检查结果
            print("📊 测试结果:")
            print(f"- 意图识别: {result.get('intention_decision', 'N/A')}")
            
            messages = result.get("messages", [])
            if messages:
                response = messages[-1]
                if isinstance(response, str):
                    print(f"- 回复长度: {len(response)} 字符")
                    print(f"- 回复预览: {response[:100]}...")
                else:
                    print(f"- 回复类型: {type(response)}")
                
                print("✅ 测试成功")
            else:
                print("❌ 没有生成回复")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 简化测试完成！")


def interactive_test():
    """交互式测试"""
    print("\n🎮 进入交互式测试模式")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 40)
    
    graph = create_simple_stage1_graph()
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            if not user_input:
                print("⚠️  请输入有效内容")
                continue
            
            state = {
                "messages": [("user", user_input)],
                "intention_decision": "",
            }
            
            print("\n🤖 AI处理中...")
            result = graph.invoke(state)
            
            messages = result.get("messages", [])
            if messages:
                response = messages[-1]
                print(f"\n🤖 AI: {response}")
            else:
                print("\n❌ AI没有回复")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    try:
        # 运行基础测试
        test_simple_graph()
        
        # 询问是否进入交互模式
        choice = input("\n是否进入交互式测试模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_test()
            
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
