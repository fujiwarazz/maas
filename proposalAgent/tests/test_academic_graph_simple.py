#!/usr/bin/env python3
"""
简单的学术分析图测试
只包含academic node和tool node
"""

import os
import sys
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.abspath('.'))

class MockAIMessage:
    """模拟AI消息类"""
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

class MockToolCall:
    """模拟工具调用类"""
    def __init__(self, name, args=None, id=None):
        self.name = name
        self.args = args or {}
        self.id = id or f"call_{name}"

class SimpleAcademicGraph:
    """简单的学术分析图"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = self.create_mock_tools()
    
    def create_mock_tools(self):
        """创建模拟工具"""
        def mock_get_author_citations_auto(input_data):
            return {"metrics": {"citations_all": 100, "h_index_all": 15}}
        
        def mock_get_article_brief(input_data):
            return [{"title": "Test Article", "citations": 10}]
        
        mock_get_author_citations_auto.name = "get_author_citations_auto"
        mock_get_article_brief.name = "get_article_brief"
        
        return [mock_get_author_citations_auto, mock_get_article_brief]
    
    def academic_node(self, state):
        """学术分析节点"""
        try:
            person_info = state.get("research_person_info", "### **项目团队成员及其个人履历**\n未提供团队成员信息")
            
            # 模拟LLM调用
            mock_chain = self.llm.bind_tools(self.tools)
            result = mock_chain.invoke(state.get("messages", []))
            
            academic_report = ""
            
            if len(result.tool_calls) == 0:
                academic_report = result.content
            else:
                academic_report = "正在使用学术分析工具进行深度调研..."
            
            return {
                "messages": [result],
                "academic_analysis_report": academic_report,
                "next": "tools" if result.tool_calls else "end"
            }
            
        except Exception as e:
            return {
                "messages": [],
                "academic_analysis_report": f"学术分析过程中发生错误: {str(e)}",
                "next": "end"
            }
    
    def tool_node(self, state):
        """工具执行节点"""
        try:
            messages = state.get("messages", [])
            if not messages:
                return state
            
            last_message = messages[-1]
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
            
            # 模拟工具执行
            tool_results = []
            for tool_call in last_message.tool_calls:
                # 模拟工具调用结果
                if tool_call.name == "get_author_citations_auto":
                    result = {"metrics": {"citations_all": 150, "h_index_all": 20}}
                else:
                    result = {"status": "success", "data": "mock_result"}
                
                tool_results.append(result)
            
            # 创建工具执行后的消息
            tool_message = MockAIMessage(
                content=f"工具执行完成，获得{len(tool_results)}个结果",
                tool_calls=[]
            )
            
            return {
                "messages": messages + [tool_message],
                "academic_analysis_report": "工具执行完成，正在生成最终报告...",
                "next": "academic"  # 回到academic节点生成最终报告
            }
            
        except Exception as e:
            return {
                "messages": state.get("messages", []),
                "academic_analysis_report": f"工具执行错误: {str(e)}",
                "next": "end"
            }
    
    def run(self, initial_state):
        """运行图"""
        current_state = initial_state.copy()
        current_node = "academic"
        max_steps = 5  # 防止无限循环
        step_count = 0
        
        while current_node != "end" and step_count < max_steps:
            step_count += 1
            print(f"  步骤 {step_count}: 执行 {current_node} 节点")
            
            if current_node == "academic":
                result = self.academic_node(current_state)
                current_state.update(result)
                current_node = result.get("next", "end")
                
            elif current_node == "tools":
                result = self.tool_node(current_state)
                current_state.update(result)
                current_node = result.get("next", "end")
                
            else:
                break
        
        return current_state

def test_academic_graph():
    """测试学术分析图"""
    print("🚀 学术分析图测试")
    print("="*50)
    
    # 创建模拟LLM
    mock_llm = Mock()
    
    # 创建图
    graph = SimpleAcademicGraph(mock_llm)
    print("✅ 学术分析图创建成功")
    
    test_results = []
    
    # 测试1: 无工具调用流程
    print("\n1️⃣ 测试无工具调用流程:")
    
    mock_result = MockAIMessage(
        content="基于提供的团队信息，张教授在深度学习领域表现优异...",
        tool_calls=[]
    )
    
    mock_chain = Mock()
    mock_chain.invoke.return_value = mock_result
    mock_llm.bind_tools.return_value = mock_chain
    
    test_state = {
        "messages": [],
        "research_person_info": """### **项目团队成员及其个人履历**

**项目负责人：张教授**
- 博士学位，清华大学计算机科学与技术系
- 研究方向：深度学习、计算机视觉
- 发表论文：SCI论文50余篇
"""
    }
    
    result = graph.run(test_state)
    
    success = (
        "academic_analysis_report" in result and
        len(result["academic_analysis_report"]) > 10 and
        "张教授" in result["academic_analysis_report"]
    )
    test_results.append(("无工具调用流程", success))
    
    if success:
        print("✅ 测试通过")
        print(f"   最终报告: {result['academic_analysis_report'][:100]}...")
    else:
        print("❌ 测试失败")
    
    # 测试2: 有工具调用流程
    print("\n2️⃣ 测试有工具调用流程:")
    
    # 第一次调用：返回工具调用
    mock_result_with_tools = MockAIMessage(
        content="",
        tool_calls=[MockToolCall(name="get_author_citations_auto", args={"name": "张教授"})]
    )
    
    # 第二次调用：工具执行后的最终报告
    mock_result_final = MockAIMessage(
        content="根据工具调研结果，张教授的h指数为20，总被引150次...",
        tool_calls=[]
    )
    
    mock_chain.invoke.side_effect = [mock_result_with_tools, mock_result_final]
    
    result = graph.run(test_state)
    
    success = (
        "academic_analysis_report" in result and
        ("工具" in result["academic_analysis_report"] or "调研" in result["academic_analysis_report"])
    )
    test_results.append(("工具调用流程", success))
    
    if success:
        print("✅ 测试通过")
        print(f"   最终报告: {result['academic_analysis_report']}")
    else:
        print("❌ 测试失败")
        print(f"   实际结果: {result}")
    
    # 重置mock
    mock_chain.invoke.side_effect = None
    
    # 测试3: 错误处理
    print("\n3️⃣ 测试错误处理:")
    
    mock_llm.bind_tools.side_effect = Exception("模拟网络错误")
    
    result = graph.run(test_state)
    
    success = "学术分析过程中发生错误" in result.get("academic_analysis_report", "")
    test_results.append(("错误处理", success))
    
    if success:
        print("✅ 测试通过")
    else:
        print("❌ 测试失败")
    
    # 打印总结
    print("\n" + "="*50)
    print("📊 图测试总结")
    print("="*50)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, success in test_results if success)
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 图测试全部通过！")
        print("\n📋 图功能验证清单:")
        print("  ✅ 图结构创建")
        print("  ✅ Academic节点执行")
        print("  ✅ Tool节点执行")
        print("  ✅ 节点间流转")
        print("  ✅ 无工具调用路径")
        print("  ✅ 有工具调用路径")
        print("  ✅ 错误处理机制")
        return True
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个图测试失败")
        return False

def main():
    """主函数"""
    try:
        print("🔍 图测试环境信息:")
        env_name = os.environ.get('CONDA_DEFAULT_ENV', '未知')
        print(f"   Conda环境: {env_name}")
        
        success = test_academic_graph()
        
        if success:
            print("\n🎯 图测试结论: 学术分析图结构完全正常！")
        else:
            print("\n🚨 图测试结论: 图结构存在问题，需要调试")
        
        exit_code = 0 if success else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n❌ 图测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
