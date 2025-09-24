#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整的Stage1图工作流测试
测试从用户输入到意图识别，再到条件路由的完整流程
包含两种路径：结构化分析 vs 直接输出
"""

import sys
import os
import pathlib
import time
from typing import Dict, Any, List

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# 导入需要的组件
from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.structure import create_structure_node
from proposalAgent.agents.stage1.schedule import create_schedule_agent
from proposalAgent.agents.stage1.output import create_output_node
from proposalAgent.graphs.conditional_logic import ConditionalLogic
from proposalAgent.agents.utils.agent_states import AgentState

class CompleteStage1WorkflowTest:
    """完整的Stage1工作流测试类"""
    
    def __init__(self, pdf_path: str):
        """
        初始化测试类
        
        Args:
            pdf_path: PDF文件路径
        """
        self.pdf_path = pdf_path
        self.llm = ChatOpenAI(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868")
        )
        
        # 初始化条件逻辑
        self.conditional_logic = ConditionalLogic()
        
        print("🚀 完整Stage1工作流测试初始化完成")
        print(f"📄 使用PDF文件: {pdf_path}")
        print("=" * 80)
    
    def create_stage1_graph(self):
        """创建Stage1的完整图工作流"""
        print("🔧 构建Stage1完整图工作流...")
        
        # 创建所有节点
        intention_node = create_intention_agent(self.llm)
        output_node = create_output_node(self.llm)
        structure_node = create_structure_node()
        planning_node = create_schedule_agent(self.llm)
        
        # 包装节点以提供调试信息
        def wrap_node(node_func, node_name):
            """包装节点以提供执行信息"""
            def wrapped_node(state):
                print(f"\n🔄 正在执行: {node_name}")
                start_time = time.time()
                try:
                    result = node_func(state)
                    end_time = time.time()
                    print(f"✅ {node_name} 完成 (耗时: {end_time - start_time:.2f}s)")
                    return result
                except Exception as e:
                    print(f"❌ {node_name} 失败: {e}")
                    raise
            return wrapped_node
        
        # 创建图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("intention_node", wrap_node(intention_node, "意图识别"))
        workflow.add_node("output_node", wrap_node(output_node, "直接输出"))
        workflow.add_node("structure_node", wrap_node(structure_node, "结构分析"))
        workflow.add_node("planning_node", wrap_node(planning_node, "权重规划"))
        
        # 添加边
        workflow.add_edge(START, "intention_node")
        
        # 条件路由：根据意图决定走哪条路径
        workflow.add_conditional_edges(
            "intention_node",
            self.conditional_logic.should_output,
            {
                "output_node": "output_node",
                "structure_node": "structure_node"
            }
        )
        
        # 直接输出路径
        workflow.add_edge("output_node", END)
        
        # 结构化分析路径
        workflow.add_edge("structure_node", "planning_node")
        workflow.add_edge("planning_node", END)
        
        # 编译图
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        
        print("✅ Stage1图工作流构建完成")
        return graph
    
    def create_initial_state(self, user_input: str, research_topics: List[str] = None, include_pdf: bool = False) -> Dict[str, Any]:
        """
        创建初始状态
        
        Args:
            user_input: 用户输入
            research_topics: 研究主题列表
            include_pdf: 是否包含PDF文件路径
            
        Returns:
            初始状态字典
        """
        if research_topics is None:
            research_topics = ["创新性", "可行性", "学术价值"]
            
        state = {
            "messages": [("user", user_input)],
            "research_topic": research_topics,
            "intention_decision": "",
            "should_output": False,
            # 基本字段
            "research_structure": "",
            "research_person_info": "",
            "research_basic_info": "",
            "research_project_team_info": "",
            "research_project_apply_info": "",
            "research_report_body_summary": "",
            "weight_distribution": {},
            # 其他必要字段
            "academic_analysis_report": None,
            "social_analysis_report": None,
            "future_influence_report": None,
            "interdisciplinary_results": [],
            "current_discipline": None,
            "debate_results": {},
            "final_analysis_summary": "",
            "completeness_check_result": None,
            "is_analysis_complete": None,
            "is_analysis_consistent": None,
            "completeness_recommendation": None,
            "reflection_decision": None,
            "skip_human_review": None,
            "human_feedback": None,
            "feedback_analysis_result": None,
            "feedback_routing_decision": None,
            "feedback_instructions": None,
            "final_report": None
        }
        
        # 如果包含PDF，添加文件路径
        if include_pdf:
            state["filepath"] = self.pdf_path
            state["file_path"] = self.pdf_path
            
        return state
    
    def test_direct_output_workflow(self):
        """测试直接输出工作流（通用模型能力）"""
        print("\n📝 测试直接输出工作流")
        print("=" * 60)
        
        graph = self.create_stage1_graph()
        
        test_cases = [
            {
                "name": "通用问答测试",
                "user_input": "请介绍一下机器学习的基本概念",
                "research_topics": ["人工智能", "机器学习"],
                "expected_path": "output_node"
            },
            {
                "name": "技术咨询测试",
                "user_input": "如何提高深度学习模型的性能？",
                "research_topics": ["深度学习", "模型优化"],
                "expected_path": "output_node"
            },
            {
                "name": "日常对话测试",
                "user_input": "你好，今天天气怎么样？",
                "research_topics": ["日常对话"],
                "expected_path": "output_node"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔄 测试用例 {i}: {case['name']}")
            print("-" * 40)
            print(f"📝 用户输入: {case['user_input']}")
            print(f"🔬 研究主题: {', '.join(case['research_topics'])}")
            
            # 创建初始状态
            state = self.create_initial_state(
                case['user_input'],
                case['research_topics'],
                include_pdf=False
            )
            
            # 执行工作流
            try:
                config = {"configurable": {"thread_id": f"direct_test_{i}"}}
                start_time = time.time()
                
                final_state = None
                execution_path = []
                
                for step in graph.stream(state, config):
                    for node_name, node_state in step.items():
                        execution_path.append(node_name)
                        final_state = node_state
                
                end_time = time.time()
                
                print(f"\n📊 执行结果:")
                print(f"  🛤️  执行路径: {' -> '.join(execution_path)}")
                print(f"  ⏱️  总耗时: {end_time - start_time:.2f}s")
                print(f"  🧠 意图识别: {final_state.get('intention_decision', 'N/A')}")
                
                # 检查是否走了正确的路径
                if case['expected_path'] in execution_path:
                    print(f"  ✅ 路径正确: 走了{case['expected_path']}")
                else:
                    print(f"  ❌ 路径错误: 期望{case['expected_path']}, 实际{execution_path}")
                
                # 检查输出
                messages = final_state.get('messages', [])
                print(f"  📨 消息数量: {len(messages)}")
                if messages and len(messages) > 1:  # 除了用户输入外还有回复
                    # 查找AI回复（非用户消息）
                    ai_messages = [msg for msg in messages if not (isinstance(msg, tuple) and msg[0] == 'user')]
                    if ai_messages:
                        last_ai_message = ai_messages[-1]
                        if isinstance(last_ai_message, str):
                            response_text = last_ai_message
                        else:
                            response_text = str(last_ai_message)
                        print(f"  💬 回复长度: {len(response_text)} 字符")
                        print(f"  📄 回复预览: {response_text[:100]}...")
                        print("  ✅ 生成了回复")
                    else:
                        print("  ❌ 未找到AI回复")
                else:
                    print("  ❌ 消息数量不足")
                
            except Exception as e:
                print(f"❌ 工作流执行失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("-" * 40)
    
    def test_structure_analysis_workflow(self):
        """测试结构化分析工作流（申请书评估）"""
        print("\n📄 测试结构化分析工作流")
        print("=" * 60)
        
        # 检查PDF文件
        if not pathlib.Path(self.pdf_path).exists():
            print(f"❌ PDF文件不存在: {self.pdf_path}")
            print("跳过结构化分析测试")
            return
        
        graph = self.create_stage1_graph()
        
        test_cases = [
            {
                "name": "申请书结构分析测试",
                "user_input": "请分析这篇科研申请书的结构和内容",
                "research_topics": ["结构分析", "内容评估"],
                "expected_path": "structure_node"
            },
            {
                "name": "申请书评估测试",
                "user_input": "评估这篇国家自然科学基金申请书的创新性和可行性",
                "research_topics": ["创新性", "可行性", "学术价值"],
                "expected_path": "structure_node"
            },
            {
                "name": "论文评估测试",
                "user_input": "分析这篇论文的研究方法和学术贡献",
                "research_topics": ["研究方法", "学术贡献"],
                "expected_path": "structure_node"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔄 测试用例 {i}: {case['name']}")
            print("-" * 40)
            print(f"📝 用户输入: {case['user_input']}")
            print(f"🔬 研究主题: {', '.join(case['research_topics'])}")
            print(f"📄 PDF文件: 已提供")
            
            # 创建初始状态
            state = self.create_initial_state(
                case['user_input'],
                case['research_topics'],
                include_pdf=True
            )
            
            # 执行工作流
            try:
                config = {"configurable": {"thread_id": f"structure_test_{i}"}}
                start_time = time.time()
                
                final_state = None
                execution_path = []
                
                for step in graph.stream(state, config):
                    for node_name, node_state in step.items():
                        execution_path.append(node_name)
                        final_state = node_state
                
                end_time = time.time()
                
                print(f"\n📊 执行结果:")
                print(f"  🛤️  执行路径: {' -> '.join(execution_path)}")
                print(f"  ⏱️  总耗时: {end_time - start_time:.2f}s")
                print(f"  🧠 意图识别: {final_state.get('intention_decision', 'N/A')}")
                
                # 检查是否走了正确的路径
                if case['expected_path'] in execution_path:
                    print(f"  ✅ 路径正确: 走了{case['expected_path']}")
                else:
                    print(f"  ❌ 路径错误: 期望{case['expected_path']}, 实际{execution_path}")
                
                # 检查结构分析结果
                if "structure_node" in execution_path:
                    structure_fields = [
                        ("research_structure", "完整结构分析"),
                        ("research_person_info", "人员信息"),
                        ("research_project_team_info", "团队信息"),
                        ("research_project_apply_info", "申请信息"),
                        ("research_report_body_summary", "报告正文摘要")
                    ]
                    
                    print(f"  📋 结构分析结果:")
                    for field, name in structure_fields:
                        content = final_state.get(field, "")
                        if content:
                            print(f"    ✅ {name}: {len(str(content))} 字符")
                        else:
                            print(f"    ❌ {name}: 无内容")
                
                # 检查权重分配结果
                if "planning_node" in execution_path:
                    weight_dist = final_state.get('weight_distribution', {})
                    if weight_dist:
                        print(f"  ⚖️  权重分配结果:")
                        total_weight = sum(weight_dist.values())
                        for agent_name, weight in weight_dist.items():
                            percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                            print(f"    📈 {agent_name}: {weight:.3f} ({percentage:.1f}%)")
                    else:
                        print(f"  ❌ 未生成权重分配")
                
            except Exception as e:
                print(f"❌ 工作流执行失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("-" * 40)
    
    def run_comprehensive_workflow_test(self):
        """运行完整的工作流测试"""
        print("🎯 开始完整Stage1工作流测试")
        print("测试两种流程：直接输出 vs 结构化分析")
        print("=" * 80)
        
        try:
            # 1. 测试直接输出工作流
            self.test_direct_output_workflow()
            
            # 2. 测试结构化分析工作流
            self.test_structure_analysis_workflow()
            
            print("\n" + "=" * 80)
            print("🎉 完整工作流测试完成！")
            print("总结：")
            print("  ✅ 直接输出流程：用于通用模型能力测试")
            print("  ✅ 结构化分析流程：用于申请书/论文评估")
            print("  ✅ 条件路由：根据意图识别结果自动选择路径")
            
        except KeyboardInterrupt:
            print("\n⚠️  测试被用户中断")
        except Exception as e:
            print(f"\n❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主测试函数"""
    # PDF文件路径
    pdf_path = "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/提交版本.pdf"
    
    try:
        # 初始化测试
        tester = CompleteStage1WorkflowTest(pdf_path)
        
        # 运行完整工作流测试
        tester.run_comprehensive_workflow_test()
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
