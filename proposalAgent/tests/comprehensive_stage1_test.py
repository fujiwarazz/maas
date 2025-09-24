#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage1 完整测试脚本 - 测试所有stage1的agent
包含意图识别、结构分析、调度分配和输出生成的完整测试
使用真实的PDF申请书进行测试
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

# 导入stage1的所有agent
from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.structure import create_structure_node
from proposalAgent.agents.stage1.schedule import create_schedule_agent
from proposalAgent.agents.stage1.output import create_output_node

class Stage1ComprehensiveTest:
    """Stage1综合测试类"""
    
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
        
        # 初始化所有agent
        self.intention_agent = create_intention_agent(self.llm)
        self.structure_agent = create_structure_node()
        self.schedule_agent = create_schedule_agent(self.llm)
        self.output_agent = create_output_node(self.llm)
        
        print("🚀 Stage1综合测试初始化完成")
        print(f"📄 使用PDF文件: {pdf_path}")
        print("=" * 80)
    
    def create_initial_state(self, user_input: str, research_topics: List[str] = None) -> Dict[str, Any]:
        """
        创建初始状态
        
        Args:
            user_input: 用户输入
            research_topics: 研究主题列表
            
        Returns:
            初始状态字典
        """
        if research_topics is None:
            research_topics = ["人工智能", "机器学习", "深度学习"]
            
        state = {
            "messages": [("user", user_input)],
            "filepath": self.pdf_path,
            "file_path": self.pdf_path,
            "research_topic": research_topics,
            "intention_decision": "",
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
        return state
    
    def test_intention_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试意图识别agent
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print("\n🧠 测试意图识别Agent")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = self.intention_agent(state)
            end_time = time.time()
            
            # 更新状态
            state.update(result)
            
            print(f"✅ 意图识别完成 (耗时: {end_time - start_time:.2f}s)")
            print(f"📝 识别结果: {result.get('intention_decision', 'N/A')}")
            
            # 分析意图类型
            intention = result.get('intention_decision', '').lower()
            if 'structure' in intention:
                print("🔍 检测到申请书/论文评估意图")
            elif 'output' in intention:
                print("💬 检测到通用对话意图")
            else:
                print("❓ 未明确识别意图类型")
                
            return state
            
        except Exception as e:
            print(f"❌ 意图识别失败: {e}")
            import traceback
            traceback.print_exc()
            return state
    
    def test_structure_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试结构分析agent
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print("\n📄 测试结构分析Agent")
        print("-" * 50)
        
        # 检查PDF文件是否存在
        if not pathlib.Path(self.pdf_path).exists():
            print(f"❌ PDF文件不存在: {self.pdf_path}")
            return state
            
        try:
            start_time = time.time()
            result = self.structure_agent(state)
            end_time = time.time()
            
            # 更新状态
            state.update(result)
            
            print(f"✅ 结构分析完成 (耗时: {end_time - start_time:.2f}s)")
            
            # 检查分析结果
            structure_fields = [
                ("research_structure", "完整结构分析"),
                ("research_basic_info", "基本信息"),
                ("research_person_info", "人员信息"),
                ("research_project_team_info", "团队信息"),
                ("research_project_apply_info", "申请信息"),
                ("research_report_body_summary", "报告正文摘要")
            ]
            
            print("\n📊 结构分析结果:")
            for field, name in structure_fields:
                content = result.get(field, "")
                if content:
                    char_count = len(str(content))
                    print(f"  ✅ {name}: {char_count} 字符")
                    # 显示内容预览
                    preview = str(content)[:100].replace('\n', ' ')
                    print(f"     预览: {preview}...")
                else:
                    print(f"  ❌ {name}: 无内容")
            
            return state
            
        except Exception as e:
            print(f"❌ 结构分析失败: {e}")
            import traceback
            traceback.print_exc()
            return state
    
    def test_schedule_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试调度分配agent
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print("\n⚖️ 测试调度分配Agent")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = self.schedule_agent(state)
            end_time = time.time()
            
            # 更新状态
            state.update(result)
            
            print(f"✅ 调度分配完成 (耗时: {end_time - start_time:.2f}s)")
            
            # 分析权重分配
            weight_dist = result.get('weight_distribution', {})
            if weight_dist:
                print("\n📊 权重分配结果:")
                total_weight = sum(weight_dist.values())
                for agent_name, weight in weight_dist.items():
                    percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                    print(f"  📈 {agent_name}: {weight:.3f} ({percentage:.1f}%)")
                print(f"  📊 总权重: {total_weight:.3f}")
            else:
                print("❌ 未生成权重分配")
            
            return state
            
        except Exception as e:
            print(f"❌ 调度分配失败: {e}")
            import traceback
            traceback.print_exc()
            return state
    
    def test_output_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试输出生成agent
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print("\n💬 测试输出生成Agent")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = self.output_agent(state)
            end_time = time.time()
            
            # 更新状态
            if 'messages' in result:
                # 将新消息添加到原有消息列表
                if 'messages' not in state:
                    state['messages'] = []
                state['messages'].extend(result['messages'])
            
            print(f"✅ 输出生成完成 (耗时: {end_time - start_time:.2f}s)")
            
            # 分析输出结果
            messages = state.get('messages', [])
            if messages:
                # 获取最后一条AI回复
                ai_responses = [msg for msg in messages if not isinstance(msg, tuple) or msg[0] != 'user']
                if ai_responses:
                    last_response = ai_responses[-1]
                    response_text = str(last_response)
                    print(f"📝 回复长度: {len(response_text)} 字符")
                    print(f"📄 回复预览: {response_text[:200]}...")
                else:
                    print("❌ 未找到AI回复")
            else:
                print("❌ 未生成任何回复")
            
            return state
            
        except Exception as e:
            print(f"❌ 输出生成失败: {e}")
            import traceback
            traceback.print_exc()
            return state
    
    def run_comprehensive_test(self, test_cases: List[Dict[str, Any]]):
        """
        运行完整的综合测试
        
        Args:
            test_cases: 测试用例列表
        """
        print("🎯 开始Stage1完整综合测试")
        print("=" * 80)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔄 测试用例 {i}: {case['name']}")
            print("=" * 60)
            
            # 创建初始状态
            state = self.create_initial_state(
                case['user_input'],
                case.get('research_topics', ["人工智能", "机器学习"])
            )
            
            print(f"📝 用户输入: {case['user_input']}")
            print(f"🔬 研究主题: {', '.join(state['research_topic'])}")
            
            # 按顺序执行所有agent测试
            try:
                # 1. 意图识别
                state = self.test_intention_agent(state)
                
                # 2. 根据意图决定是否执行结构分析
                intention = state.get('intention_decision', '').lower()
                if 'structure' in intention:
                    state = self.test_structure_agent(state)
                else:
                    print("\n📄 跳过结构分析（意图为通用对话）")
                
                # 3. 调度分配（如果有研究主题）
                if state.get('research_topic'):
                    state = self.test_schedule_agent(state)
                else:
                    print("\n⚖️ 跳过调度分配（无研究主题）")
                
                # 4. 输出生成
                state = self.test_output_agent(state)
                
                # 显示测试总结
                print(f"\n📋 测试用例 {i} 总结:")
                print(f"  🧠 意图识别: {state.get('intention_decision', 'N/A')}")
                print(f"  📄 结构分析: {'完成' if state.get('research_structure') else '跳过/失败'}")
                print(f"  ⚖️ 调度分配: {'完成' if state.get('weight_distribution') else '跳过/失败'}")
                print(f"  💬 输出生成: {'完成' if state.get('messages') else '失败'}")
                print("  ✅ 整体状态: 成功")
                
            except Exception as e:
                print(f"\n❌ 测试用例 {i} 失败: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "=" * 60)
    
    def run_single_agent_tests(self):
        """运行单个agent的独立测试"""
        print("\n🔧 开始单个Agent独立测试")
        print("=" * 80)
        
        # 意图识别测试用例
        intention_cases = [
            "请分析这篇科研申请书的结构和内容",
            "评估这个项目的可行性和创新性",
            "你好，请介绍一下人工智能",
            "什么是深度学习？",
            "今天天气怎么样？"
        ]
        
        print("\n🧠 意图识别Agent独立测试")
        print("-" * 50)
        for i, user_input in enumerate(intention_cases, 1):
            print(f"\n测试 {i}: {user_input}")
            state = self.create_initial_state(user_input)
            state = self.test_intention_agent(state)
        
        # 结构分析独立测试
        print("\n📄 结构分析Agent独立测试")
        print("-" * 50)
        structure_state = self.create_initial_state("分析申请书结构")
        structure_state = self.test_structure_agent(structure_state)
        
        # 调度分配独立测试
        print("\n⚖️ 调度分配Agent独立测试")
        print("-" * 50)
        schedule_state = self.create_initial_state(
            "分配智能体权重", 
            ["交叉性", "创新性", "可行性", "未来影响力"]
        )
        schedule_state = self.test_schedule_agent(schedule_state)
        
        # 输出生成独立测试
        print("\n💬 输出生成Agent独立测试")
        print("-" * 50)
        output_cases = [
            "请解释什么是机器学习",
            "介绍一下深度学习的基本概念",
            "如何写好一份科研申请书？"
        ]
        for i, user_input in enumerate(output_cases, 1):
            print(f"\n输出测试 {i}: {user_input}")
            state = self.create_initial_state(user_input)
            state = self.test_output_agent(state)


def main():
    """主测试函数"""
    # PDF文件路径
    pdf_path = "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/提交版本.pdf"
    
    # 检查PDF文件是否存在
    if not pathlib.Path(pdf_path).exists():
        print(f"❌ PDF文件不存在: {pdf_path}")
        print("请确保PDF文件路径正确")
        return
    
    try:
        # 初始化测试
        tester = Stage1ComprehensiveTest(pdf_path)
        
        # 定义完整测试用例
        comprehensive_test_cases = [
            {
                "name": "申请书评估测试",
                "user_input": "请全面分析这篇国家自然科学基金申请书的结构、创新性和可行性",
                "research_topics": ["创新性", "可行性", "交叉性", "未来影响力"]
            },
            {
                "name": "项目结构分析测试",
                "user_input": "分析这个科研项目的研究内容和团队构成",
                "research_topics": ["学术分析", "团队分析"]
            },
            {
                "name": "申请书结构化分析测试",
                "user_input": "请对这篇申请书进行结构分析和内容评估",
                "research_topics": ["结构分析", "内容评估"]
            },
            {
                "name": "论文评估测试",
                "user_input": "评估这篇论文的学术价值和创新性",
                "research_topics": ["学术价值", "创新性评估"]
            },
            {
                "name": "通用问答测试",
                "user_input": "请介绍一下机器学习的基本概念和应用领域",
                "research_topics": ["人工智能", "机器学习"]
            },
            {
                "name": "技术咨询测试",
                "user_input": "如何提高深度学习模型的性能？",
                "research_topics": ["深度学习", "模型优化"]
            }
        ]
        
        # 运行综合测试
        tester.run_comprehensive_test(comprehensive_test_cases)
        
        # 询问是否运行单个agent测试
        print("\n" + "=" * 80)
        choice = input("是否运行单个Agent独立测试？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            tester.run_single_agent_tests()
        
        print("\n🎉 所有测试完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
