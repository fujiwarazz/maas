#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用您自己的智能体的Stage3测试脚本
测试您现有的智能体组件和工作流
"""

import sys
import os
import json
from typing import Dict, Any

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# 导入您现有的智能体
from proposalAgent.agents.stage3.completeness_checker import create_completeness_checker_agent
from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.generator import create_generator_agent
from proposalAgent.agents.stage3.reflection_agent import create_reflection_agent
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.model_config import TONGYI_CONFIG

class MyStage3AgentsTest:
    """使用您自己的智能体的Stage3测试类"""
    
    def __init__(self):
        """初始化测试类"""
        self.llm = ChatOpenAI(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=SecretStr(TONGYI_CONFIG.get("api_key"))
        )
        
        # 创建您现有的智能体
        self.completeness_checker = create_completeness_checker_agent(self.llm)
        self.feedback_analyzer = create_feedback_analysis_agent(self.llm)
        self.generator = create_generator_agent(self.llm)
        self.reflection_agent = create_reflection_agent(self.llm)
        self.final_analyst = create_final_analyst_agent(self.llm)
        
        print("✅ 已初始化您的智能体:")
        print("   - 完备性检查智能体")
        print("   - 反馈分析智能体")
        print("   - 报告生成智能体")
        print("   - 反思智能体")
        print("   - 最终分析智能体")
    
    def create_test_data(self) -> Dict[str, Any]:
        """创建测试数据"""
        return {
            "messages": [HumanMessage(content="开始分析")],
            "research_topic": "基于大语言模型的智能教育系统研究",
            "research_basic_info": """
            申请人: 张三，清华大学计算机系副教授
            申请代码: F0212 数据科学与大数据计算
            研究周期: 3年
            申请金额: 80万元
            研究团队: 5人，包括2名博士生，2名硕士生，1名本科生
            """,
            
            
            "academic_analysis_result": """
            学术能力评估:
            - 申请人具有扎实的计算机科学背景，在自然语言处理领域有丰富经验
            - 已发表SCI论文15篇，其中一区论文8篇
            - 主持过2项国家自然科学基金项目
            - 在Transformer架构优化方面有重要贡献
            """,
            "social_analysis_result": """
            社会影响分析:
            - 教育领域数字化转型的重要技术支撑
            - 有助于提升教育公平性和个性化学习
            - 可能对传统教育模式产生深远影响
            - 需要关注数据隐私和算法公平性问题
            """,
            "future_influence_result": """
            未来影响预测:
            - 短期(1-2年): 在教育辅助工具方面有应用前景
            - 中期(3-5年): 可能推动个性化教育模式普及
            - 长期(5-10年): 有望重塑教育生态系统
            - 风险: 技术依赖、数字鸿沟、伦理问题
            """,
            "interdisciplinary_result": """
            - 与教育学: 需要深入理解教育理论和实践
            - 与心理学: 认知科学理论对模型设计很重要
            - 与伦理学: 需要关注AI伦理和教育公平
            - 与数据科学: 大数据处理和分析能力
            """,
            "debate_result": """
            辩论结果:
            可行性: 技术基础扎实，团队配置合理，预期目标可实现
            创新性: 在个性化教育和大模型结合方面有创新点
            争议点: 数据隐私保护、算法透明度、教育公平性
            建议: 加强伦理审查，建立多方协作机制
            """
        }
    
    def build_analysis_messages(self, data: Dict[str, Any]) -> str:
        """构建分析消息"""
        messages = []
        
        # 基本信息
        if data.get('research_topic'):
            messages.append(f"研究主题: {data['research_topic']}")
        
        if data.get('research_basic_info'):
            messages.append(f"基本信息: {data['research_basic_info']}")
        
        # 各阶段分析结果
        if data.get('academic_analysis_result'):
            messages.append(f"学术分析结果: {data['academic_analysis_result']}")
        
        if data.get('social_analysis_result'):
            messages.append(f"社会分析结果: {data['social_analysis_result']}")
        
        if data.get('future_influence_result'):
            messages.append(f"未来影响分析: {data['future_influence_result']}")
        
        if data.get('interdisciplinary_result'):
            messages.append(f"跨学科分析: {data['interdisciplinary_result']}")
        
        if data.get('debate_result'):
            messages.append(f"辩论结果: {data['debate_result']}")
        
        return "\n".join(messages) if messages else "暂无分析数据"
    
    def test_completeness_checker(self):
        """测试完备性检查智能体"""
        print("=" * 60)
        print("🧪 测试您的完备性检查智能体")
        print("=" * 60)
        
        test_data = self.create_test_data()
        messages = self.build_analysis_messages(test_data)
        
        try:
            # 创建模拟状态
            state = {
                "messages": [HumanMessage(content=messages)],
                "research_topic": test_data.get("research_topic", ""),
                "research_basic_info": test_data.get("research_basic_info", ""),
                "academic_analysis_result": test_data.get("academic_analysis_result", ""),
                "social_analysis_result": test_data.get("social_analysis_result", ""),
                "future_influence_result": test_data.get("future_influence_result", ""),
                "interdisciplinary_result": test_data.get("interdisciplinary_result", ""),
                "debate_result": test_data.get("debate_result", "")
            }
            
            # 执行完备性检查
            result_state = self.completeness_checker(state)
            
            # 获取结果
            result = result_state.get("completeness_check_result")
            
            print("✅ 完备性检查完成")
            print(f"   完整性: {result.get('is_complete') if result else 'N/A'}")
            print(f"   一致性: {result.get('is_consistent') if result else 'N/A'}")
            print(f"   质量评分: {result.get('overall_quality', 'N/A') if result else 'N/A'}")
            print(f"   建议: {result.get('recommendation', 'N/A') if result else 'N/A'}")
            
            if result and result.get('missing_parts'):
                print(f"   缺失部分: {result.get('missing_parts')}")
            
            if result and result.get('inconsistencies'):
                print(f"   不一致问题: {result.get('inconsistencies')}")
            
            print("\n📋 完整结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2) if result else "无结果")
            
            return result
            
        except Exception as e:
            print(f"❌ 完备性检查失败: {e}")
            return None
    
    def test_feedback_analysis(self):
        """测试反馈分析智能体"""
        print("=" * 60)
        print("🧪 测试您的反馈分析智能体")
        print("=" * 60)
        
        test_data = self.create_test_data()
        
        try:
            # 创建模拟状态
            state = {
                "messages": [HumanMessage(content=self.build_analysis_messages(test_data))],
                "human_feedback": "学术分析部分需要更深入，特别是申请人的研究背景和学术能力评估",
                "research_topic": test_data.get("research_topic", ""),
                "research_basic_info": test_data.get("research_basic_info", ""),
                "academic_analysis_result": test_data.get("academic_analysis_result", ""),
                "social_analysis_result": test_data.get("social_analysis_result", ""),
                "future_influence_result": test_data.get("future_influence_result", ""),
                "interdisciplinary_result": test_data.get("interdisciplinary_result", ""),
                "debate_result": test_data.get("debate_result", "")
            }
            
            # 执行反馈分析
            result_state = self.feedback_analyzer(state)
            
            # 获取结果
            result = result_state.get("feedback_analysis_result")
            
            print("✅ 反馈分析完成")
            print(f"   路由决策: {result.get('routing_decision', 'N/A') if result else 'N/A'}")
            print(f"   问题类型: {result.get('problem_type', 'N/A') if result else 'N/A'}")
            print(f"   优先级: {result.get('priority', 'N/A') if result else 'N/A'}")
            print(f"   执行指令: {result.get('instructions', 'N/A') if result else 'N/A'}")
            
            print("\n📋 完整结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2) if result else "无结果")
            
            return result
            
        except Exception as e:
            print(f"❌ 反馈分析失败: {e}")
            return None
    
    def test_generator(self):
        """测试报告生成智能体"""
        print("=" * 60)
        print("🧪 测试您的报告生成智能体")
        print("=" * 60)
        
        test_data = self.create_test_data()
        messages = self.build_analysis_messages(test_data)
        
        try:
            # 创建模拟状态
            state = {
                "messages": [HumanMessage(content=messages)],
                "research_topic": test_data.get("research_topic", ""),
                "research_basic_info": test_data.get("research_basic_info", ""),
                "academic_analysis_result": test_data.get("academic_analysis_result", ""),
                "social_analysis_result": test_data.get("social_analysis_result", ""),
                "future_influence_result": test_data.get("future_influence_result", ""),
                "interdisciplinary_result": test_data.get("interdisciplinary_result", ""),
                "debate_result": test_data.get("debate_result", "")
            }
            
            # 执行报告生成
            result_state = self.generator(state)
            
            # 获取结果
            report = result_state.get("final_report")
            
            print("✅ 报告生成完成")
            print(f"   报告长度: {len(report)} 字符" if report else "   报告长度: 0 字符")
            
            if report:
                print("\n📋 生成的报告预览:")
                print("-" * 40)
                print(report[:500] + "..." if len(report) > 500 else report)
                print("-" * 40)
            
            return report
            
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            return None
    
    def test_reflection(self):
        """测试反思智能体"""
        print("=" * 60)
        print("🧪 测试您的反思智能体")
        print("=" * 60)
        
        test_data = self.create_test_data()
        messages = self.build_analysis_messages(test_data)
        
        try:
            # 执行反思
            response = self.reflection_agent.invoke({"messages": messages})
            
            # 解析结果
            result = json.loads(response.content)
            
            print("✅ 反思完成")
            print(f"   置信度: {result.get('confidence_score', 'N/A')}")
            print(f"   建议: {result.get('recommendation', 'N/A')}")
            
            print("\n📋 完整结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            return result
            
        except Exception as e:
            print(f"❌ 反思失败: {e}")
            return None
    
    def test_final_analyst(self):
        """测试最终分析智能体"""
        print("=" * 60)
        print("🧪 测试您的最终分析智能体")
        print("=" * 60)
        
        test_data = self.create_test_data()
        messages = self.build_analysis_messages(test_data)
        
        try:
            # 创建模拟状态
            state = {
                "messages": [HumanMessage(content=messages)],
                "research_topic": test_data.get("research_topic", ""),
                "research_basic_info": test_data.get("research_basic_info", ""),
                "academic_analysis_result": test_data.get("academic_analysis_result", ""),
                "social_analysis_result": test_data.get("social_analysis_result", ""),
                "future_influence_result": test_data.get("future_influence_result", ""),
                "interdisciplinary_result": test_data.get("interdisciplinary_result", ""),
                "debate_result": test_data.get("debate_result", "")
            }
            
            # 执行最终分析
            result_state = self.final_analyst(state)
            
            # 获取结果
            result = result_state.get("final_analysis_result")
            
            print("✅ 最终分析完成")
            print(f"   分析结果长度: {len(str(result))} 字符" if result else "   分析结果长度: 0 字符")
            
            if result:
                print("\n📋 分析结果预览:")
                print("-" * 40)
                result_str = str(result)
                print(result_str[:500] + "..." if len(result_str) > 500 else result_str)
                print("-" * 40)
            
            return result
            
        except Exception as e:
            print(f"❌ 最终分析失败: {e}")
            return None
    
    def test_complete_workflow(self):
        """测试完整工作流"""
        print("=" * 60)
        print("🧪 测试您的完整Stage3工作流")
        print("=" * 60)
        
        test_data = self.create_test_data()
        
        try:
            # 1. 完备性检查
            print("\n1️⃣ 执行完备性检查...")
            completeness_result = self.test_completeness_checker()
            
            # 2. 反馈分析（如果需要）
            if completeness_result and completeness_result.get('recommendation') == 'need_human_review':
                print("\n2️⃣ 执行反馈分析...")
                feedback_result = self.test_feedback_analysis()
            else:
                print("\n2️⃣ 跳过反馈分析（完备性检查通过）")
                feedback_result = None
            
            # 3. 报告生成
            print("\n3️⃣ 执行报告生成...")
            report = self.test_generator()
            
            # 4. 反思
            print("\n4️⃣ 执行反思...")
            reflection_result = self.test_reflection()
            
            # 5. 最终分析
            print("\n5️⃣ 执行最终分析...")
            final_result = self.test_final_analyst()
            
            print("\n✅ 完整工作流测试完成!")
            
            return {
                "completeness_result": completeness_result,
                "feedback_result": feedback_result,
                "report": report,
                "reflection_result": reflection_result,
                "final_result": final_result
            }
            
        except Exception as e:
            print(f"❌ 完整工作流测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始测试您的Stage3智能体")
        
        try:
            # 测试各个组件
            print("\n1. 测试完备性检查智能体")
            completeness_result = self.test_completeness_checker()
            
            print("\n2. 测试反馈分析智能体")
            feedback_result = self.test_feedback_analysis()
            
            print("\n3. 测试报告生成智能体")
            report = self.test_generator()
            
            print("\n4. 测试反思智能体")
            reflection_result = self.test_reflection()
            
            print("\n5. 测试最终分析智能体")
            final_result = self.test_final_analyst()
            
            print("\n6. 测试完整工作流")
            workflow_result = self.test_complete_workflow()
            
            print("\n✅ 所有测试完成!")
            
            # 总结结果
            print("\n📊 测试结果总结:")
            print(f"   完备性检查: {'✅ 成功' if completeness_result else '❌ 失败'}")
            print(f"   反馈分析: {'✅ 成功' if feedback_result else '❌ 失败'}")
            print(f"   报告生成: {'✅ 成功' if report else '❌ 失败'}")
            print(f"   反思分析: {'✅ 成功' if reflection_result else '❌ 失败'}")
            print(f"   最终分析: {'✅ 成功' if final_result else '❌ 失败'}")
            print(f"   完整工作流: {'✅ 成功' if workflow_result else '❌ 失败'}")
            
        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    test = MyStage3AgentsTest()
    test.run_all_tests()

if __name__ == "__main__":
    main()
