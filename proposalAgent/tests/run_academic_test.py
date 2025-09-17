#!/usr/bin/env python3
"""
学术分析测试运行脚本
包含完整的测试用例和示例
"""

import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from proposalAgent.agents.stage2.academic import create_academic_agent
from proposalAgent.graphs.academic_test_graph import create_academic_test_graph
from proposalAgent.model_config import TONGYI_CONFIG
from langchain_openai import ChatOpenAI


class AcademicTestRunner:
    """学术分析测试运行器"""
    
    def __init__(self):
        self.mock_llm = Mock(spec=ChatOpenAI)
        self.test_results = []
    
    def run_basic_agent_test(self):
        """运行基础agent测试"""
        print("=== 基础Agent测试 ===")
        
        # 创建academic agent
        academic_agent = create_academic_agent(self.mock_llm, {})
        
        # 测试状态
        test_state = {
            "messages": [],
            "research_topic": ["机器学习", "数据挖掘"],
            "research_person_info": "王教授，博士，专注于机器学习算法研究，发表论文50余篇",
        }
        
        # 测试1: 无工具调用
        print("\n1. 测试无工具调用场景:")
        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.content = self._generate_mock_academic_report()
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        
        with patch.object(self.mock_llm, 'bind_tools', return_value=mock_chain):
            result = academic_agent(test_state)
            
            success = self._validate_agent_result(result, expect_tool_calls=False)
            self.test_results.append(("基础Agent-无工具调用", success))
            
            if success:
                print("✅ 测试通过")
                print(f"报告长度: {len(result['academic_analysis_report'])} 字符")
            else:
                print("❌ 测试失败")
        
        # 测试2: 有工具调用
        print("\n2. 测试有工具调用场景:")
        mock_result_with_tools = Mock()
        mock_result_with_tools.tool_calls = [
            Mock(name="get_author_citations_auto", args={"name": "王教授"}, id="call_1")
        ]
        mock_result_with_tools.content = ""
        
        mock_chain.invoke.return_value = mock_result_with_tools
        
        with patch.object(self.mock_llm, 'bind_tools', return_value=mock_chain):
            result = academic_agent(test_state)
            
            success = self._validate_agent_result(result, expect_tool_calls=True)
            self.test_results.append(("基础Agent-有工具调用", success))
            
            if success:
                print("✅ 测试通过")
                print(f"报告内容: {result['academic_analysis_report']}")
            else:
                print("❌ 测试失败")
        
        # 测试3: 错误处理
        print("\n3. 测试错误处理:")
        with patch.object(self.mock_llm, 'bind_tools') as mock_bind_tools:
            mock_bind_tools.side_effect = Exception("模拟API错误")
            
            result = academic_agent(test_state)
            success = "学术分析过程中发生错误" in result.get("academic_analysis_report", "")
            self.test_results.append(("基础Agent-错误处理", success))
            
            if success:
                print("✅ 错误处理测试通过")
            else:
                print("❌ 错误处理测试失败")
    
    def run_graph_test(self):
        """运行图测试"""
        print("\n=== 图结构测试 ===")
        
        try:
            # 创建测试图
            test_graph = create_academic_test_graph(self.mock_llm)
            
            print("1. 图创建测试:")
            success = test_graph.graph is not None
            self.test_results.append(("图创建", success))
            
            if success:
                print("✅ 图创建成功")
            else:
                print("❌ 图创建失败")
                return
            
            # 测试图执行
            print("\n2. 图执行测试:")
            test_state = {
                "research_topic": ["人工智能"],
                "research_person_info": "张博士，AI研究员",
                "messages": []
            }
            
            # 模拟执行
            mock_result = Mock()
            mock_result.tool_calls = []
            mock_result.content = "学术分析完成"
            
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result
            
            with patch.object(self.mock_llm, 'bind_tools', return_value=mock_chain):
                try:
                    final_state = test_graph.run(test_state)
                    success = "academic_analysis_report" in final_state
                    self.test_results.append(("图执行", success))
                    
                    if success:
                        print("✅ 图执行测试通过")
                    else:
                        print("❌ 图执行测试失败")
                except Exception as e:
                    print(f"❌ 图执行出错: {e}")
                    self.test_results.append(("图执行", False))
            
        except Exception as e:
            print(f"❌ 图测试出错: {e}")
            self.test_results.append(("图测试", False))
    
    def run_integration_test(self):
        """运行集成测试"""
        print("\n=== 集成测试 ===")
        
        # 模拟完整的工作流程
        print("1. 完整工作流程测试:")
        
        test_scenarios = [
            {
                "name": "计算机科学研究者",
                "state": {
                    "research_topic": ["深度学习", "计算机视觉"],
                    "research_person_info": "李教授，博士，发表顶会论文20篇，h指数15",
                }
            },
            {
                "name": "跨学科研究者", 
                "state": {
                    "research_topic": ["生物信息学", "机器学习"],
                    "research_person_info": "陈博士，生物学背景，转向AI应用研究",
                }
            },
            {
                "name": "新兴领域研究者",
                "state": {
                    "research_topic": ["量子计算", "量子机器学习"],
                    "research_person_info": "刘副教授，物理学博士，专注量子算法",
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n测试场景: {scenario['name']}")
            
            academic_agent = create_academic_agent(self.mock_llm, {})
            
            # 准备测试状态
            test_state = scenario["state"].copy()
            test_state["messages"] = []
            
            # 模拟不同的LLM响应
            mock_result = Mock()
            mock_result.tool_calls = []
            mock_result.content = self._generate_scenario_report(scenario["name"])
            
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result
            
            with patch.object(self.mock_llm, 'bind_tools', return_value=mock_chain):
                result = academic_agent(test_state)
                
                success = self._validate_agent_result(result, expect_tool_calls=False)
                self.test_results.append((f"集成测试-{scenario['name']}", success))
                
                if success:
                    print(f"✅ {scenario['name']} 测试通过")
                else:
                    print(f"❌ {scenario['name']} 测试失败")
    
    def run_performance_test(self):
        """运行性能测试"""
        print("\n=== 性能测试 ===")
        
        import time
        
        # 测试响应时间
        academic_agent = create_academic_agent(self.mock_llm, {})
        
        test_state = {
            "messages": [],
            "research_topic": ["测试领域"],
            "research_person_info": "测试研究者",
        }
        
        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.content = "快速响应测试"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        
        # 测试多次调用的性能
        times = []
        for i in range(5):
            with patch.object(self.mock_llm, 'bind_tools', return_value=mock_chain):
                start_time = time.time()
                result = academic_agent(test_state)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        success = avg_time < 1.0  # 期望平均响应时间小于1秒
        
        self.test_results.append(("性能测试", success))
        
        if success:
            print(f"✅ 性能测试通过，平均响应时间: {avg_time:.3f}秒")
        else:
            print(f"❌ 性能测试失败，平均响应时间: {avg_time:.3f}秒")
    
    def _generate_mock_academic_report(self) -> str:
        """生成模拟的学术分析报告"""
        return """
# 学术分析报告

## 申请人基本信息
- **姓名**: 王教授
- **学历**: 博士
- **研究方向**: 机器学习算法研究

## 学术成果分析
### 发表论文
申请人发表论文50余篇，显示出较强的学术产出能力。

### 研究影响力
在机器学习和数据挖掘领域有一定影响力。

## 综合评估
申请人具备扎实的学术基础和丰富的研究经验。

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 学术产出 | A | 论文数量充足 |
| 研究质量 | B+ | 需要进一步调研 |
| 影响力 | B | 有一定声誉 |
        """.strip()
    
    def _generate_scenario_report(self, scenario_name: str) -> str:
        """为特定场景生成报告"""
        reports = {
            "计算机科学研究者": "该研究者在深度学习和计算机视觉领域表现优异，h指数15表明其研究影响力较强。",
            "跨学科研究者": "该研究者具有生物学背景，转向AI应用研究，体现了良好的跨学科整合能力。",
            "新兴领域研究者": "该研究者专注于量子计算这一前沿领域，具有开拓性研究潜力。"
        }
        return reports.get(scenario_name, "标准学术分析报告")
    
    def _validate_agent_result(self, result: Dict[str, Any], expect_tool_calls: bool = False) -> bool:
        """验证agent结果"""
        if not isinstance(result, dict):
            return False
        
        if "messages" not in result or "academic_analysis_report" not in result:
            return False
        
        if expect_tool_calls:
            return result["academic_analysis_report"] == "正在使用学术分析工具进行深度调研..."
        else:
            return len(result["academic_analysis_report"]) > 10
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*50)
        print("测试总结")
        print("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, success in self.test_results if success)
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        print("\n详细结果:")
        for test_name, success in self.test_results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  {test_name}: {status}")
        
        if passed_tests == total_tests:
            print("\n🎉 所有测试通过!")
        else:
            print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查相关功能")


def main():
    """主函数"""
    print("学术分析Agent和图结构测试")
    print("="*50)
    
    runner = AcademicTestRunner()
    
    try:
        # 运行所有测试
        runner.run_basic_agent_test()
        runner.run_graph_test()
        runner.run_integration_test()
        runner.run_performance_test()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生未预期错误: {e}")
    finally:
        # 打印总结
        runner.print_summary()


if __name__ == "__main__":
    main()
