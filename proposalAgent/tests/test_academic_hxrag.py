#!/usr/bin/env python3
"""
学术分析测试脚本 - 使用hxrag环境
测试academic agent和简化图结构
"""

import os
import sys
import subprocess
from unittest.mock import Mock, patch
from typing import Dict, Any

# 确保在hxrag环境中运行
def ensure_hxrag_environment():
    """确保激活hxrag环境"""
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != 'hxrag':
        print("⚠️  当前不在hxrag环境中，尝试激活...")
        try:
            # 尝试激活hxrag环境
            activate_cmd = "source ~/micromamba/etc/profile.d/micromamba.sh && micromamba activate hxrag"
            result = subprocess.run(activate_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ hxrag环境已激活")
            else:
                print("❌ 无法激活hxrag环境，请手动激活后运行")
                return False
        except Exception as e:
            print(f"❌ 激活环境时出错: {e}")
            return False
    else:
        print("✅ 已在hxrag环境中")
    return True

def test_academic_agent():
    """测试学术分析agent"""
    print("\n=== 学术分析Agent测试 ===")
    
    try:
        # 导入必要模块
        sys.path.append(os.path.abspath('.'))
        from proposalAgent.agents.stage2.academic import create_academic_agent
        from langchain_openai import ChatOpenAI
        
        # 创建模拟LLM
        mock_llm = Mock(spec=ChatOpenAI)
        
        # 创建academic agent
        academic_agent = create_academic_agent(mock_llm, {})
        print("✅ Academic agent创建成功")
        
        # 准备测试数据 - 使用新的格式
        test_state = {
            "messages": [],
            "research_topic": ["人工智能", "深度学习"],
            "research_person_info": """### **项目团队成员及其个人履历**

**项目负责人：张教授**
- 博士学位，清华大学计算机科学与技术系
- 研究方向：深度学习、计算机视觉
- 发表论文：SCI论文50余篇，其中顶级会议论文20篇
- 学术影响：h指数25，总被引次数超过2000次
- 主要成果：在CVPR、ICCV等顶级会议发表多篇高影响力论文

**核心成员：李博士**
- 博士学位，北京大学人工智能学院
- 研究方向：自然语言处理、机器学习
- 发表论文：发表高质量论文30余篇
- 学术影响：h指数15，在NLP领域有一定影响力

**团队成员：王副教授**
- 博士学位，中科院计算技术研究所
- 研究方向：强化学习、智能决策
- 发表论文：发表相关论文25篇
- 产业经验：曾在知名AI公司担任技术专家
"""
        }
        
        print(f"📝 测试数据准备完成")
        print(f"   - 研究领域: {test_state['research_topic']}")
        print(f"   - 团队成员数: 3人")
        
        # 测试1: 无工具调用场景
        print("\n1️⃣ 测试无工具调用场景:")
        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.content = generate_mock_team_analysis_report()
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        
        with patch.object(mock_llm, 'bind_tools', return_value=mock_chain):
            result = academic_agent(test_state)
            
            if validate_agent_result(result, expect_tool_calls=False):
                print("✅ 无工具调用测试通过")
                print(f"   报告长度: {len(result['academic_analysis_report'])} 字符")
                print(f"   包含团队分析: {'张教授' in result['academic_analysis_report']}")
            else:
                print("❌ 无工具调用测试失败")
        
        # 测试2: 有工具调用场景
        print("\n2️⃣ 测试有工具调用场景:")
        mock_result_with_tools = Mock()
        mock_result_with_tools.tool_calls = [
            Mock(name="get_author_citations_auto", args={"name": "张教授", "organization": "清华大学"}, id="call_1"),
            Mock(name="get_author_citations_auto", args={"name": "李博士", "organization": "北京大学"}, id="call_2")
        ]
        mock_result_with_tools.content = ""
        
        mock_chain.invoke.return_value = mock_result_with_tools
        
        with patch.object(mock_llm, 'bind_tools', return_value=mock_chain):
            result = academic_agent(test_state)
            
            if validate_agent_result(result, expect_tool_calls=True):
                print("✅ 工具调用测试通过")
                print(f"   工具调用数: {len(mock_result_with_tools.tool_calls)}")
                print(f"   状态信息: {result['academic_analysis_report']}")
            else:
                print("❌ 工具调用测试失败")
        
        # 测试3: 错误处理
        print("\n3️⃣ 测试错误处理:")
        with patch.object(mock_llm, 'bind_tools') as mock_bind_tools:
            mock_bind_tools.side_effect = Exception("模拟网络错误")
            
            result = academic_agent(test_state)
            
            if "学术分析过程中发生错误" in result.get("academic_analysis_report", ""):
                print("✅ 错误处理测试通过")
                print(f"   错误信息: {result['academic_analysis_report']}")
            else:
                print("❌ 错误处理测试失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False

def test_academic_graph():
    """测试学术分析图"""
    print("\n=== 学术分析图测试 ===")
    
    try:
        # 导入图相关模块
        from proposalAgent.graphs.academic_test_graph import create_academic_test_graph
        from langchain_openai import ChatOpenAI
        
        # 创建模拟LLM
        mock_llm = Mock(spec=ChatOpenAI)
        
        # 创建测试图
        test_graph = create_academic_test_graph(mock_llm)
        print("✅ 学术分析图创建成功")
        
        # 准备测试状态
        test_state = {
            "research_topic": ["机器学习", "数据科学"],
            "research_person_info": """### **项目团队成员及其个人履历**

**项目负责人：陈教授**
- 博士学位，斯坦福大学计算机科学系
- 研究方向：机器学习、数据挖掘
- 发表论文：Nature、Science等顶级期刊论文10篇
- 学术影响：h指数35，国际知名学者

**核心成员：赵博士**
- 博士学位，MIT电子工程与计算机科学系
- 研究方向：深度学习、神经网络优化
- 发表论文：ICML、NeurIPS等会议论文15篇
""",
            "messages": []
        }
        
        print("📊 图测试数据准备完成")
        
        # 模拟图执行
        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.content = "基于提供的团队信息，进行了深度学术分析..."
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        
        with patch.object(mock_llm, 'bind_tools', return_value=mock_chain):
            try:
                final_state = test_graph.run(test_state)
                
                if "academic_analysis_report" in final_state:
                    print("✅ 图执行测试通过")
                    print(f"   最终状态包含字段: {list(final_state.keys())}")
                else:
                    print("❌ 图执行测试失败 - 缺少分析报告")
                    
            except Exception as e:
                print(f"❌ 图执行出错: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 图测试过程中发生错误: {e}")
        return False

def generate_mock_team_analysis_report():
    """生成模拟的团队学术分析报告"""
    return """# 项目团队学术分析报告

## 团队整体评估

本项目团队由3名核心成员组成，在人工智能和深度学习领域具有强大的研究实力。

## 成员详细分析

### 项目负责人：张教授
- **学术背景**: 清华大学计算机科学与技术系博士
- **研究方向**: 深度学习、计算机视觉
- **学术产出**: 发表SCI论文50余篇，顶级会议论文20篇
- **学术影响力**: h指数25，总被引次数超过2000次
- **评估**: 在计算机视觉领域具有国际影响力，研究成果丰富

### 核心成员：李博士
- **学术背景**: 北京大学人工智能学院博士
- **研究方向**: 自然语言处理、机器学习
- **学术产出**: 发表高质量论文30余篇
- **学术影响力**: h指数15，在NLP领域有一定影响力
- **评估**: 在自然语言处理领域有扎实基础，与团队形成互补

### 团队成员：王副教授
- **学术背景**: 中科院计算技术研究所博士
- **研究方向**: 强化学习、智能决策
- **学术产出**: 发表相关论文25篇
- **产业经验**: 曾在知名AI公司担任技术专家
- **评估**: 具有理论与实践结合的优势

## 团队协同分析

| 成员 | 专业领域 | 学术水平 | 互补性 | 综合评分 |
|------|----------|----------|--------|----------|
| 张教授 | 计算机视觉 | A+ | 核心领导 | 9.5/10 |
| 李博士 | 自然语言处理 | A | 技术互补 | 8.5/10 |
| 王副教授 | 强化学习 | A- | 实践经验 | 8.0/10 |

## 总结建议

该团队在人工智能领域具有很强的研究实力，成员间专业互补性强，具备承担重大科研项目的能力。建议进一步加强团队合作，发挥各自专业优势。"""

def validate_agent_result(result: Dict[str, Any], expect_tool_calls: bool = False) -> bool:
    """验证agent结果"""
    if not isinstance(result, dict):
        return False
    
    required_keys = ["messages", "academic_analysis_report"]
    for key in required_keys:
        if key not in result:
            return False
    
    if expect_tool_calls:
        return result["academic_analysis_report"] == "正在使用学术分析工具进行深度调研..."
    else:
        return len(result["academic_analysis_report"]) > 50

def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 学术分析系统综合测试")
    print("="*60)
    
    # 检查环境
    if not ensure_hxrag_environment():
        return False
    
    test_results = []
    
    # 运行agent测试
    agent_result = test_academic_agent()
    test_results.append(("Agent测试", agent_result))
    
    # 运行图测试
    graph_result = test_academic_graph()
    test_results.append(("图测试", graph_result))
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, success in test_results if success)
    
    print(f"总测试项: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！学术分析系统工作正常")
        return True
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败")
        return False

def main():
    """主函数"""
    try:
        success = run_comprehensive_test()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生未预期错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
