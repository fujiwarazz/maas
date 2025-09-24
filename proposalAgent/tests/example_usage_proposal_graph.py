#!/usr/bin/env python3
"""
ProposalAgentGraph 使用示例

展示如何使用更新后的 ProposalAgentGraph 类进行项目评估，
包括 interrupt API 的使用。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from proposalAgent.graphs.proposal_graph import ProposalAgentGraph
from proposalAgent.model_config import TONGYI_CONFIG

def main():
    """主函数演示 ProposalAgentGraph 的使用"""
    print("=== ProposalAgentGraph Interrupt API 使用示例 ===\n")
    
    # 1. 初始化项目评估图
    print("1. 初始化项目评估系统...")
    try:
        # 使用默认配置初始化
        graph = ProposalAgentGraph(config=TONGYI_CONFIG)
        print("✅ 系统初始化成功")
        
        # 显示系统信息
        info = graph.get_graph_info()
        print(f"   - LLM提供商: {info['llm_provider']}")
        print(f"   - 深度思考模型: {info['deep_think_model']}")
        print(f"   - 快速思考模型: {info['quick_think_model']}")
        print(f"   - 支持中断: {info['interrupt_supported']}")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 2. 开始项目评估
    print("\n2. 开始项目评估...")
    
    project_prompt = "评估基于深度学习的智能医疗诊断辅助系统项目"
    user_interests = ["技术可行性", "社会影响", "创新性", "风险评估"]
    
    print(f"   - 项目: {project_prompt}")
    print(f"   - 关注点: {', '.join(user_interests)}")
    
    # 执行评估
    result = graph.evaluate_project(
        user_prompt=project_prompt,
        user_interests=user_interests,
        filepath="",  # 可以提供项目文档路径
        thread_id="medical_ai_demo_001"
    )
    
    # 3. 处理评估结果
    print(f"\n3. 评估结果: {result['status']}")
    
    if result["status"] == "interrupted":
        print("🛑 评估被中断，需要人类审核")
        print(f"   - 线程ID: {result['thread_id']}")
        print(f"   - 中断原因: {result['message']}")
        
        # 显示中断信息
        interrupt_info = result["interrupt_info"]
        print(f"   - 任务: {interrupt_info.get('task', '未知')}")
        
        # 显示部分结果
        partial = result.get("partial_result", {})
        if partial.get("academic_analysis_report"):
            print(f"   - 已完成学术分析: {len(partial['academic_analysis_report'])} 字符")
        if partial.get("social_analysis_report"):
            print(f"   - 已完成社会分析: {len(partial['social_analysis_report'])} 字符")
        
        print("\n" + "="*60)
        print("👤 人类审核模拟")
        print("="*60)
        
        # 模拟人类审核过程
        print("\n审核员审查分析结果...")
        print("发现以下问题需要改进：")
        print("- 技术风险评估不够详细")
        print("- 缺少与现有技术的对比")
        print("- 需要补充法规合规性分析")
        
        # 模拟人类反馈
        human_feedback = """
分析质量总体良好，但需要以下改进：

1. 技术风险评估需要更详细：
   - 深度学习模型的可解释性风险
   - 数据隐私和安全风险
   - 模型偏见和公平性问题

2. 竞争分析不足：
   - 与IBM Watson Health对比
   - 与Google DeepMind Health对比
   - 差异化优势分析

3. 法规合规性：
   - FDA医疗器械认证要求
   - HIPAA数据保护合规
   - 欧盟MDR法规要求

请基于以上反馈重新分析相关部分。
        """.strip()
        
        print(f"\n💬 人类反馈: {human_feedback[:100]}...")
        
        # 4. 恢复评估
        print("\n4. 恢复评估执行...")
        resume_result = graph.resume_evaluation(
            thread_id=result["thread_id"],
            human_feedback=human_feedback
        )
        
        print(f"   - 恢复状态: {resume_result['status']}")
        
        if resume_result["status"] == "completed":
            print("✅ 评估完成！")
            display_final_results(resume_result)
            
            # 5. 反思和记忆更新
            print("\n5. 反思和记忆更新...")
            evaluation_outcome = {
                "final_score": 8.5,
                "human_feedback": human_feedback,
                "improvement_suggestions": ["风险评估", "竞争分析", "法规合规"]
            }
            graph.reflect_and_remember(evaluation_outcome)
            print("✅ 反思完成，经验已更新到记忆系统")
            
        elif resume_result["status"] == "interrupted":
            print("🛑 评估再次中断")
            print("   需要进一步的人类审核...")
        else:
            print(f"❌ 恢复失败: {resume_result.get('error', '未知错误')}")
    
    elif result["status"] == "completed":
        print("✅ 评估直接完成（无需人类干预）")
        display_final_results(result)
    
    else:
        print(f"❌ 评估失败: {result.get('error', '未知错误')}")

def display_final_results(result):
    """显示最终评估结果"""
    print("\n📄 最终评估报告:")
    print("-" * 50)
    
    # 显示各维度分析摘要
    if result.get("academic_analysis"):
        print(f"🎓 学术分析: {len(result['academic_analysis'])} 字符")
    
    if result.get("social_analysis"):
        print(f"🌍 社会分析: {len(result['social_analysis'])} 字符")
    
    if result.get("future_influence"):
        print(f"🔮 未来影响: {len(result['future_influence'])} 字符")
    
    if result.get("debate_results"):
        debate_count = len(result["debate_results"])
        print(f"⚖️ 辩论结果: {debate_count} 个学科领域")
    
    # 显示最终报告摘要
    final_report = result.get("final_report", "")
    if final_report:
        print(f"\n📋 最终报告长度: {len(final_report)} 字符")
        print(f"报告摘要: {final_report[:200]}...")
    
    # 显示分析摘要
    analysis_summary = result.get("analysis_summary", "")
    if analysis_summary:
        print(f"\n📊 分析摘要: {analysis_summary[:200]}...")

def demonstrate_api_features():
    """演示API功能特性"""
    print("\n" + "="*60)
    print("🔧 ProposalAgentGraph API 功能特性")
    print("="*60)
    
    print("""
✨ 主要功能:

1️⃣ evaluate_project() - 执行项目评估
   - 支持自定义用户关注点
   - 自动生成线程ID或使用自定义ID
   - 智能中断检测和处理

2️⃣ resume_evaluation() - 恢复中断的评估
   - 使用线程ID恢复特定评估
   - 传递人类反馈继续执行
   - 支持多次中断和恢复

3️⃣ reflect_and_remember() - 反思和记忆更新
   - 基于评估结果更新各组件记忆
   - 提高未来评估质量
   - 持续学习和改进

4️⃣ get_evaluation_status() - 查询评估状态
   - 检查特定线程的执行状态
   - 监控长时间运行的评估

5️⃣ get_graph_info() - 获取系统信息
   - 显示配置信息
   - 检查功能支持状态

🎯 使用场景:
- 复杂项目的多维度评估
- 需要人类专家审核的评估
- 大规模项目评估的自动化
- 评估质量的持续改进

⚡ 技术特性:
- 基于LangGraph的工作流引擎
- 支持多种LLM提供商
- 内置记忆和学习机制
- 人机协作的智能中断
- 可扩展的工具和代理系统
""")

if __name__ == "__main__":
    main()
    demonstrate_api_features()



