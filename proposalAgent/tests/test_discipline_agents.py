"""
测试学科Agent系统
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.collaborative_evaluation_system import CollaborativeEvaluationSystem

async def test_discipline_agents():
    """测试学科Agent系统"""
    
    # 初始化系统
    system = CollaborativeEvaluationSystem("proposalAgent/data/course.json")
    system.load_agent_prompts()
    
    # 测试研究内容
    research_text = """
    本研究提出了一种基于深度学习和量子计算的新型算法，用于解决大规模优化问题。
    该算法结合了神经网络的学习能力和量子计算的并行优势，在理论上具有指数级的加速效果。
    我们通过实验验证了该算法在物流优化、金融风险管理和生物信息学等领域的有效性。
    """
    
    research_title = "基于深度学习和量子计算的混合优化算法研究"
    
    # 进行评估
    print("开始多学科协作评估...")
    result = await system.evaluate_research(research_text, research_title)
    
    # 输出结果
    print("\n=== 综合评估结果 ===")
    print(f"相关性评分: {result['overall_scores']['relevance']}/10")
    print(f"可行性评分: {result['overall_scores']['feasibility']}/10")
    print(f"创新性评分: {result['overall_scores']['innovation']}/10")
    
    print(f"\n综合建议: {result['recommendation']}")
    
    print("\n=== 各学科专业评估 ===")
    for eval in result['discipline_evaluations']:
        print(f"\n{eval['discipline_name']} ({eval['discipline_code']}):")
        print(f"  相关性: {eval['relevance_score']}/10")
        print(f"  可行性: {eval['feasibility_score']}/10")
        print(f"  创新性: {eval['innovation_score']}/10")
        print(f"  优势: {', '.join(eval['advantages'])}")
        print(f"  挑战: {', '.join(eval['challenges'])}")
    
    print("\n=== 综合分析 ===")
    analysis = result['synthesized_analysis']
    print(f"主要优势: {', '.join(analysis['key_advantages'])}")
    print(f"主要挑战: {', '.join(analysis['main_challenges'])}")
    print(f"改进建议: {', '.join(analysis['improvement_suggestions'])}")
    print(f"合作机会: {', '.join(analysis['collaboration_opportunities'])}")

if __name__ == "__main__":
    asyncio.run(test_discipline_agents())
