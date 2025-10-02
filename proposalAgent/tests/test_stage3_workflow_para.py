import sys
import os
import asyncio
import copy
from typing import Dict, Any, Optional

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.utils.agent_utils import Toolkit
from proposalAgent.graphs.conditional_logic import ConditionalLogic
from proposalAgent.agents.stage3.completeness_checker import create_completeness_checker_agent
from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.generator import create_generator_agent
from proposalAgent.agents.stage3.reflection_agent import create_reflection_agent
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.agents.utils.agent_utils import create_msg_delete

config = TONGYI_CONFIG
toolkit = Toolkit(config=config)
deep_think_llm = ChatOpenAI(model="qwen-plus",
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                            api_key=TONGYI_CONFIG.get("api_key"))

conditional_logic = ConditionalLogic()

workflow = StateGraph(AgentState)

# 创建智能体节点
final_analyst_node = create_final_analyst_agent(deep_think_llm)
completeness_checker_node = create_completeness_checker_agent(deep_think_llm)
feedback_analysis_node = create_feedback_analysis_agent(deep_think_llm)
generator_node = create_generator_agent(deep_think_llm)
reflection_node = create_reflection_agent(deep_think_llm)

# 人类审核节点
def human_review_node(state: AgentState) -> AgentState:
    """
    人类审核节点，使用interrupt API等待人类反馈
    """
    print("👤 等待人类审核...")
    print("="*60)
    print("📋 人类审核 - 需要检查的内容:")
    print("="*60)
    
    # 显示项目基本信息
    print(f"\n🔍 项目基本信息:")
    print(f"   研究主题: {state.get('research_topic', 'N/A')}")
    print(f"   申请人: {state.get('research_person_info', 'N/A')}")
    print(f"   申请代码: {state.get('research_project_apply_info', 'N/A')}")
    print(f"   研究团队: {state.get('research_project_team_info', 'N/A')}")
    
    # 显示各阶段分析结果
    print(f"\n📊 学术分析结果:")
    academic_report = state.get('academic_analysis_report', '未提供')
    print(f"   {academic_report[:200]}{'...' if len(academic_report) > 200 else ''}")
    
    print(f"\n🌍 社会分析结果:")
    social_report = state.get('social_analysis_report', '未提供')
    print(f"   {social_report[:200]}{'...' if len(social_report) > 200 else ''}")
    
    print(f"\n🔮 未来影响分析:")
    future_report = state.get('future_influence_report', '未提供')
    print(f"   {future_report[:200]}{'...' if len(future_report) > 200 else ''}")
    
    print(f"\n🔗 跨学科分析结果:")
    interdis_results = state.get('interdisciplinary_results', [])
    print(f"   涉及学科: {interdis_results}")
    
    print(f"\n⚖️ 辩论结果:")
    debate_results = state.get('debate_results', {})
    for debate_type, result in debate_results.items():
        if isinstance(result, dict):
            summary = result.get('judge_summary', 'N/A')
            print(f"   {debate_type}: {summary}")
        else:
            print(f"   {debate_type}: {result}")
    
    print(f"\n📝 最终分析摘要:")
    final_summary = state.get('final_analysis_summary', '未提供')
    print(f"   {final_summary[:300]}{'...' if len(final_summary) > 300 else ''}")
    
    # 显示完备性检查结果
    completeness_result = state.get('completeness_check_result', {})
    print(f"\n🔍 完备性检查结果:")
    print(f"   完整性: {completeness_result.get('is_complete', 'N/A')}")
    print(f"   一致性: {completeness_result.get('is_consistent', 'N/A')}")
    print(f"   质量评分: {completeness_result.get('overall_quality', 'N/A')}/5")
    print(f"   缺失部分: {completeness_result.get('missing_parts', [])}")
    print(f"   不一致问题: {completeness_result.get('inconsistencies', [])}")
    print(f"   建议: {completeness_result.get('recommendation', 'N/A')}")
    print(f"   原因: {completeness_result.get('reason', 'N/A')}")
    
    print("\n" + "="*60)
    print("🤔 人类审核提示:")
    print("请仔细检查以上内容，判断分析是否完整、准确、一致。")
    print("如果需要改进，请指出具体的问题和建议。")
    print("如果分析满足要求，可以输入'approved'。")
    print("="*60)
    
    # 模拟人类反馈（在实际使用中，这里会等待真实的人类输入）
    if not state.get('human_feedback'):
        # 根据完备性检查结果生成模拟反馈
        missing_parts = completeness_result.get('missing_parts', [])
        
        if "学术分析" in str(missing_parts):
            state['human_feedback'] = "学术分析部分需要更深入，特别是申请人的研究背景和学术能力评估"
        elif "社会分析" in str(missing_parts):
            state['human_feedback'] = "社会影响分析不够全面，需要补充更多实际案例和数据支撑"
        elif "未来影响" in str(missing_parts):
            state['human_feedback'] = "未来影响预测过于乐观，需要更客观的分析"
        else:
            state['human_feedback'] = "整体分析质量很好，可以直接生成报告"
    
    print(f"\n📝 收到人类反馈: {state['human_feedback']}")
    return state

# 路由函数
def _route_after_human_review(state: AgentState) -> str:
    """根据人类审核结果路由"""
    if state.get('skip_human_review'):
        return "generate"
    else:
        return "feedback_analysis"

def _route_after_feedback(state: AgentState) -> str:
    """根据反馈分析结果路由"""
    routing_decision = state.get('feedback_routing_decision', 'generate')
    return routing_decision

def _route_after_completeness(state: AgentState) -> str:
    """根据完备性检查结果路由"""
    if state.get('completeness_recommendation') == 'complete':
        return "generate"
    else:
        return "human_review"

# 添加节点到工作流
workflow.add_node("final_analyst_node", final_analyst_node)
workflow.add_node("completeness_checker_node", completeness_checker_node)
workflow.add_node("human_review_node", human_review_node)
workflow.add_node("feedback_analysis_node", feedback_analysis_node)
workflow.add_node("generator_node", generator_node)
workflow.add_node("reflection_node", reflection_node)

# 设置工作流路径
workflow.add_edge(START, "final_analyst_node")
workflow.add_edge("final_analyst_node", "completeness_checker_node")

# 完备性检查后的条件路由
workflow.add_conditional_edges(
    "completeness_checker_node",
    _route_after_completeness,
    {
        "generate": "generator_node",  # 完备性检查通过，直接生成
        "human_review": "human_review_node",  # 需要人类审核
    },
)

# 人类审核后的条件路由
workflow.add_conditional_edges(
    "human_review_node",
    _route_after_human_review,
    {
        "generate": "generator_node",  # 直接生成
        "feedback_analysis": "feedback_analysis_node",  # 需要分析反馈
    },
)

# 反馈分析后的条件路由
workflow.add_conditional_edges(
    "feedback_analysis_node",
    _route_after_feedback,
    {
        "academic_analysis": "generator_node",  # 简化路由，直接到生成器
        "social_analysis": "generator_node",
        "future_influence": "generator_node",
        "interdisciplinary": "generator_node",
        "debate": "generator_node",
        "generate": "generator_node",
    },
)

# 生成报告后到反思
workflow.add_edge("generator_node", "reflection_node")
workflow.add_edge("reflection_node", END)

# 编译工作流
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

async def main():
    """主函数"""
    print("🚀 开始Stage3工作流测试")
    print("流程: final_analysis总结 -> 判断完备 -> 引入人类 -> 人类评审的判断 -> 信息补全 -> 生成报告")
    
    state = {
        "messages": [],
        "research_topic": "基于大语言模型的智能教育系统研究",
        "research_basic_info": """
        申请人: 张三，清华大学计算机系副教授
        申请代码: F0212 数据科学与大数据计算
        研究周期: 3年
        申请金额: 80万元
        研究团队: 5人，包括2名博士生，2名硕士生，1名本科生
        """,
        "research_structure": "未提供",
        "research_person_info": "张三，清华大学计算机系副教授",
        "research_project_team_info": "5人团队，包括2名博士生，2名硕士生，1名本科生",
        "research_project_apply_info": "申请代码: F0212，研究周期: 3年，申请金额: 80万元",
        "research_report_body_summary": "未提供",
        
        "academic_analysis_report": """
        学术能力评估:
        - 申请人具有扎实的计算机科学背景，在自然语言处理领域有丰富经验
        - 已发表SCI论文15篇，其中一区论文8篇
        - 主持过2项国家自然科学基金项目
        - 在Transformer架构优化方面有重要贡献
        """,
        "social_analysis_report": """
        社会影响分析:
        - 教育领域数字化转型的重要技术支撑
        - 有助于提升教育公平性和个性化学习
        - 可能对传统教育模式产生深远影响
        - 需要关注数据隐私和算法公平性问题
        """,
        "future_influence_report": """
        未来影响预测:
        - 短期(1-2年): 在教育辅助工具方面有应用前景
        - 中期(3-5年): 可能推动个性化教育模式普及
        - 长期(5-10年): 有望重塑教育生态系统
        - 风险: 技术依赖、数字鸿沟、伦理问题
        """,
        
        "interdisciplinary_results": ["教育学", "心理学", "伦理学", "数据科学"],
        "current_discipline": "计算机科学",
        "debate_results": {
            "feasibility": {
                "judge_summary": "技术基础扎实，团队配置合理，预期目标可实现",
                "full_history": "可行性辩论：正方认为技术成熟，反方担心资源不足，裁判认为可行"
            },
            "innovation": {
                "judge_summary": "在个性化教育和大模型结合方面有创新点",
                "full_history": "创新性辩论：正方强调技术融合创新，反方质疑创新程度，裁判认为有创新"
            }
        },
        
        "final_analysis_summary": """
        综合分析：
        该项目在技术可行性、团队配置、创新性方面表现良好，但在社会影响评估和风险分析方面需要进一步完善。
        建议加强跨学科协作，完善伦理审查机制。
        """,
        
        "completeness_check_result": {},
        "is_analysis_complete": None,
        "is_analysis_consistent": None,
        "completeness_recommendation": "",
        "skip_human_review": None,
        
        "reflection_decision": "",
        "human_feedback": "",
        
        "feedback_analysis_result": {},
        "feedback_routing_decision": "",
        "feedback_instructions": "",
        
        "final_report": ""
    }
    
    # 运行工作流
    config = {"configurable": {"thread_id": "stage3_test_thread"}}
    result = await graph.ainvoke(state, config=config)
    
    print("\n" + "="*60)
    print("🎉 Stage3工作流测试完成!")
    print("="*60)
    
    # 显示关键结果
    print(f"\n📋 最终结果:")
    print(f"   完备性检查: {result.get('completeness_recommendation', 'N/A')}")
    print(f"   人类反馈: {result.get('human_feedback', 'N/A')}")
    print(f"   反馈分析: {result.get('feedback_routing_decision', 'N/A')}")
    print(f"   最终报告长度: {len(result.get('final_report', ''))}")
    print(f"   反思结果: {result.get('reflection_decision', 'N/A')}")
    
    return result

if __name__ == "__main__":
    asyncio.run(main())
