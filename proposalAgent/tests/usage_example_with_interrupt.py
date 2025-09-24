#!/usr/bin/env python3
"""
实际使用示例：如何使用更新后的项目评估图与interrupt API

这个文件展示了如何在实际项目中使用带有interrupt API的评估图。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.types import Command
from proposalAgent.graphs.setup import GraphSetup
from proposalAgent.graphs.propagation import Propagator
from proposalAgent.agents.utils.agent_utils import Toolkit
from proposalAgent.model_config import get_models
from langchain_core.messages import HumanMessage

def simulate_usage():
    """模拟实际使用场景"""
    print("=== 项目评估系统 - Interrupt API 使用示例 ===\n")
    
    # 注意：这里只是展示API使用方式，实际运行需要完整的环境配置
    print("1. 初始化系统组件...")
    
    # 模拟配置（实际使用时需要真实的配置）
    print("   - 配置LLM模型")
    print("   - 初始化工具包")
    print("   - 设置记忆系统")
    
    print("\n2. 创建项目评估图...")
    # 实际代码示例（需要真实配置才能运行）:
    """
    # 获取模型配置
    models = get_models()
    quick_llm = models['quick_thinking_llm']
    deep_llm = models['deep_thinking_llm']
    
    # 创建工具包和记忆
    toolkit = Toolkit()
    # ... 其他记忆和工具配置
    
    # 创建图设置
    graph_setup = GraphSetup(
        quick_thinking_llm=quick_llm,
        deep_think_llm=deep_llm,
        # ... 其他参数
    )
    
    # 构建图
    graph = graph_setup.setup_graph()
    """
    
    print("\n3. 准备初始状态...")
    propagator = Propagator()
    initial_state = propagator.create_initial_state(
        user_prompt="评估基于深度学习的医疗诊断辅助系统项目",
        user_interest=["技术可行性", "社会影响", "创新性"],
        filepath="/path/to/project/document.pdf"
    )
    
    print(f"   - 项目主题: {initial_state['research_topic']}")
    print(f"   - 关注重点: 技术可行性, 社会影响, 创新性")
    
    print("\n4. 开始执行评估流程...")
    thread_config = {"configurable": {"thread_id": "medical_ai_project_001"}}
    
    print("   - 执行意图识别...")
    print("   - 进行结构化分析...")
    print("   - 执行学术分析...")
    print("   - 进行社会影响分析...")
    print("   - 分析未来影响...")
    print("   - 跨学科分析...")
    print("   - 进行可行性和创新性辩论...")
    print("   - 生成最终分析...")
    print("   - 执行完备性检查...")
    
    print("\n5. 遇到人类审核点...")
    print("   ⏸️  图形在human_review_node处中断")
    print("   📋 中断信息包含:")
    print("      - 任务描述: 请审查项目评估分析并提供反馈意见")
    print("      - 分析摘要: [完整的分析报告]")
    print("      - 完备性问题: [检测到的问题列表]")
    print("      - 各维度报告: [学术、社会、未来影响等报告]")
    print("      - 辩论结果: [可行性和创新性辩论摘要]")
    
    print("\n6. 人类审核过程...")
    print("   👤 人类审核员审查分析结果")
    print("   📝 审核员发现需要补充以下内容:")
    print("      - 需要更详细的风险评估")
    print("      - 缺少与现有技术的对比分析")
    print("      - 建议增加成本效益分析")
    
    print("\n7. 恢复执行...")
    human_feedback = """
    分析整体质量良好，但需要以下改进：
    1. 风险评估部分需要更详细，特别是技术风险和法规风险
    2. 缺少与现有医疗AI技术的对比分析
    3. 建议增加详细的成本效益分析
    4. 社会接受度分析可以更深入
    请重新进行学术分析和社会分析，补充以上内容。
    """
    
    print(f"   💬 人类反馈: {human_feedback[:100]}...")
    
    # 模拟恢复执行的代码:
    """
    # 使用Command.resume恢复执行
    resume_result = graph.invoke(
        Command(resume={"feedback": human_feedback}),
        config=thread_config
    )
    """
    
    print("\n8. 反馈分析和重新执行...")
    print("   🔄 反馈分析代理分析人类反馈")
    print("   📍 路由决策: 返回学术分析节点")
    print("   🔄 重新执行学术分析（补充风险和对比分析）")
    print("   🔄 重新执行社会分析（深化社会接受度分析）")
    print("   ⚡ 跳过其他已完成的分析")
    print("   📊 重新生成最终分析")
    print("   ✅ 完备性检查通过")
    
    print("\n9. 生成最终报告...")
    print("   📄 生成器代理创建最终评估报告")
    print("   ✅ 评估流程完成")
    
    print("\n=== 关键优势 ===")
    print("✨ 使用interrupt API的优势:")
    print("   1. 动态中断: 只在真正需要时中断")
    print("   2. 丰富上下文: 中断时提供完整的分析上下文")
    print("   3. 灵活恢复: 可以携带人类反馈恢复执行")
    print("   4. 智能路由: 根据反馈智能决定下一步")
    print("   5. 无需预定义: 不需要预先定义中断点")

def show_api_usage():
    """展示API使用方法"""
    print("\n=== API 使用方法 ===")
    
    print("""
# 1. 在节点中使用interrupt API
def human_review_node(state):
    # 执行完备性检查
    state = completeness_checker_node(state)
    
    if state["completeness_recommendation"] == "complete":
        return state  # 直接继续
    
    # 需要人类输入时使用interrupt
    result = interrupt({
        "task": "请审查分析并提供反馈",
        "analysis_data": state["final_analysis_summary"],
        "instructions": "请提供反馈或输入'approved'"
    })
    
    state["human_feedback"] = result["feedback"]
    return state

# 2. 编译图（不需要interrupt_before）
graph = workflow.compile()

# 3. 执行图
thread_config = {"configurable": {"thread_id": "unique_id"}}
try:
    result = graph.invoke(initial_state, config=thread_config)
except InterruptException:
    # 图在interrupt点暂停
    pass

# 4. 恢复执行
resume_result = graph.invoke(
    Command(resume={"feedback": "用户的反馈内容"}),
    config=thread_config
)
""")

if __name__ == "__main__":
    simulate_usage()
    show_api_usage()
