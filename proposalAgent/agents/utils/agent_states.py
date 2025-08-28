from typing import Annotated, Sequence
from datetime import date, timedelta, datetime
from typing_extensions import TypedDict, Optional
from langchain_openai import ChatOpenAI
from proposalAgent.agents import *
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START, MessagesState
from typing import List,Dict

# 这个 TypedDict 类定义了在“可行性”或“创新性”辩论环节中的状态。
# 它是一个可复用的结构，用于清晰地记录每一场正反方辩论的详细过程和结果。
class DebateState(TypedDict):
    """
    单个辩论（如可行性或创新性）的状态表示。
    这个结构用于追踪正方、反方和裁判代理之间的互动。
    """
    good_agent_history: Annotated[
        List[str], "History of the 'pro' side agent's arguments"
    ]  # 正方代理的发言历史，记录所有支持性论点。
    bad_agent_history: Annotated[
        List[str], "History of the 'con' side agent's arguments"
    ]  # 反方代理的发言历史，记录所有批判性或反对性论点。
    full_history: Annotated[
        List[str], "Full transcript of the debate"
    ]  # 整场辩论的完整对话记录。
    judge_summary: Annotated[
        str, "Final summary and conclusion from the judge agent"
    ]  # “裁判”代理在听取双方观点后，给出的最终总结和结论。
    debate_rounds: Annotated[int, "Number of rounds in the debate"] # 记录当前辩论进行的轮次。


class AgentState(MessagesState):
    """
    整个多智能体工作流的全局状态。
    它聚合了所有阶段和节点产生的信息，驱动整个决策流程。
    """
    # --- 阶段 0 & 1: 初始输入与规划 ---
    research_topic: Annotated[str, "The initial user input or research topic"] # 整个流程的起点，由用户输入的初始研究主题或问题。
    intention_decision: Annotated[
        str, "Decision from the intention node, e.g., 'output' or 'structure'"
    ] # 意图识别节点的输出，决定是直接输出还是进入复杂分析流程。
    research_structure: Annotated[
        str, "The structured outline of the research from the structure_node"
    ] # 结构规划节点生成的分析大纲。
    execution_plan: Annotated[
        List[str], "A detailed step-by-step plan from the planning_node"
    ] # 规划节点生成的详细执行计划。

    # --- 阶段 2: 信息收集 ---
    # 每个字段都存储了对应分析节点产出的报告或关键信息。
    academic_analysis_report: Annotated[Optional[str], "Report from the academic analysis"] # 学术分析节点的产出报告。
    social_analysis_report: Annotated[Optional[str], "Report from the social analysis"] # 社会分析节点的产出报告。
    future_influence_report: Annotated[Optional[str], "Report from the future influence analysis"] # 未来影响分析节点的产出报告。
    interdisciplinary_results: Annotated[
        List[str], "List of disciplines identified for debate"
    ] # 跨学科分析节点识别出的、需要进行后续辩论的学科领域列表。

    # --- 阶段 2: 辩论 ---
    # 这个字段结构比较复杂，用于存储所有并行辩论的结果。
    current_discipline: Annotated[
        Optional[str], "The discipline currently being debated"
    ] # 一个临时状态，用于在循环中告知辩论子图当前正在处理哪个学科。
    debate_results: Annotated[
        List[Dict[str, Dict[str, DebateState]]],
        "A list containing results for each discipline's debates (feasibility and innovation)"
    ] # 存储所有辩论的最终结果。结构为：[{学科A: {"可行性": DebateState, "创新性": DebateState}}, {学科B: ...}]

    # --- 阶段 3: 综合、反思与人机交互 ---
    final_analysis_summary: Annotated[
        str, "A comprehensive summary from the final_analyst_node"
    ] # 最终分析节点整合所有信息后生成的综合分析摘要。
    reflection_decision: Annotated[
        str, "Decision from reflection, e.g., 'generate' or 'review'"
    ] # 反思节点的输出，决定是直接生成报告还是请求人类审核。
    human_feedback: Annotated[
        Optional[str], "Feedback provided by the human reviewer"
    ] # 人类审核者提供的反馈意见，如果流程触发了人工审核，则会填充此字段。
    feedback_routing_decision: Annotated[
        str, "Decision on where to go after analyzing human feedback"
    ] # 反馈分析节点的输出，决定下一步应该跳转回哪个节点（如 'redo_academic', 'redo_debate', 'generate'）。

    # --- 最终产出 ---
    final_report: Annotated[Optional[str], "The final generated report"] # 生成器节点产出的最终报告。
