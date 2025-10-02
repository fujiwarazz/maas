import asyncio
import copy
from typing import Any, Dict
from logging import getLogger

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.output import create_output_node
from proposalAgent.agents.stage1.schedule import create_schedule_agent
from proposalAgent.agents.stage1.structure import create_structure_node
from proposalAgent.agents.stage2.academic import create_academic_agent
from proposalAgent.agents.stage2.debate.feasible.feasible_bad import (
    create_feasible_bad_agent,
)
from proposalAgent.agents.stage2.debate.feasible.feasible_good import (
    create_feasible_good_agent,
)
from proposalAgent.agents.stage2.debate.feasible.feasible_manager import (
    create_feasible_manager,
)
from proposalAgent.agents.stage2.debate.innovation.innovation_bad import (
    create_innovation_bad_agent,
)
from proposalAgent.agents.stage2.debate.innovation.innovation_good import (
    create_innovation_good_agent,
)
from proposalAgent.agents.stage2.debate.innovation.innovation_manager import (
    create_innovation_manager,
)
from proposalAgent.agents.stage2.future_influence import create_future_influence_agent
from proposalAgent.agents.stage2.interdis import create_interdis_agent
from proposalAgent.agents.stage3.completeness_checker import (
    create_completeness_checker_agent,
)
from proposalAgent.agents.stage3.feedback_analysis_agent import (
    create_feedback_analysis_agent,
)
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.agents.stage3.generator import create_generator_agent
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.utils.agent_utils import Toolkit, create_msg_delete
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.graphs.conditional_logic import ConditionalLogic
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.tools.academic_analysis.google_scholar import (
    get_article_brief,
    get_author_articles_citations,
    get_author_citations,
    get_author_citations_auto,
    resolve_author_candidates,
)
from proposalAgent.tools.baidu_util import baidu_search_with_content
from proposalAgent.tools.secondary_discipline_rag import secondary_discipline_search


tool_nodes = {
    "academic": ToolNode(
        [
            get_article_brief,
            resolve_author_candidates,
            get_author_citations,
            get_author_citations_auto,
            get_author_articles_citations,
        ]
    ),
    "social": ToolNode([]),
    "influence": ToolNode([baidu_search_with_content]),
    "interdisciplinary": ToolNode([secondary_discipline_search]),
    "feasibility": ToolNode([]),
    "innovation": ToolNode([]),
}


config = TONGYI_CONFIG
toolkit = Toolkit(config=config)
deep_think_llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=TONGYI_CONFIG.get("api_key"),
)


quick_think_llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=TONGYI_CONFIG.get("api_key"),
)


conditional_logic = ConditionalLogic()
logger = getLogger("WorkflowFullTest")
workflow = StateGraph(AgentState)


# === Stage 1 setup ===
intention_node = create_intention_agent(quick_think_llm)
output_node = create_output_node(quick_think_llm)
structure_node = create_structure_node()
planning_node = create_schedule_agent(deep_think_llm)
planning_clear_node = create_msg_delete()

workflow.add_node("intention_node", intention_node)
workflow.add_node("output_node", output_node)
workflow.add_node("structure_node", structure_node)
workflow.add_node("planning_node", planning_node)
workflow.add_node("planning_clear_node", planning_clear_node)

workflow.add_edge(START, "intention_node")
workflow.add_conditional_edges(
    "intention_node",
    conditional_logic.should_output,
    {"output_node": "output_node", "structure_node": "structure_node"},
)
workflow.add_edge("output_node", END)
workflow.add_edge("structure_node", "planning_node")
workflow.add_edge("planning_node", "planning_clear_node")


# === Stage 2 setup ===
academic_analysis_node = create_academic_agent(quick_think_llm, toolkit=toolkit)
academic_tool_exc_node = tool_nodes["academic"]
academic_msg_clear_node = create_msg_delete()

future_influence_node = create_future_influence_agent(deep_think_llm, toolkit)
future_influence_tool_exc_node = tool_nodes["influence"]
future_influence_msg_clear_node = create_msg_delete()

# academic subgraph
academic_workflow = StateGraph(AgentState)
academic_workflow.add_node("academic_analysis_node", academic_analysis_node)
academic_workflow.add_node("academic_tool_exc_node", academic_tool_exc_node)
academic_workflow.add_node("academic_msg_clear_node", academic_msg_clear_node)
academic_workflow.add_edge(START, "academic_analysis_node")
academic_workflow.add_conditional_edges(
    "academic_analysis_node",
    conditional_logic.should_continue_academic_analysis,
    {
        "tools_academic": "academic_tool_exc_node",
        "msg_clear_academic": "academic_msg_clear_node",
    },
)
academic_workflow.add_edge("academic_tool_exc_node", "academic_analysis_node")
academic_workflow.add_edge("academic_msg_clear_node", END)
compiled_academic_graph = academic_workflow.compile()


# future influence subgraph
future_influence_workflow = StateGraph(AgentState)
future_influence_workflow.add_node("future_influence_node", future_influence_node)
future_influence_workflow.add_node(
    "future_influence_tool_exc_node", future_influence_tool_exc_node
)
future_influence_workflow.add_node(
    "future_influence_msg_clear_node", future_influence_msg_clear_node
)
future_influence_workflow.add_edge(START, "future_influence_node")
future_influence_workflow.add_conditional_edges(
    "future_influence_node",
    conditional_logic.should_continue_future_influence,
    {
        "tools_future_influence": "future_influence_tool_exc_node",
        "msg_clear_future_influence": "future_influence_msg_clear_node",
    },
)
future_influence_workflow.add_edge(
    "future_influence_tool_exc_node", "future_influence_node"
)
future_influence_workflow.add_edge("future_influence_msg_clear_node", END)
compiled_future_influence_graph = future_influence_workflow.compile()


async def stage2_parallel_runner(state: AgentState):
    academic_state = copy.deepcopy(state)
    future_state = copy.deepcopy(state)

    academic_result, future_result = await asyncio.gather(
        compiled_academic_graph.ainvoke(academic_state),
        compiled_future_influence_graph.ainvoke(future_state),
    )

    merged_state = copy.deepcopy(state)

    for key, value in academic_result.items():
        if key == "messages":
            continue
        merged_state[key] = value
  
    for key, value in future_result.items():
        if key == "messages":
            continue
        if key in merged_state and not value:
            continue
        merged_state[key] = value
   
    
    merged_state["messages"] = state.get("messages")
  
    return merged_state


interdisciplinary_node = create_interdis_agent(deep_think_llm, toolkit)
interdisciplinary_tool_exc_node = tool_nodes["interdisciplinary"]
interdisciplinary_msg_clear_node = create_msg_delete()


workflow.add_node("academic_analysis_node", academic_analysis_node)
workflow.add_node("academic_tool_exc_node", academic_tool_exc_node)
workflow.add_node("academic_msg_clear_node", academic_msg_clear_node)
workflow.add_node("future_influence_node", future_influence_node)
workflow.add_node(
    "future_influence_tool_exc_node", future_influence_tool_exc_node
)
workflow.add_node(
    "future_influence_msg_clear_node", future_influence_msg_clear_node
)
workflow.add_node("interdisciplinary_node", interdisciplinary_node)
workflow.add_node(
    "interdisciplinary_tool_exc_node", interdisciplinary_tool_exc_node
)
workflow.add_node(
    "interdisciplinary_msg_clear_node", interdisciplinary_msg_clear_node
)
workflow.add_node("stage2_parallel_node", stage2_parallel_runner)

workflow.add_edge("planning_clear_node", "stage2_parallel_node")
workflow.add_edge("stage2_parallel_node", "interdisciplinary_node")

workflow.add_conditional_edges(
    "interdisciplinary_node",
    conditional_logic.should_continue_interdisciplinary,
    {
        "tools_interdisciplinary": "interdisciplinary_tool_exc_node",
        "msg_clear_interdisciplinary": "interdisciplinary_msg_clear_node",
    },
)

workflow.add_edge("interdisciplinary_tool_exc_node", "interdisciplinary_node")


# debate memories
feasible_good_memory = EmbeddingMemory(name="feasible_good_memory", config=config)
feasible_bad_memory = EmbeddingMemory(name="feasible_bad_memory", config=config)
feasible_manager_memory = EmbeddingMemory(name="feasible_manager_memory", config=config)

innovation_good_memory = EmbeddingMemory(name="innovation_good_memory", config=config)
innovation_bad_memory = EmbeddingMemory(name="innovation_bad_memory", config=config)
innovation_manager_memory = EmbeddingMemory(name="innovation_manager_memory", config=config)


feasible_good_node = create_feasible_good_agent(
    deep_think_llm, toolkit, feasible_good_memory
)
feasible_bad_node = create_feasible_bad_agent(
    deep_think_llm, toolkit, feasible_bad_memory
)
feasible_judge_node = create_feasible_manager(deep_think_llm, feasible_manager_memory)

innovation_good_node = create_innovation_good_agent(
    deep_think_llm, toolkit, innovation_good_memory
)
innovation_bad_node = create_innovation_bad_agent(
    deep_think_llm, toolkit, innovation_bad_memory
)
innovation_judge_node = create_innovation_manager(
    deep_think_llm, innovation_manager_memory
)


feasibility_debate_workflow = StateGraph(AgentState)
feasibility_debate_workflow.add_node("feasible_good_node", feasible_good_node)
feasibility_debate_workflow.add_node("feasible_bad_node", feasible_bad_node)
feasibility_debate_workflow.add_node("feasible_judge_node", feasible_judge_node)
feasibility_debate_workflow.add_edge(START, "feasible_good_node")
feasibility_debate_workflow.add_edge("feasible_good_node", "feasible_bad_node")
feasibility_debate_workflow.add_edge("feasible_bad_node", "feasible_judge_node")
feasibility_debate_workflow.add_edge("feasible_judge_node", END)
compiled_feasibility_debate_graph = feasibility_debate_workflow.compile()


innovation_debate_workflow = StateGraph(AgentState)
innovation_debate_workflow.add_node("innovation_good_node", innovation_good_node)
innovation_debate_workflow.add_node("innovation_bad_node", innovation_bad_node)
innovation_debate_workflow.add_node("innovation_judge_node", innovation_judge_node)
innovation_debate_workflow.add_edge(START, "innovation_good_node")
innovation_debate_workflow.add_edge("innovation_good_node", "innovation_bad_node")
innovation_debate_workflow.add_edge("innovation_bad_node", "innovation_judge_node")
innovation_debate_workflow.add_edge("innovation_judge_node", END)
compiled_innovation_debate_graph = innovation_debate_workflow.compile()


async def debate_controller(state: AgentState):
    disciplines = state.get("interdisciplinary_results", [])
    all_debate_outputs = {}
    tasks = []

    for discipline in disciplines:
        async def single_task():
            input_state = state.copy()
            input_state["messages"] = state["messages"] + [
                SystemMessage(content=f"Starting debates for discipline: {discipline}")
            ]
            input_state["current_discipline"] = discipline

            f_task = compiled_feasibility_debate_graph.ainvoke(input_state)
            i_task = compiled_innovation_debate_graph.ainvoke(input_state)
            results = await asyncio.gather(f_task, i_task)

            output_result_merge: Dict[str, Any] = {}
            output_result_merge["可行性"] = (
                results[0]
                .get("debate_results", {})
                .get(discipline, {})
                .get("可行性", {})
            )
            output_result_merge["创新性"] = (
                results[1]
                .get("debate_results", {})
                .get(discipline, {})
                .get("创新性", {})
            )

            return output_result_merge

        tasks.append(single_task())

    results = await asyncio.gather(*tasks) if tasks else []
    for discipline, result in zip(disciplines, results):
        all_debate_outputs[discipline] = result

    state["debate_results"] = all_debate_outputs
    return state


workflow.add_node("feasible_good_node", feasible_good_node)
workflow.add_node("feasible_bad_node", feasible_bad_node)
workflow.add_node("feasible_judge_node", feasible_judge_node)
workflow.add_node("innovation_good_node", innovation_good_node)
workflow.add_node("innovation_bad_node", innovation_bad_node)
workflow.add_node("innovation_judge_node", innovation_judge_node)
workflow.add_node("debate_controller", debate_controller)

workflow.add_edge("interdisciplinary_msg_clear_node", "debate_controller")
workflow.add_edge("debate_controller", "final_analyst_node")
workflow.add_edge("feasible_judge_node", "final_analyst_node")
workflow.add_edge("innovation_judge_node", "final_analyst_node")


# === Stage 3 setup ===
final_analyst_node = create_final_analyst_agent(deep_think_llm)
completeness_checker_node = create_completeness_checker_agent(deep_think_llm)
feedback_analysis_node = create_feedback_analysis_agent(deep_think_llm)
generator_node = create_generator_agent(deep_think_llm)

workflow.add_node("final_analyst_node", final_analyst_node)
workflow.add_node("completeness_checker_node", completeness_checker_node)
workflow.add_node("feedback_analysis_node", feedback_analysis_node)
workflow.add_node("generator_node", generator_node)

def _route_after_completeness(state: AgentState) -> str:
    recommendation = state.get("completeness_recommendation")
    if recommendation == "complete":
        return "generate"
    return "human_review"

def human_review_node(state: AgentState) -> AgentState:
    """在完备性不通过时触发人类审核并收集反馈。"""
    recommendation = state.get("completeness_recommendation")
    if recommendation is None:
        logger.info("缺少完备性检查结果，重新执行完备性检查")
        state = completeness_checker_node(state)
        recommendation = state.get("completeness_recommendation", "need_human_review")
    else:
        recommendation = recommendation or "need_human_review"
    if recommendation == "complete":
        logger.info("完备性检查通过，跳过人工审核直达生成阶段")
        state["skip_human_review"] = True
        return state

    logger.info("完备性检查未通过，进入人类审核流程")
    state["skip_human_review"] = False

    review_payload = {
        "task": "请审查项目评估分析并提供反馈意见",
        "analysis_summary": state.get("final_analysis_summary", ""),
        "completeness_issues": state.get("completeness_check_result", {}),
        "instructions": "请提供您的反馈意见。如果分析满足要求，请输入'approved'。如果需要改进，请详细说明需要改进的方面。",
    }

    result = interrupt(review_payload)

    if isinstance(result, dict):
        state["human_feedback"] = result.get("feedback", "")
    elif isinstance(result, str):
        state["human_feedback"] = result

    return state


def _route_after_human_review(state: AgentState) -> str:
    skip_human_review = state.get("skip_human_review", False)
    if skip_human_review:
        logger.info("人工审核被跳过，直接生成最终报告")
        return "generate"

    human_feedback = state.get("human_feedback", "")
    if human_feedback.strip():
        logger.info("收到人类反馈，进入反馈分析阶段")
        return "feedback_analysis"

    logger.info("未收到人类反馈，默认仍进入反馈分析以避免停滞")
    return "feedback_analysis"


def _route_after_feedback(state: AgentState) -> str:
    decision = state.get("feedback_routing_decision") or "generate"
    valid_routes = {
        "academic_analysis",
        "future_influence",
        "interdisciplinary",
        "debate",
        "generate",
    }
    return decision if decision in valid_routes else "generate"


workflow.add_node("human_review_node", human_review_node)

workflow.add_edge("final_analyst_node", "completeness_checker_node")
workflow.add_conditional_edges(
    "completeness_checker_node",
    _route_after_completeness,
    {
        "generate": "generator_node",
        "human_review": "human_review_node",
    },
)

workflow.add_conditional_edges(
    "human_review_node",
    _route_after_human_review,
    {
        "generate": "generator_node",
        "feedback_analysis": "feedback_analysis_node",
    },
)

workflow.add_conditional_edges(
    "feedback_analysis_node",
    _route_after_feedback,
    {
        "academic_analysis": "academic_analysis_node",
        "future_influence": "future_influence_node",
        "interdisciplinary": "interdisciplinary_node",
        "debate": "debate_controller",
        "generate": "generator_node",
    },
)

workflow.add_edge("generator_node", END)


checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


async def main():
    state: AgentState = {
        "messages": [{"role": "user", "content": "帮我分析这篇文章"}],
        "filepath": "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/提交版本.pdf",
        "research_topic": ["新颖性", "可行性"],
        "intention_decision": "",
        "research_structure": "",
        "research_person_info": "",
        "research_project_team_info": "",
        "research_basic_info": "",
        "research_project_apply_info": "",
        "research_report_body_summary": "",
        "weight_distribution": {},
        "academic_analysis_report": "",
        "academic_analysis_limit": 10,
        "academic_analysis_count": 0,
        "social_analysis_report": "",
        "future_influence_report": "",
        "future_influence_limit": 5,
        "future_influence_count": 0,
        "interdisciplinary_results": [],
        "current_discipline": "",
        "debate_results": {},
        "final_analysis_summary": "",
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
        "final_report": "",
    }

    config = {
        "configurable": {"thread_id": "workflow_full_test_thread"},
        "recursion_limit": 60,
    }

    result = None
    async for chunk in graph.astream(state, config=config):
        print(f"收到 chunk: {chunk}")
        result = chunk

    print("\n\n==== 工作流最终输出 ====")
    print(result)

    if result and "__interrupt__" in result and result["__interrupt__"]:
        interrupt_info = result["__interrupt__"][0]
        print("🛑 图在人类审核点中断，等待输入...")
        print(f"中断信息: {interrupt_info.value}")
        human_input = input("请输入人类反馈（例如: approved）: ")

        final_state = None
        async for chunk in graph.astream(
            Command(resume={"feedback": human_input}),
            config=config,
        ):
            print(f"恢复执行 chunk: {chunk}")
            final_state = chunk

        print("✅ 工作流执行完成")
        if final_state:
            print(final_state)


if __name__ == "__main__":
    asyncio.run(main())


