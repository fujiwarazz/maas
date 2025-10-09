import asyncio
import copy
from typing import Dict, Any, Optional
from logging import getLogger
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.utils.agent_utils import Toolkit

# from proposalAgent.graphs import workflow
from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.output import create_output_node
from proposalAgent.agents.stage1.structure import create_structure_node
from proposalAgent.agents.stage1.schedule import create_schedule_agent

from proposalAgent.agents.stage2.academic import create_academic_agent

# from proposalAgent.agents.stage2.social import create_social_analysis_agent
from proposalAgent.agents.stage2.future_influence import create_future_influence_agent
from proposalAgent.agents.stage2.interdis import create_interdis_agent

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

from proposalAgent.agents.stage3.feedback_analysis_agent import (
    create_feedback_analysis_agent,
)
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.agents.stage3.completeness_checker import (
    create_completeness_checker_agent,
)
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.agents.utils.agent_utils import create_msg_delete
from .conditional_logic import ConditionalLogic

logger = getLogger("GraphSetup")


class GraphSetup:
    """
    处理智能体工作流图的设置和配置。
    这个类是构建器，将所有独立的智能体、工具和逻辑组装成一个可执行的图。
    """

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_think_llm: Any,
        structure_llm: Optional[Any],
        toolkit: Toolkit,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        feasible_good_memory:EmbeddingMemory,
        feasible_bad_memory:EmbeddingMemory,
        feasible_manager_memory:EmbeddingMemory,
        innovation_good_memory:EmbeddingMemory,
        innovation_bad_memory:EmbeddingMemory,
        innovation_manager_memory:EmbeddingMemory,
       
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_think_llm = deep_think_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.structure_llm = structure_llm
        self.conditional_logic = conditional_logic
        self.feasible_good_memory=feasible_good_memory
        self.feasible_bad_memory=feasible_bad_memory
        self.feasible_manager_memory=feasible_manager_memory
        self.innovation_good_memory=innovation_good_memory
        self.innovation_bad_memory=innovation_bad_memory
        self.innovation_manager_memory=innovation_manager_memory

    def setup_graph(self):
        """
        构建并返回工作流图。
        这个方法将所有组件（智能体、工具、逻辑）整合到一个StateGraph中，定义了它们的交互规则。
        """
        # Stage 1 setup
        intention_node = create_intention_agent(self.quick_thinking_llm)
        output_node = create_output_node(self.quick_thinking_llm)
        structure_node = create_structure_node()
        planning_node = create_schedule_agent(self.deep_think_llm)
        planning_clear_node = create_msg_delete()

        # Stage 2 setup
        academic_analysis_node = create_academic_agent(
            self.quick_thinking_llm, toolkit=self.toolkit
        )
        academic_tool_exc_node = self.tool_nodes["academic"]
        academic_msg_clear_node = create_msg_delete()

        future_influence_node = create_future_influence_agent(
            self.deep_think_llm, self.toolkit
        )
        future_influence_tool_exc_node = self.tool_nodes["influence"]
        future_influence_msg_clear_node = create_msg_delete()

        interdisciplinary_node = create_interdis_agent(
            self.deep_think_llm, self.toolkit
        )
        interdisciplinary_tool_exc_node = self.tool_nodes["interdisciplinary"]
        interdisciplinary_msg_clear_node = create_msg_delete()

        feasible_good_node = create_feasible_good_agent(
            self.deep_think_llm, self.toolkit, self.feasible_good_memory
        )
        feasible_bad_node = create_feasible_bad_agent(
            self.deep_think_llm, self.toolkit, self.feasible_bad_memory
        )
        feasible_manager_node = create_feasible_manager(
            self.deep_think_llm, self.feasible_manager_memory
        )
        innovation_good_node = create_innovation_good_agent(
            self.deep_think_llm, self.toolkit, self.innovation_good_memory
        )
        innovation_bad_node = create_innovation_bad_agent(
            self.deep_think_llm, self.toolkit, self.innovation_bad_memory
        )
        innovation_manager_node = create_innovation_manager(
            self.deep_think_llm, self.innovation_manager_memory
        )

        if getattr(self.conditional_logic, "max_debate_rounds", 1) < 3:
            self.conditional_logic.max_debate_rounds = 3

        # Stage 3 setup
        final_analyst_node = create_final_analyst_agent(self.deep_think_llm)
        completeness_checker_node = create_completeness_checker_agent(
            self.deep_think_llm
        )
        feedback_analysis_node = create_feedback_analysis_agent(self.deep_think_llm)

        # Academic subgraph
        academic_workflow = StateGraph(AgentState)
        academic_workflow.add_node("academic_analysis_node", academic_analysis_node)
        academic_workflow.add_node("academic_tool_exc_node", academic_tool_exc_node)
        academic_workflow.add_node("academic_msg_clear_node", academic_msg_clear_node)
        academic_workflow.add_edge(START, "academic_analysis_node")
        academic_workflow.add_conditional_edges(
            "academic_analysis_node",
            self.conditional_logic.should_continue_academic_analysis,
            {
                "tools_academic": "academic_tool_exc_node",
                "msg_clear_academic": "academic_msg_clear_node",
            },
        )
        academic_workflow.add_edge("academic_tool_exc_node", "academic_analysis_node")
        academic_workflow.add_edge("academic_msg_clear_node", END)
        compiled_academic_graph = academic_workflow.compile()

        # Future influence subgraph
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
            self.conditional_logic.should_continue_future_influence,
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

        async def rerun_academic_subgraph(state: AgentState):
            subgraph_state = copy.deepcopy(state)
            result = await compiled_academic_graph.ainvoke(subgraph_state)

            merged_state = copy.deepcopy(state)
            for key, value in result.items():
                if key == "messages":
                    continue
                merged_state[key] = value

            merged_state["messages"] = state.get("messages")
            return merged_state

        async def rerun_future_influence_subgraph(state: AgentState):
            subgraph_state = copy.deepcopy(state)
            result = await compiled_future_influence_graph.ainvoke(subgraph_state)

            merged_state = copy.deepcopy(state)
            for key, value in result.items():
                if key == "messages":
                    continue
                merged_state[key] = value

            merged_state["messages"] = state.get("messages")
            return merged_state

        # Feasibility debate subgraph
        feasibility_debate_workflow = StateGraph(AgentState)
        feasibility_debate_workflow.add_node("feasible_good_node", feasible_good_node)
        feasibility_debate_workflow.add_node("feasible_bad_node", feasible_bad_node)
        feasibility_debate_workflow.add_node("feasible_judge_node", feasible_manager_node)
        feasibility_debate_workflow.add_edge(START, "feasible_good_node")
        feasibility_debate_workflow.add_conditional_edges(
            "feasible_good_node",
            self.conditional_logic.should_continue_feasibility,
            {
                "good": "feasible_good_node",
                "bad": "feasible_bad_node",
                "judge": "feasible_judge_node",
            },
        )
        feasibility_debate_workflow.add_conditional_edges(
            "feasible_bad_node",
            self.conditional_logic.should_continue_feasibility,
            {
                "good": "feasible_good_node",
                "bad": "feasible_bad_node",
                "judge": "feasible_judge_node",
            },
        )
        feasibility_debate_workflow.add_edge("feasible_judge_node", END)
        compiled_feasibility_debate_graph = feasibility_debate_workflow.compile()

        # Innovation debate subgraph
        innovation_debate_workflow = StateGraph(AgentState)
        innovation_debate_workflow.add_node("innovation_good_node", innovation_good_node)
        innovation_debate_workflow.add_node("innovation_bad_node", innovation_bad_node)
        innovation_debate_workflow.add_node("innovation_judge_node", innovation_manager_node)
        innovation_debate_workflow.add_edge(START, "innovation_good_node")
        innovation_debate_workflow.add_conditional_edges(
            "innovation_good_node",
            self.conditional_logic.should_continue_innovation,
            {
                "good": "innovation_good_node",
                "bad": "innovation_bad_node",
                "judge": "innovation_judge_node",
            },
        )
        innovation_debate_workflow.add_conditional_edges(
            "innovation_bad_node",
            self.conditional_logic.should_continue_innovation,
            {
                "good": "innovation_good_node",
                "bad": "innovation_bad_node",
                "judge": "innovation_judge_node",
            },
        )
        innovation_debate_workflow.add_edge("innovation_judge_node", END)
        compiled_innovation_debate_graph = innovation_debate_workflow.compile()

        async def debate_controller(state: AgentState):
            disciplines = state.get("interdisciplinary_results", [])
            all_debate_outputs = {}
            tasks = []

            async def single_task(discipline_name: str):
                input_state = state.copy()
                input_state["messages"] = state.get("messages", []) + [
                    SystemMessage(
                        content=f"Starting debates for discipline: {discipline_name}"
                    )
                ]
                input_state["current_discipline"] = discipline_name

                f_task = compiled_feasibility_debate_graph.ainvoke(input_state)
                i_task = compiled_innovation_debate_graph.ainvoke(input_state)
                results = await asyncio.gather(f_task, i_task)

                output_result_merge = {}
                output_result_merge["可行性"] = (
                    results[0]
                    .get("debate_results", {})
                    .get(discipline_name, {})
                    .get("可行性", {})
                )
                output_result_merge["创新性"] = (
                    results[1]
                    .get("debate_results", {})
                    .get(discipline_name, {})
                    .get("创新性", {})
                )

                return discipline_name, output_result_merge

            for discipline in disciplines:
                tasks.append(single_task(discipline))

            results = await asyncio.gather(*tasks) if tasks else []
            for discipline_name, result in results:
                all_debate_outputs[discipline_name] = result

            state["debate_results"] = all_debate_outputs
            return state

        # Main workflow construction
        workflow = StateGraph(AgentState)
        workflow.add_node("intention_node", intention_node)
        workflow.add_node("output_node", output_node)
        workflow.add_node("structure_node", structure_node)
        workflow.add_node("planning_node", planning_node)
        workflow.add_node("planning_clear_node", planning_clear_node)

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
        workflow.add_node("academic_subgraph_node", rerun_academic_subgraph)
        workflow.add_node(
            "future_influence_subgraph_node", rerun_future_influence_subgraph
        )

        workflow.add_node("debate_controller", debate_controller)

        workflow.add_node("final_analyst_node", final_analyst_node)
        workflow.add_node("completeness_checker_node", completeness_checker_node)
        workflow.add_node("feedback_analysis_node", feedback_analysis_node)
        workflow.add_node("feasible_good_node", feasible_good_node)
        workflow.add_node("feasible_bad_node", feasible_bad_node)
        workflow.add_node("feasible_judge_node", feasible_manager_node)
        workflow.add_node("innovation_good_node", innovation_good_node)
        workflow.add_node("innovation_bad_node", innovation_bad_node)
        workflow.add_node("innovation_judge_node", innovation_manager_node)

        def _route_after_completeness(state: AgentState) -> str:
            recommendation = state.get("completeness_recommendation")
            if recommendation == "complete":
                return "generate"
            return "human_review"

        def human_review_node(state: AgentState) -> AgentState:
            recommendation = state.get("completeness_recommendation")

            if recommendation is None:
                logger.info("缺少完备性检查结果，重新执行完备性检查")
                state_updated = completeness_checker_node(state)
                recommendation_local = state_updated.get(
                    "completeness_recommendation", "need_human_review"
                )
                state = state_updated
            else:
                recommendation_local = recommendation or "need_human_review"

            if recommendation_local == "complete":
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

        # Stage 1 edges
        workflow.add_edge(START, "intention_node")
        workflow.add_conditional_edges(
            "intention_node",
            self.conditional_logic.should_output,
            {
                "output_node": "output_node",
                "structure_node": "structure_node",
            },
        )
        workflow.add_edge("output_node", END)
        workflow.add_edge("structure_node", "planning_node")
        workflow.add_edge("planning_node", "planning_clear_node")

        # Stage 2 edges
        workflow.add_edge("planning_clear_node", "stage2_parallel_node")
        workflow.add_edge("stage2_parallel_node", "interdisciplinary_node")
        workflow.add_conditional_edges(
            "interdisciplinary_node",
            self.conditional_logic.should_continue_interdisciplinary,
            {
                "tools_interdisciplinary": "interdisciplinary_tool_exc_node",
                "msg_clear_interdisciplinary": "interdisciplinary_msg_clear_node",
            },
        )
        workflow.add_edge(
            "interdisciplinary_tool_exc_node", "interdisciplinary_node"
        )
        workflow.add_edge("interdisciplinary_msg_clear_node", "debate_controller")

        workflow.add_edge("debate_controller", "final_analyst_node")

        # Stage 3 edges
        workflow.add_edge("final_analyst_node", "completeness_checker_node")
        workflow.add_conditional_edges(
            "completeness_checker_node",
            _route_after_completeness,
            {
                "generate": END,
                "human_review": "human_review_node",
            },
        )

        workflow.add_conditional_edges(
            "human_review_node",
            _route_after_human_review,
            {
                "generate": END,
                "feedback_analysis": "feedback_analysis_node",
            },
        )

        workflow.add_conditional_edges(
            "feedback_analysis_node",
            _route_after_feedback,
            {
                "academic_analysis": "academic_subgraph_node",
                "future_influence": "future_influence_subgraph_node",
                "interdisciplinary": "interdisciplinary_node",
                "debate": "debate_controller",
                "generate": END,
            },
        )

        checkpointer = MemorySaver()

        return workflow.compile(checkpointer=checkpointer)
