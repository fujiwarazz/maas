from typing import Dict, Any,Optional, final
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import F
from langgraph.types import Send
from proposalAgent.agents import *
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.utils.agent_utils import Toolkit,create_msg_delete
from proposalAgent.graphs import workflow
from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.reflection_agent import create_reflection_agent
from tools import *
from .conditional_logic import ConditionalLogic
import asyncio

class GraphSetup:
    """
    处理智能体工作流图的设置和配置。
    这个类是构建器，将所有独立的智能体、工具和逻辑组装成一个可执行的图。
    """
    
    def __init__(
        self,
        quick_thinking_llm:Any,
        deep_think_llm:Any,
        structure_llm:Optional[Any],
        toolkit:Toolkit,
        tool_nodes:Dict[str,ToolNode],
        conditional_logic:ConditionalLogic,
        impact_memory:Any,
        planning_memory:Any,
        future_influence_memory:Any,
        risk_memory:Any,
        interdisciplinary_memory:Any,
        academic_memory:Any,
        feasibility_memory:Any,
        innovation_memory:Any,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_think_llm = deep_think_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.structure_llm = structure_llm
        self.conditional_logic = conditional_logic
        self.impact_memory = impact_memory
        self.future_influence_memory = future_influence_memory
        self.risk_memory = risk_memory
        self.interdisciplinary_memory = interdisciplinary_memory
        self.academic_memory = academic_memory
        self.feasibility_memory = feasibility_memory
        self.innovation_memory = innovation_memory
        self.planning_memory = planning_memory
    
    def setup_graph(self):
        """
        构建并返回工作流图。
        这个方法将所有组件（智能体、工具、逻辑）整合到一个StateGraph中，定义了它们的交互规则。
        """
        # stage 0: normal for intention recognization
        intention_node = create_intention_agent(self.quick_thinking_llm,tools = self.toolkit.get_tools['intention'])
        output_node = create_output_agent(self.quick_thinking_llm,tools = self.toolkit.get_tools['output'])
        
        
        # stage 1
        structure_node = create_structure_node(self.structure_llm,tools=self.toolkit.get_tools['output'])
        planning_node = create_planning_agent(self.deep_think_llm,self.planning_memory)

        # stage 2
        ## 信息收集部分
        academic_analysis_node = create_academic_analysis_agent(self.quick_thinking_llm,self.academic_analysis_memory,tools=self.toolkit.get_tools['academic'])
        academic_tool_exc_node = self.tool_nodes['academic']
        academic_msg_clear_node = create_msg_delete()
        
        social_analysis_node = create_social_analysis_agent(self.quick_thinking_llm,self.social_analysis_memory,tools=self.toolkit.get_tools['social'])
        social_tool_exc_node = self.tool_nodes['social']
        social_msg_clear_node = create_msg_delete()
        
        future_influence_node = create_future_influence_agent(self.deep_think_llm,self.future_influence_memory,self.toolkit.get_tools['influence'])
        future_influence_tool_exc_node = self.tool_nodes['influence']
        future_influence_msg_clear_node = create_msg_delete()
        
        interdisciplinary_node = create_interdisciplinary_agent(self.deep_think_llm,self.interdisciplinary_memory,self.toolkit.get_tools['interdisciplinary'])
        interdisciplinary_tool_exc_node = self.tool_nodes['interdisciplinary']
        interdisciplinary_msg_clear_node = create_msg_delete()
        
        ## 辩论部分
        # 可行性辩论
        feasible_good_node = create_feasible_good_agent(self.deep_think_llm)
        feasible_good_tool_exc_node = self.tool_nodes['feasibility']
        feasible_good_msg_clear_node = create_msg_delete()
        
        feasible_bad_node = create_feasible_bad_agent(self.deep_think_llm,self)
        feasible_bad_tool_exc_node = self.tool_nodes['feasibility']
        feasible_bad_msg_clear_node = create_msg_delete()
        # 可行性总结
        feasible_judge_node = create_feasibility_judge_agent(self.deep_think_llm,self.feasibility_memory)
        
        
        # 创新性辩论
        innovation_good_node = create_innovation_good_agent(self.deep_think_llm)
        innovation_good_tool_exc_node = self.tool_nodes['innovation']
        innovation_good_msg_clear_node = create_msg_delete()
        
        innovation_bad_node = create_innovation_bad_agent(self.deep_think_llm)
        innovation_bad_tool_exc_node = self.tool_nodes['innovation']
        innovation_bad_msg_clear_node = create_msg_delete()
        # 创新性总结
        innovation_judge_node = create_innovation_judge_agent(self.deep_think_llm,self.innovation_memory)
        
        
        ## stage 3
        final_analyst_node = create_final_analyst_agent(self.deep_think_llm)
        generator_node = create_generator_agent(self.deep_think_llm)
        
        
        workflow = StateGraph(AgentState)
        ## stage 1 nodes
        workflow.add_node("intention_node",intention_node)
        workflow.add_node("planning_node",planning_node)
        workflow.add_node("output_node",output_node)
        workflow.add_node("structure_node",structure_node)
        
        ## stage 2 nodes
        ### 信息收集节点
        ### 学术分析节点
        workflow.add_node("academic_analysis_node",academic_analysis_node)
        workflow.add_node("academic_analysis_tool_exc_node",academic_tool_exc_node)
        workflow.add_node("academic_analysis_msg_clear_node",academic_msg_clear_node)
        
        ### 社会分析节点
        workflow.add_node("social_analysis_node",social_analysis_node)
        workflow.add_node("social_analysis_tool_exc_node",social_tool_exc_node)
        workflow.add_node("social_analysis_msg_clear_node",social_msg_clear_node)
        
        ### 未来影响分析节点
        workflow.add_node("future_influence_node",future_influence_node)
        workflow.add_node("future_influence_tool_exc_node",future_influence_tool_exc_node)
        workflow.add_node("future_influence_msg_clear_node",future_influence_msg_clear_node)
        
        ### 跨学科分析节点
        workflow.add_node("interdisciplinary_node",interdisciplinary_node)
        workflow.add_node("interdisciplinary_tool_exc_node",interdisciplinary_tool_exc_node)
        workflow.add_node("interdisciplinary_msg_clear_node",interdisciplinary_msg_clear_node)
        
        ### 辩论节点
        ### 可行性辩论节点
        workflow.add_node("feasible_good_node",feasible_good_node)
        workflow.add_node("feasible_good_tool_exc_node",feasible_good_tool_exc_node)
        workflow.add_node("feasible_good_msg_clear_node",feasible_good_msg_clear_node)
    
        workflow.add_node("feasible_bad_node",feasible_bad_node)
        workflow.add_node("feasible_bad_tool_exc_node",feasible_bad_tool_exc_node)
        workflow.add_node("feasible_bad_msg_clear_node",feasible_bad_msg_clear_node)
        workflow.add_node("feasible_judge_node",feasible_judge_node)
        
        ### 创新性辩论节点
        workflow.add_node("innovation_good_node",innovation_good_node)
        workflow.add_node("innovation_good_tool_exc_node",innovation_good_tool_exc_node)
        workflow.add_node("innovation_good_msg_clear_node",innovation_good_msg_clear_node)
        
        workflow.add_node("innovation_bad_node",innovation_bad_node)
        workflow.add_node("innovation_bad_tool_exc_node",innovation_bad_tool_exc_node)
        workflow.add_node("innovation_bad_msg_clear_node",innovation_bad_msg_clear_node)
        workflow.add_node("innovation_judge_node",innovation_judge_node)
        
        ## stage 3 nodes
        workflow.add_node("final_analyst_node",final_analyst_node)
        workflow.add_node("generator_node",generator_node)
        
        ## edges
        workflow.add_edge(START,"intention_node")
        workflow.add_conditional_edges("intention_node",should_output,{
            "output_node":output_node,
            "structure_node":structure_node
        })
        
        workflow.add_edge("output_node",END)
        workflow.add_edge("structure_node","planning_node")
        workflow.add_edge("planning_node","academic_analysis_node")
        workflow.add_conditional_edges("academic_analysis_node",should_continue_academic_analysis,{
            "tools_academic":"academic_tool_exc_node",
            "msg_clear_academic":"academic_msg_clear_node",
            "final_analyst_node":"final_analyst_node"
        }) 
        workflow.add_edge("tools_academic","academic_analysis_node")
        workflow.add_edge("academic_analysis_node","social_analysis_node")
        workflow.add_conditional_edges("social_analysis_node",should_continue_social_analysis,{
            "tools_social":"social_analysis_tool_exc_node",
            "msg_clear_social":"social_analysis_msg_clear_node",
            "final_analyst_node":"final_analyst_node"
        })
        workflow.add_edge("tools_social","social_analysis_node")
        workflow.add_edge("social_analysis_node","future_influence_node")

        workflow.add_conditional_edges("future_influence_node",should_continue_future_influence,{
            "tools_future_influence":"future_influence_tool_exc_node",
            "msg_clear_future_influence":"future_influence_msg_clear_node",
            "final_analyst_node":"final_analyst_node"

        })
        workflow.add_edge("tools_future_influence","future_influence_node")
        workflow.add_edge("future_influence_node","interdisciplinary_node")
        
        workflow.add_conditional_edges("interdisciplinary_node",should_continue_interdisciplinary,{
            "tools_interdisciplinary":"interdisciplinary_tool_exc_node",
            "msg_clear_interdisciplinary":"interdisciplinary_msg_clear_node",
            "final_analyst_node":"final_analyst_node"
        })
        workflow.add_edge("tools_interdisciplinary", "interdisciplinary_node")


        # 1. 可行性辩论子图
        feasibility_debate_workflow = StateGraph(AgentState)
        feasibility_debate_workflow.add_node("feasible_good_node", feasible_good_node)
        feasibility_debate_workflow.add_node("feasible_good_tool_exc_node", feasible_good_tool_exc_node)
        feasibility_debate_workflow.add_node("feasible_bad_node", feasible_bad_node)
        feasibility_debate_workflow.add_node("feasible_bad_tool_exc_node", feasible_bad_tool_exc_node)
        feasibility_debate_workflow.add_node("feasible_judge_node", feasible_judge_node)

        feasibility_debate_workflow.add_edge(START, "feasible_good_node")

        feasibility_debate_workflow.add_conditional_edges(
            "feasible_good_node",
            self.conditional_logic.should_continue_feasibility,
            {"continue": "feasible_bad_node", "end": "feasible_judge_node", "tools": "feasible_good_tool_exc_node"}
        )
        feasibility_debate_workflow.add_edge("feasible_good_tool_exc_node", "feasible_good_node")

        feasibility_debate_workflow.add_conditional_edges(
            "feasible_bad_node",
            self.conditional_logic.should_continue_feasibility,
            {"continue": "feasible_good_node", "end": "feasible_judge_node", "tools": "feasible_bad_tool_exc_node"}
        )
        feasibility_debate_workflow.add_edge("feasible_bad_tool_exc_node", "feasible_bad_node")
        feasibility_debate_workflow.add_edge("feasible_judge_node", END)
        compiled_feasibility_debate_graph = feasibility_debate_workflow.compile()

        # 2. 创新性辩论子图
        innovation_debate_workflow = StateGraph(AgentState)
        innovation_debate_workflow.add_node("innovation_good_node", innovation_good_node)
        innovation_debate_workflow.add_node("innovation_good_tool_exc_node", innovation_good_tool_exc_node)
        innovation_debate_workflow.add_node("innovation_bad_node", innovation_bad_node)
        innovation_debate_workflow.add_node("innovation_bad_tool_exc_node", innovation_bad_tool_exc_node)
        innovation_debate_workflow.add_node("innovation_judge_node", innovation_judge_node)

        innovation_debate_workflow.add_edge(START, "innovation_good_node")

        innovation_debate_workflow.add_conditional_edges(
            "innovation_good_node",
            self.conditional_logic.should_continue_innovation,
            {"continue": "innovation_bad_node", "end": "innovation_judge_node", "tools": "innovation_good_tool_exc_node"}
        )
        innovation_debate_workflow.add_edge("innovation_good_tool_exc_node", "innovation_good_node")

        innovation_debate_workflow.add_conditional_edges(
            "innovation_bad_node",
            self.conditional_logic.should_continue_innovation,
            {"continue": "innovation_good_node", "end": "innovation_judge_node", "tools": "innovation_bad_tool_exc_node"}
        )
        innovation_debate_workflow.add_edge("innovation_bad_tool_exc_node", "innovation_bad_node")
        innovation_debate_workflow.add_edge("innovation_judge_node", END)
        compiled_innovation_debate_graph = innovation_debate_workflow.compile()

        # 3. 辩论节点
        async def debate_controller(state: AgentState):
            disciplines = state.get('interdisciplinary_results', [])
            all_debate_outputs = []
            for discipline in disciplines:
                input_state = state.copy()
                input_state["messages"] = state["messages"] + [("system", f"Starting debates for discipline: {discipline}")]
                input_state["current_discipline"] = discipline

                # 并行运行可行性和创新性辩论
                f_task = compiled_feasibility_debate_graph.ainvoke(input_state)
                i_task = compiled_innovation_debate_graph.ainvoke(input_state)
                results = await asyncio.gather(f_task, i_task)
                all_debate_outputs.append({discipline: results})
            
            state['debate_results'] = all_debate_outputs
            return state

        workflow.add_node("debate_controller", debate_controller)
        workflow.add_edge("interdisciplinary_node", "debate_controller")
        workflow.add_edge("debate_controller", "final_analyst_node")
        

        # stage 3
        """
        stage3做的事情：
        1、能够根据分析结果动态更新记忆
        2、将分析结果使用human in the loop引入人类评审（不一定要，如果final analyst觉得置信度高的话可以直接走到生成最终报表，但是如何评价置信度我还没想好），
        3、引入评审之后如果人类评审没问题就生成，有问题的话就根据人类的评价，分析出来问题出现在哪里，更新他的记忆，并且重新执行那一部分节点，然后再输出报表。
        """
        reflection_node = create_reflection_agent(self.deep_think_llm)
        feedback_analysis_node = create_feedback_analysis_agent(self.deep_think_llm)

        # The human review node is a placeholder to allow the graph to interrupt for human input.
        def human_review_node(state: AgentState) -> AgentState:
            # The graph will be configured to interrupt before this node.
            # The application running the graph will collect human feedback
            # and resume execution.
            return state

        # 2. Add the new nodes to the workflow
        workflow.add_node("reflection_node", reflection_node)
        workflow.add_node("human_review_node", human_review_node)
        workflow.add_node("feedback_analysis_node", feedback_analysis_node)


        workflow.add_edge("final_analyst_node", "reflection_node")

        workflow.add_conditional_edges(
            "reflection_node",
            self.conditional_logic.should_request_human_review,
            {
                "generate": "generator_node",
                "review": "human_review_node"
            }
        )

        # 人类评审完直接分析
        workflow.add_edge("human_review_node", "feedback_analysis_node")

        
        # 这里是直接跳转，对应部分节点完成之后能重新走到part3的部分
        workflow.add_conditional_edges(
            "feedback_analysis_node",
            self.conditional_logic.route_after_feedback,
            {
                # Loop back to earlier stages based on feedback
                "academic_analysis": "academic_analysis_node",
                "social_analysis": "social_analysis_node",
                "future_influence": "future_influence_node",
                "interdisciplinary": "interdisciplinary_node",
                "debate": "debate_controller",
                "generate": "generator_node",
            }
        )

        workflow.add_edge("generator_node", END)


        # To enable the human-in-the-loop, you need to compile the graph
        # with an instruction to interrupt before the human_review_node.
        return workflow.compile(interrupt_before=["human_review_node"])