from typing import Dict, Any,Optional, final
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import F
from langgraph.types import Send
from proposalAgent.agents import *
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

from proposalAgent.agents.stage2.debate.feasible.feasible_bad import create_feasible_bad_agent
from proposalAgent.agents.stage2.debate.feasible.feasible_good import create_feasible_good_agent
from proposalAgent.agents.stage2.debate.feasible.feasible_manager import create_feasible_manager
from proposalAgent.agents.stage2.debate.innovation.innovation_bad import create_innovation_bad_agent
from proposalAgent.agents.stage2.debate.innovation.innovation_good import create_innovation_good_agent
from proposalAgent.agents.stage2.debate.innovation.innovation_manager import create_innovation_manager

from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.reflection_agent import create_reflection_agent
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.agents.stage3.completeness_checker import create_completeness_checker_agent
from proposalAgent.agents.stage3.generator import create_generator_agent
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
        intention_node = create_intention_agent(self.quick_thinking_llm)
        output_node = create_output_node()
        
        
        # stage 1
        structure_node = create_structure_node()
        planning_node = create_schedule_agent(self.deep_think_llm)

        # stage 2
        ## 信息收集部分
        academic_analysis_node = create_academic_agent(self.quick_thinking_llm, toolkit=self.toolkit, memory=self.academic_memory)
        academic_tool_exc_node = self.tool_nodes['academic']
        
        # 使用一个简单的消息清除函数替代
        def create_msg_delete():
            def msg_delete(state):
                return {"messages": []}
            return msg_delete
        
        academic_msg_clear_node = create_msg_delete()
        
        # social_analysis_node = create_social_analysis_agent(self.quick_thinking_llm, self.toolkit, self.impact_memory)  # 暂时使用impact_memory
        def placeholder_social_analysis_node(state):
            return {"messages": [], "social_analysis_report": "社会分析模块暂未实现"}
        social_analysis_node = placeholder_social_analysis_node
        
        social_tool_exc_node = self.tool_nodes['social']
        social_msg_clear_node = create_msg_delete()
        
        future_influence_node = create_future_influence_agent(self.deep_think_llm, self.toolkit, self.future_influence_memory)
        future_influence_tool_exc_node = self.tool_nodes['influence']
        future_influence_msg_clear_node = create_msg_delete()
        
        interdisciplinary_node = create_interdis_agent(self.deep_think_llm, self.toolkit)
        interdisciplinary_tool_exc_node = self.tool_nodes['interdisciplinary']
        interdisciplinary_msg_clear_node = create_msg_delete()
        
        # ## 辩论部分
        # # 可行性辩论
        feasible_good_node = create_feasible_good_agent(self.deep_think_llm, self.toolkit, self.feasibility_memory)
        feasible_good_tool_exc_node = self.tool_nodes['feasibility']
        feasible_good_msg_clear_node = create_msg_delete()
        
        feasible_bad_node = create_feasible_bad_agent(self.deep_think_llm, self.toolkit, self.feasibility_memory)
        feasible_bad_tool_exc_node = self.tool_nodes['feasibility']
        feasible_bad_msg_clear_node = create_msg_delete()
        # 可行性总结
        feasible_judge_node = create_feasible_manager(self.deep_think_llm, self.feasibility_memory)
        
        
        # 创新性辩论
        innovation_good_node = create_innovation_good_agent(self.deep_think_llm, self.toolkit, self.innovation_memory)
        innovation_good_tool_exc_node = self.tool_nodes['innovation']
        innovation_good_msg_clear_node = create_msg_delete()
        
        innovation_bad_node = create_innovation_bad_agent(self.deep_think_llm, self.toolkit, self.innovation_memory)
        innovation_bad_tool_exc_node = self.tool_nodes['innovation']
        innovation_bad_msg_clear_node = create_msg_delete()
        # 创新性总结
        innovation_judge_node = create_innovation_manager(self.deep_think_llm, self.innovation_memory)
        
        
        ## stage 3
        final_analyst_node = create_final_analyst_agent(self.deep_think_llm)
        completeness_checker_node = create_completeness_checker_agent(self.deep_think_llm)
        generator_node = create_generator_agent(self.deep_think_llm)
        
        
        
        
    
        # 创建StateGraph实例
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
        
        ## 辩论节点
        ## 可行性辩论节点
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
        workflow.add_node("completeness_checker_node",completeness_checker_node)
        workflow.add_node("generator_node",generator_node)
        
        ## edges
        workflow.add_edge(START,"intention_node")
        workflow.add_conditional_edges("intention_node",self.conditional_logic.should_output,{
            "output_node":"output_node",
            "structure_node":"structure_node"
        })
        
        workflow.add_edge("output_node",END)
        workflow.add_edge("structure_node","planning_node")
        workflow.add_edge("planning_node","academic_analysis_node")
        workflow.add_conditional_edges("academic_analysis_node",self.conditional_logic.should_continue_academic_analysis,{
            "tools_academic":"academic_tool_exc_node",
            "msg_clear_academic":"academic_msg_clear_node",
            "final_analyst_node":"final_analyst_node"
        }) 
        workflow.add_edge("tools_academic","academic_analysis_node")
        workflow.add_edge("academic_analysis_node","social_analysis_node")
        workflow.add_conditional_edges("social_analysis_node",self.conditional_logic.should_continue_social_analysis,{
            "tools_social":"social_analysis_tool_exc_node",
            "msg_clear_social":"social_analysis_msg_clear_node",
            "final_analyst_node":"final_analyst_node"
        })
        workflow.add_edge("tools_social","social_analysis_node")
        workflow.add_edge("social_analysis_node","future_influence_node")

        workflow.add_conditional_edges("future_influence_node",self.conditional_logic.should_continue_future_influence,{
            "tools_future_influence":"future_influence_tool_exc_node",
            "msg_clear_future_influence":"future_influence_msg_clear_node",
            "final_analyst_node":"final_analyst_node"
        })
        
        workflow.add_edge("tools_future_influence","future_influence_node")
        workflow.add_edge("future_influence_node","interdisciplinary_node")
        
        workflow.add_conditional_edges("interdisciplinary_node",self.conditional_logic.should_continue_interdisciplinary,{
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

        # 2. 创新性辩论子图 remove tool
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
            all_debate_outputs = {}
            for discipline in disciplines:
                input_state = state.copy()
                from langchain_core.messages import SystemMessage
                input_state["messages"] = state["messages"] + [SystemMessage(content=f"Starting debates for discipline: {discipline}")]
                input_state["current_discipline"] = discipline

                # 并行运行可行性和创新性辩论
                f_task = compiled_feasibility_debate_graph.ainvoke(input_state)
                i_task = compiled_innovation_debate_graph.ainvoke(input_state)
                results = await asyncio.gather(f_task, i_task)
                all_debate_outputs[discipline]= results
            
            state['debate_results'] = all_debate_outputs
            return state

        workflow.add_node("debate_controller", debate_controller)
        workflow.add_edge("interdisciplinary_node", "debate_controller")
        workflow.add_edge("debate_controller", "final_analyst_node")
        
        # 路由函数：根据完备性检查和人类反馈情况决定下一步
        def _route_after_human_review(state: AgentState) -> str:
            """
            根据完备性检查结果和是否有人类反馈来决定路由
            """
            # 检查是否跳过人类审核
            skip_human_review = state.get("skip_human_review", False)
            if skip_human_review:
                print("完备性检查通过，直接生成报告")
                return "generate"
            
            # 检查是否有人类反馈
            human_feedback = state.get("human_feedback")
            if human_feedback and human_feedback.strip():
                print("检测到人类反馈，进行反馈分析")
                return "feedback_analysis"
            else:
                print("没有人类反馈，可能需要等待用户输入")
                # 这种情况下图形应该已经中断等待输入
                # 如果到这里说明有问题，默认分析反馈
                return "feedback_analysis"
        

        # stage 3
        """
        stage3做的事情：
        1、能够根据分析结果动态更新记忆
        2、将分析结果使用human in the loop引入人类评审（不一定要，如果final analyst觉得置信度高的话可以直接走到生成最终报表，但是如何评价置信度我还没想好），
        3、引入评审之后如果人类评审没问题就生成，有问题的话就根据人类的评价，分析出来问题出现在哪里，更新他的记忆，并且重新执行那一部分节点，然后再输出报表。
        """
    #    reflection_node = create_reflection_agent(self.deep_think_llm)
        feedback_analysis_node = create_feedback_analysis_agent(self.deep_think_llm)

        # Enhanced human review node that includes completeness checking
        def human_review_node(state: AgentState) -> AgentState:
            """
            增强的人类审核节点，首先进行完备性检查，然后根据结果决定是否需要人类输入。
            如果完备性检查通过，将跳过人类审核直接生成报告。
            如果未通过，则等待人类反馈。
            """
            # 首先进行完备性检查
            print("=== 执行完备性检查 ===")
            state = completeness_checker_node(state)
            
            # 检查完备性结果
            completeness_recommendation = state.get("completeness_recommendation", "need_human_review")
            
            if completeness_recommendation == "complete":
                print("完备性检查通过，将直接生成报告")
                # 设置跳过人类审核的标记
                state["skip_human_review"] = True
                return state
            else:
                print("完备性检查未通过，需要人类审核")
                # 需要人类审核 - 图形将在此中断等待人类输入
                state["skip_human_review"] = False
                return state

        # 2. Add the new nodes to the workflow
     #  workflow.add_node("reflection_node", reflection_node)
        workflow.add_node("human_review_node", human_review_node)
        workflow.add_node("feedback_analysis_node", feedback_analysis_node)


        # 从最终分析到人类审核节点（包含完备性检查）
        workflow.add_edge("final_analyst_node", "human_review_node")

        # 根据完备性检查和人类反馈情况进行条件路由
        workflow.add_conditional_edges(
            "human_review_node",
            _route_after_human_review,
            {
                "generate": "generator_node",  # 完备性检查通过，直接生成
                "feedback_analysis": "feedback_analysis_node"  # 需要分析人类反馈
            }
        )

        
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


        # 创建一个条件中断函数
        def should_interrupt_for_human_review(state: AgentState) -> bool:
            """
            只有当完备性检查未通过且需要人类审核时才中断
            """
            completeness_recommendation = state.get("completeness_recommendation", "need_human_review")
            skip_human_review = state.get("skip_human_review", False)
            
            # 如果完备性检查通过，不需要中断
            if skip_human_review or completeness_recommendation == "complete":
                return False
            
            # 如果已经有人类反馈，不需要再次中断
            human_feedback = state.get("human_feedback")
            if human_feedback and human_feedback.strip():
                return False
                
            # 需要人类审核
            return True
        
        # 编译图形，设置条件中断
        # 注意：langgraph的interrupt_before不支持条件中断，
        # 我们改为在human_review_node内部通过检查来决定是否需要等待输入
        return workflow.compile(interrupt_before=["human_review_node"])