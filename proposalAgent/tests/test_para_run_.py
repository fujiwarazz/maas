import asyncio
import copy
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.utils.agent_utils import Toolkit
from proposalAgent.graphs.conditional_logic import ConditionalLogic

from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.output import create_output_node
from proposalAgent.agents.stage1.structure import create_structure_node
from proposalAgent.agents.stage1.schedule import create_schedule_agent

from proposalAgent.agents.stage2.academic import create_academic_agent
from proposalAgent.agents.stage2.future_influence import create_future_influence_agent
from proposalAgent.agents.stage2.interdis import create_interdis_agent

from proposalAgent.agents.stage2.debate.feasible.feasible_good import create_feasible_good_agent
from proposalAgent.agents.stage2.debate.feasible.feasible_bad import create_feasible_bad_agent
from proposalAgent.agents.stage2.debate.feasible.feasible_manager import create_feasible_manager
from proposalAgent.agents.stage2.debate.innovation.innovation_good import create_innovation_good_agent
from proposalAgent.agents.stage2.debate.innovation.innovation_bad import create_innovation_bad_agent
from proposalAgent.agents.stage2.debate.innovation.innovation_manager import create_innovation_manager


from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.agents.utils.agent_utils import create_msg_delete
from proposalAgent.tools.academic_analysis.google_scholar import get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations
from proposalAgent.tools.secondary_discipline_rag import secondary_discipline_search
from proposalAgent.tools.baidu_util import baidu_search_with_content


tool_nodes = {
            "academic": ToolNode([
                get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations,
          #      wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
                ]),
            "social": ToolNode([]),
            "influence": ToolNode([
                baidu_search_with_content
                ]),
            "interdisciplinary": ToolNode([secondary_discipline_search]),
            "feasibility": ToolNode([]),
            "innovation": ToolNode([]),
        }


config = TONGYI_CONFIG
toolkit = Toolkit(config=config)
deep_think_llm = ChatOpenAI(model="qwen-plus",
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                            api_key=TONGYI_CONFIG.get("api_key"))


quick_think_llm = ChatOpenAI(model="qwen-plus",
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                            api_key=TONGYI_CONFIG.get("api_key"))


conditional_logic = ConditionalLogic()
workflow = StateGraph(AgentState)
#full stage 1
intention_node = create_intention_agent(quick_think_llm)
output_node = create_output_node(quick_think_llm)
structure_node = create_structure_node()
planning_node = create_schedule_agent(deep_think_llm)

planning_clear_node = create_msg_delete()
# 添加stage1节点
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

# full stage 2
academic_analysis_node = create_academic_agent(
    quick_think_llm, toolkit=toolkit
)
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
        merged_state[key] = value

    merged_state["messages"] = (
        future_result.get("messages")
        or academic_result.get("messages")
        or state.get("messages")
    )

    return merged_state


# debate
interdisciplinary_node = create_interdis_agent(
            deep_think_llm, toolkit
        )

interdisciplinary_tool_exc_node = tool_nodes["interdisciplinary"]
interdisciplinary_msg_clear_node = create_msg_delete()


# 交叉性分析
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
workflow.add_edge("interdisciplinary_msg_clear_node", "debate_controller")



feasible_good_memory = EmbeddingMemory(name="feasible_good_memory", config=config)
feasible_bad_memory = EmbeddingMemory(name="feasible_bad_memory", config=config)
feasible_manager_memory = EmbeddingMemory(name="feasible_manager_memory", config=config)

innovation_good_memory = EmbeddingMemory(name="innovation_good_memory", config=config)
innovation_bad_memory = EmbeddingMemory(name="innovation_bad_memory", config=config)
innovation_manager_memory = EmbeddingMemory(name="innovation_manager_memory", config=config)


feasible_good_node = create_feasible_good_agent(
            deep_think_llm, toolkit, feasible_good_memory
        )

feasible_good_tool_exc_node = tool_nodes["feasibility"]

feasible_good_msg_clear_node = create_msg_delete()

feasible_bad_node = create_feasible_bad_agent(
    deep_think_llm, toolkit, feasible_bad_memory
)
#feasible_bad_tool_exc_node = tool_nodes["feasibility"]

#feasible_bad_msg_clear_node = create_msg_delete()

feasible_judge_node = create_feasible_manager(
    deep_think_llm, feasible_manager_memory
)

# 创新性辩论
innovation_good_node = create_innovation_good_agent(
    deep_think_llm, toolkit, innovation_good_memory
)

#innovation_good_tool_exc_node = tool_nodes["innovation"]

#innovation_good_msg_clear_node = create_msg_delete()

innovation_bad_node = create_innovation_bad_agent(
    deep_think_llm, toolkit, innovation_bad_memory
)

#innovation_bad_tool_exc_node = tool_nodes["innovation"]

#innovation_bad_msg_clear_node = create_msg_delete()
# 创新性总结
innovation_judge_node = create_innovation_manager(
    deep_think_llm, innovation_manager_memory
)

 # 1. 可行性辩论子图
feasibility_debate_workflow = StateGraph(AgentState)
feasibility_debate_workflow.add_node("feasible_good_node", feasible_good_node)
#feasibility_debate_workflow.add_node(
#    "feasible_good_tool_exc_node", feasible_good_tool_exc_node
#)
feasibility_debate_workflow.add_node("feasible_bad_node", feasible_bad_node)
#feasibility_debate_workflow.add_node(
#    "feasible_bad_tool_exc_node", feasible_bad_tool_exc_node
#)
feasibility_debate_workflow.add_node("feasible_judge_node", feasible_judge_node)

feasibility_debate_workflow.add_edge(START, "feasible_good_node")

# feasibility_debate_workflow.add_conditional_edges(
#     "feasible_good_node",
#     conditional_logic.should_continue_feasibility,
#     {
#         "continue": "feasible_bad_node",
#         "end": "feasible_judge_node",
#         "tools": "feasible_good_tool_exc_node",
#     },
# )
# feasibility_debate_workflow.add_edge(
#     "feasible_good_tool_exc_node", "feasible_good_node"
# )

# feasibility_debate_workflow.add_conditional_edges(
#     "feasible_bad_node",
#     conditional_logic.should_continue_feasibility,
#     {
#         "continue": "feasible_good_node",
#         "end": "feasible_judge_node",
#         "tools": "feasible_bad_tool_exc_node",
#     },
# )
feasibility_debate_workflow.add_edge(
    "feasible_good_node", "feasible_bad_node"
)
feasibility_debate_workflow.add_edge(
    "feasible_bad_node", "feasible_judge_node"
)
feasibility_debate_workflow.add_edge("feasible_judge_node", END)

compiled_feasibility_debate_graph = feasibility_debate_workflow.compile()

# 2. 创新性辩论子图 remove tool
innovation_debate_workflow = StateGraph(AgentState)
innovation_debate_workflow.add_node(
    "innovation_good_node", innovation_good_node
)
# innovation_debate_workflow.add_node(
#     "innovation_good_tool_exc_node", innovation_good_tool_exc_node
# )
innovation_debate_workflow.add_node("innovation_bad_node", innovation_bad_node)
# innovation_debate_workflow.add_node(
#     "innovation_bad_tool_exc_node", innovation_bad_tool_exc_node
# )

innovation_debate_workflow.add_node(
    "innovation_judge_node", innovation_judge_node
)

innovation_debate_workflow.add_edge(START, "innovation_good_node")

innovation_debate_workflow.add_edge("innovation_good_node", "innovation_bad_node")
innovation_debate_workflow.add_edge("innovation_bad_node", "innovation_judge_node")

innovation_debate_workflow.add_edge("innovation_judge_node", END)

compiled_innovation_debate_graph = innovation_debate_workflow.compile()



async def debate_controller(state: AgentState):
    disciplines = state.get("interdisciplinary_results", [])
    all_debate_outputs = {}
    tasks = []
    
    # parallel run outer
    for discipline in disciplines:
        async def single_task():
            input_state = state.copy()
            input_state["messages"] = state["messages"] + [
                SystemMessage(
                    content=f"Starting debates for discipline: {discipline}"
                )
            ]
            input_state["current_discipline"] = discipline

            # parallel run inner
            f_task = compiled_feasibility_debate_graph.ainvoke(input_state)
            i_task = compiled_innovation_debate_graph.ainvoke(input_state)
            results = await asyncio.gather(f_task, i_task)
            print(results)
            print(f"debug:{results[0].get('debate_results',{}).get(discipline,{}).get('可行性',{})}")
            print(f"debug:{results[1].get('debate_results',{}).get(discipline,{}).get('创新性',{})}")
            # merge results
            output_result_merge = {}
            output_result_merge["可行性"] = results[0].get("debate_results",{}).get(discipline,{}).get("可行性",{})
            output_result_merge["创新性"] = results[1].get("debate_results",{}).get(discipline,{}).get("创新性",{})
            
            return output_result_merge
        
        tasks.append(single_task())
    results = await asyncio.gather(*tasks)
    for discipline, result in zip(disciplines, results):
        all_debate_outputs[discipline] = result
    
    print(f"debug:all_debate_outputs:{all_debate_outputs}")
        
    state["debate_results"] = all_debate_outputs
    return state

workflow.add_node("debate_controller", debate_controller)

workflow.add_edge("debate_controller", END)

graph = workflow.compile()


async def main():
    
   
    
    state2:AgentState = {
            "messages": [{"role":"user","content":"帮我分析这篇文章"}],
            
            "filepath": "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/提交版本.pdf",
            "research_topic": ["新颖性","可行性"],
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
             "future_influence_limit":10,
            "future_influence_count":0,
            "interdisciplinary_results": [],
            "current_discipline": "",
            "debate_results": {},
            
            "final_analysis_summary": """
            
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
    config = {
        "configurable": {"thread_id": "stage2_test_thread"},
        "recursion_limit": 40
    }
    
    result = await graph.ainvoke(state2, config=config)
    print("\n\n")
    print(result)
    #result = await debate_controller(state)
   # print(result)

if __name__ == "__main__":
    asyncio.run(main())