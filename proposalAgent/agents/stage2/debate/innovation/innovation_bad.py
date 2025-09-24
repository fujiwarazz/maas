
# 不用tool的单agent
# 1、制定学科的可信性分析
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime
from proposalAgent.agents.utils.agent_states import AgentState,DebateState
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.prompts.discipline_innovation_agent_template import generate_discipline_agent_prompt
def create_innovation_bad_agent(llm, toolkit,memory:EmbeddingMemory):
    """
    创建创新性反方辩论agent，用于反对项目的创新性
    
    Args:
        llm: 语言模型实例
        toolkit: 工具包（暂未使用，保留接口兼容性）
    
    Returns:
        innovation_bad_agent: 创新性反方辩论agent函数
    """
    def innovation_bad_agent(state: AgentState):
    
    
        research_info = state.get("research_basic_info")
        academic_report = state.get("academic_analysis_report")
        research_project_apply_info = state.get("research_project_apply_info")
        
        
        research_body = state.get("research_report_body_summary")
        
        
        debate_kind = "创新性"
        _curr = state.get("current_discipline")
        _disc_name = _curr[1] if isinstance(_curr, tuple) and len(_curr) >= 2 else ""
        _disc_code =  _curr[0] if isinstance(_curr, tuple) and len(_curr) >= 2 else ""
        current_debate =  state.get("debate_results", {}).get(_disc_name, {}).get(debate_kind, {})
        full_history = []
        if _disc_name:
            full_history = current_debate.get("full_history", [])
            
        
        feasbile_bad_his = current_debate.get("bad_agent_history",[])
        feasbile_bad_his = current_debate.get("bad_agent_history",[])
        
        prev_debate_str = "\n".join(full_history) if isinstance(full_history, list) else ""
        
        
        curr_situation = f"{research_info}\n\n{academic_report}\n\n{research_project_apply_info}\n\n{research_body}"
        past_memories = []
        #past_memories = memory.get_memories(curr_situation, n_matches=2)
        
        role_description = generate_discipline_agent_prompt(_disc_code,_disc_name)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"
        if prev_debate_str:
            past_memory_str += "\n\n历史辩论片段:\n" + prev_debate_str
        
        if research_info and research_body:
            prompt = f"""
            ### 人物背景
            {role_description}

            **同时你是一个项目创新性论证专家**,请根据给定的研究基础信息与正文摘要，提出**反对**该项目创新性的**反方论点**，并且对正方观点进行驳斥。

            ### 输出要求：
            - 以中文输出，条理清晰，分点阐述。
            - 在开始之前，如果有正方观点，你应该先反对对方的观点，并且说明理由

            - 每个论点应简洁明了，避免冗长。
            - 论点应具体且有说服力，避免泛泛而谈。
            - 不要包含反对意见或不确定的内容。
            - 不要提及任何与你角色无关的信息。

            ### 其他要求：
            判定依据（仅作参考，不用复述）：项目背景、研究方法、数据资源、团队能力、技术路线等。

            ### 可用相关信息：
            历史辩论信息: {prev_debate_str}
            研究基础信息：{research_info}
            项目申请正文：{research_body}
            学术分析报告：{academic_report}
            项目申请信息：{research_project_apply_info}
            可供分析的历史辩论消息：{past_memory_str}
            
            ### 要点：
            请聚焦于可操作性的见解和持续改进。在总结以往经验的基础上，批判性地评估各方面观点，确保每一项决策都能推动项目取得更优成果。"""

            response = llm.invoke(prompt)
            
            innovation_report = str(response.content).strip()
            
        
            argument = f"创新性反方观点: {innovation_report}"

        
            new_innovation_bad_debate_state = {
                "full_history":full_history.append(argument),
                "bad_agent_history":feasbile_bad_his.append(argument),
                "bad_agent_history":feasbile_bad_his,
                "debate_rounds":current_debate.get("debate_rounds",1),
                "judge_summary":"",
                
                
            }
            new_debate_result= {
                "创新性":new_innovation_bad_debate_state,
                "可行性": state.get("debate_results", {}).get(_disc_name, {}).get("可行性", {})
            }
            
            origin_debate_results = state.get("debate_results",{})
            new_debate_results = origin_debate_results
            new_debate_result[_disc_name]=new_debate_result
            
            return {"debate_results": new_debate_results}

        else:
            raise ValueError("缺少必要的研究信息或正文摘要，无法进行创新性分析。")

    return innovation_bad_agent
