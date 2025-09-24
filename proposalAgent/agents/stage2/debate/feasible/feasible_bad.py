# 不用tool的单agent
# 1、制定学科的可信性分析
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime
from proposalAgent.agents.utils.agent_states import AgentState,DebateState
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.prompts.discipline_feasible_agent_template import generate_discipline_agent_prompt
from proposalAgent.utils.logger import get_logger

logger = get_logger("feasible_bad_agent")

def create_feasible_bad_agent(llm, toolkit,memory:EmbeddingMemory):
    """
    创建可行性反方辩论agent，用于反对项目的可行性
    
    Args:
        llm: 语言模型实例
        toolkit: 
    
    Returns:
        feasible_bad_agent: 可行性正方辩论agent函数
    """
    def feasible_bad_agent(state: AgentState):
    
    
        research_info = state.get("research_basic_info")
        academic_report = state.get("academic_analysis_report")
        research_project_apply_info = state.get("research_project_apply_info")
        
        
        research_body = state.get("research_report_body_summary")

        
        debate_kind = "可行性"
        _curr = state.get("current_discipline")
        # 学科分类
        _disc_name = _curr[1] if isinstance(_curr, tuple) and len(_curr) >= 2 else ""
        _disc_code =  _curr[0] if isinstance(_curr, tuple) and len(_curr) >= 2 else ""
        current_debate =  state.get("debate_results", {}).get(_disc_name, {}).get(debate_kind, {})
        full_history = []
        
        
        if _disc_name:
            full_history = current_debate.get("full_history", [])
            
        
        feasbile_good_his = current_debate.get("good_agent_history",[])
        feasbile_bad_his = current_debate.get("bad_agent_history",[])
        
        prev_debate_str = "\n".join(full_history) if isinstance(full_history, list) else ""
        
        
        curr_situation = f"{research_info}\n\n{academic_report}\n\n{research_project_apply_info}\n\n{research_body}"
        
        if _disc_code and _disc_name:
            role_description = generate_discipline_agent_prompt(_disc_code,_disc_name)
        else:
            role_description = ""
            logger.error("缺少学科信息分类")
            
        # past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memories = []

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"
        if prev_debate_str:
            past_memory_str += "\n\n历史辩论片段:\n" + prev_debate_str
        
        if research_info and research_body:
            prompt = f"""
            ### 人物背景
            {role_description}

            **同时你是一个项目可行性论证专家**,请根据给定的研究基础信息与正文摘要，提出**反对**该项目可行性的**反方论点**，并且对正方观点进行驳斥。

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
            
            feasible_report = str(response.content).strip()
            
        
            argument = f"可行性反方观点: {feasible_report}"

        
            new_full_history = full_history + [argument]
            new_bad_history = feasbile_bad_his + [argument]
            
            new_feasible_bad_debate_state = {
                "full_history": new_full_history,
                "good_agent_history": feasbile_good_his,
                "bad_agent_history": new_bad_history,
                "debate_rounds": current_debate.get("debate_rounds", 1),
                "judge_summary": "",
            }
            
            current_disc_debates = state.get("debate_results", {}).get(_disc_name, {})
            new_debate_result = {
                "可行性": new_feasible_bad_debate_state,
                "创新性": current_disc_debates.get("创新性", {})
            }
            
            origin_debate_results = state.get("debate_results", {})
            new_debate_results = dict(origin_debate_results)  
            new_debate_results[_disc_name] = new_debate_result
            
            return {"debate_results": new_debate_results}

        else:
            raise ValueError("缺少必要的研究信息或正文摘要，无法进行可行性分析。")

    return feasible_bad_agent
