from typing import cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState

def create_final_analyst_agent(llm: ChatOpenAI):
    """
    创建最终分析智能体，用于收集和整理所有前置智能体的输出。
    该智能体会分析来自各个阶段的信息，包括：
    - 学术分析报告
    - 社会分析报告  
    - 未来影响分析报告
    - 跨学科分析结果
    - 各学科的可行性和创新性辩论结果
    """
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个最终分析智能体，负责收集、整理和综合分析所有前置智能体的输出结果。

            你需要分析以下信息：
            1. 研究主题和基本信息
            2. 学术分析报告 - 申请人的学术能力和科研背景
            3. 社会分析报告 - 申请人的社会影响力
            4. 未来影响分析报告 - 项目可能的未来影响力
            5. 跨学科分析结果 - 识别的相关学科领域
            6. 各学科的辩论结果：
            - 可行性辩论（正方、反方、裁判总结）
            - 创新性辩论（正方、反方、裁判总结）

            你的任务是：
            1. 整合所有信息，提供一个全面的详细分析总结
            2. 识别关键发现和潜在问题
            3. 评估整体项目的优势和劣势
            4. 提供置信度评估和建议

            请用中文输出，结构化地组织你的分析结果。
            """
        ),
        (
            "human", 
            """基于当前状态中的所有信息，请进行综合分析：

            研究主题：{research_topic}
            申请人信息：{research_person_info}
            基本信息：{research_basic_info}
            项目团队信息：{research_project_team_info}
            项目申请信息：{research_project_apply_info}
            报告主体摘要：{research_report_body_summary}

            分析报告：
            学术分析：{academic_analysis_report}
            未来影响分析：{future_influence_report}


            辩论结果：{debate_results}
            请提供全面的综合分析和完整的总结。"""
        ),
    ])
    
    def final_analyst_node(state: AgentState):
        """
        最终分析节点的执行函数
        """
        # 准备输入数据
        input_data = {
            "research_topic": state.get("research_topic", "未提供"),
            "research_structure": state.get("research_structure", "未提供"),
            "research_person_info": state.get("research_person_info", "未提供"),
            "research_basic_info": state.get("research_basic_info", "未提供"),
            "research_project_team_info": state.get("research_project_team_info", "未提供"),
            "research_project_apply_info": state.get("research_project_apply_info", "未提供"),
            "research_report_body_summary": state.get("research_report_body_summary", "未提供"),
            "academic_analysis_report": state.get("academic_analysis_report", "未进行学术分析"),
            "social_analysis_report": state.get("social_analysis_report", "未进行社会分析"),
            "future_influence_report": state.get("future_influence_report", "未进行未来影响分析"),
            "interdisciplinary_results": state.get("interdisciplinary_results", []),
            "debate_results": _format_debate_results(state.get("debate_results", {}))
        }
        
        # 调用LLM进行分析
        chain = prompt | llm
        result = chain.invoke(input_data)
        
        # 更新状态
        final_analysis_content = result.content if hasattr(result, 'content') else str(result)
        state["final_analysis_summary"] = cast(str, final_analysis_content)
        
        # 添加分析消息到消息历史
        state["messages"].append(AIMessage(content=f"最终分析完成：\n{final_analysis_content}"))
        
        return state
    
    return final_analyst_node
def _format_debate_results(debate_results) -> str:
    """
    格式化辩论结果，使其更易于阅读和分析
    结构：{学科: {辩论类型: 结果文本}}
    """
    if not debate_results:
        return "未进行辩论分析"
    
    formatted_results = []
    
    for discipline, debate_data in debate_results.items():
        formatted_results.append(f"\n=== {discipline} 学科辩论结果 ===")
        
        # 处理不同的数据结构
        if isinstance(debate_data, dict):
            # 如果是字典格式 {辩论类型: 结果文本}
            for debate_type, result_text in debate_data.items():
                formatted_results.append(f"\n【{debate_type}】")
                if isinstance(result_text, str):
                    formatted_results.append(result_text)
                elif isinstance(result_text, dict):
                    # 如果结果是DebateState类型的字典
                    if "judge_summary" in result_text:
                        formatted_results.append(f"裁判总结：{result_text['judge_summary']}")
                    # if "full_history" in result_text:
                    #     formatted_results.append(f"辩论历史：{result_text['full_history']}")
                else:
                    formatted_results.append(str(result_text))
        elif isinstance(debate_data, list):
            # 如果是列表格式 [可行性辩论结果, 创新性辩论结果]
            debate_type_names = ["可行性辩论", "创新性辩论"]
            for i, result_data in enumerate(debate_data):
                if i < len(debate_type_names):
                    formatted_results.append(f"\n【{debate_type_names[i]}】")
                    if isinstance(result_data, dict):
                        if "judge_summary" in result_data:
                            formatted_results.append(f"裁判总结：{result_data['judge_summary']}")
                        # if "full_history" in result_data:
                        #     formatted_results.append(f"辩论历史：{result_data['full_history']}")
                    else:
                        formatted_results.append(str(result_data))
        else:
            formatted_results.append(str(debate_data))
    
    return "\n".join(formatted_results) if formatted_results else "辩论结果格式异常"

