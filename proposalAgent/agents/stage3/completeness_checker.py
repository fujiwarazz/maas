import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState

def create_completeness_checker_agent(llm: ChatOpenAI):
    """
    创建完备性检查智能体，用于判断分析结果是否完备和自洽。
    该智能体会评估所有收集到的信息是否足够全面，逻辑是否一致。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个专业的完备性和自洽性检查智能体。你的任务是评估项目分析的完整程度和逻辑一致性。

            请从以下几个维度进行评估：
            
            1. **完备性检查**：
            - 学术分析是否深入（申请人背景、学术能力等）
            - 未来影响预测是否合理
            - 辩论结果是否涵盖了关键争议点
            
            2. **自洽性检查**：
            - 各部分分析结论是否相互支持
            - 是否存在矛盾的观点
            - 评价标准是否一致
            - 论据和结论是否匹配
            
            3. **质量评估**：
            - 分析深度是否充足
            - 证据支撑是否充分
            - 风险评估是否全面
            
            请输出JSON格式结果，包含以下字段：
            - "is_complete": boolean, 是否完备
            - "is_consistent": boolean, 是否自洽
            - "overall_quality": 1-5分，整体质量评分
            - "missing_parts": [], 缺失的部分列表
            - "inconsistencies": [], 发现的不一致问题
            - "recommendation": "complete" 或 "need_human_review", 建议下一步操作
            - "reason": 推荐理由的详细说明
            
            如果分析完备且自洽，建议"complete"；否则建议"need_human_review"。
            """
        ),
        (
            "human", 
            """请评估以下项目分析的完备性和自洽性：

            申请人信息：{research_person_info}
            项目团队信息：{research_project_team_info}
            项目申请信息：{research_project_apply_info}
            报告主体摘要：{research_report_body_summary}

            分析报告：
            学术分析：{academic_analysis_report}
            社会分析：{social_analysis_report}
            未来影响分析：{future_influence_report}

            跨学科分析结果：{interdisciplinary_results}
            辩论结果：{debate_results}
            
            最终分析摘要：{final_analysis_summary}
            
            请进行全面的完备性和自洽性评估。"""
        ),
    ])
    
    def completeness_checker_node(state: AgentState):
        """
        完备性检查节点的执行函数
        """
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
            "debate_results": _format_debate_results(state.get("debate_results", {})),
            "final_analysis_summary": state.get("final_analysis_summary", "未完成最终分析")
        }
        
        chain = prompt | llm
        result = chain.invoke(input_data)
        
        try:
            content = result.content if hasattr(result, 'content') else str(result)
            print(f"LLM 原始响应: {content}")  # 调试信息
            
            # 尝试清理响应内容
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            completeness_result = json.loads(content)
            
            state["completeness_check_result"] = completeness_result
            state["is_analysis_complete"] = completeness_result.get("is_complete", False)
            state["is_analysis_consistent"] = completeness_result.get("is_consistent", False)
            state["completeness_recommendation"] = completeness_result.get("recommendation", "need_human_review")
            
            check_summary = f"""完备性检查完成：
                        完备性：{'通过' if completeness_result.get('is_complete') else '未通过'}
                        自洽性：{'通过' if completeness_result.get('is_consistent') else '未通过'}
                        质量评分：{completeness_result.get('overall_quality', 'N/A')}/5
                        建议：{completeness_result.get('recommendation', 'need_human_review')}
                        原因：{completeness_result.get('reason', '无详细说明')}"""
            
            state["messages"].append(AIMessage(content=check_summary))
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析完备性检查结果时出错: {e}")
            state["completeness_check_result"] = {
                "is_complete": False,
                "is_consistent": False,
                "overall_quality": 1,
                "missing_parts": ["解析错误"],
                "inconsistencies": [],
                "recommendation": "need_human_review",
                "reason": f"解析检查结果时出错: {e}"
            }
            state["is_analysis_complete"] = False
            state["is_analysis_consistent"] = False
            state["completeness_recommendation"] = "need_human_review"
            
            state["messages"].append(AIMessage(content=f"完备性检查遇到错误，默认需要人类审核: {e}"))
        
        return state
    
    return completeness_checker_node

def _format_debate_results(debate_results) -> str:
    """
    格式化辩论结果，使其更易于阅读和分析
    """
    if not debate_results:
        return "未进行辩论分析"
    
    formatted_results = []
    
    for discipline, debate_data in debate_results.items():
        formatted_results.append(f"\n=== {discipline} 学科辩论结果 ===")
        
        if isinstance(debate_data, dict):
            for debate_type, result_text in debate_data.items():
                formatted_results.append(f"\n【{debate_type}】")
                if isinstance(result_text, str):
                    formatted_results.append(result_text)
                elif isinstance(result_text, dict):
                    if "judge_summary" in result_text:
                        formatted_results.append(f"裁判总结：{result_text['judge_summary']}")
                    if "full_history" in result_text:
                        formatted_results.append(f"辩论历史：{result_text['full_history']}")
                else:
                    formatted_results.append(str(result_text))
        elif isinstance(debate_data, list):
            debate_type_names = ["可行性辩论", "创新性辩论"]
            for i, result_data in enumerate(debate_data):
                if i < len(debate_type_names):
                    formatted_results.append(f"\n【{debate_type_names[i]}】")
                    if isinstance(result_data, dict):
                        if "judge_summary" in result_data:
                            formatted_results.append(f"裁判总结：{result_data['judge_summary']}")
                        if "full_history" in result_data:
                            formatted_results.append(f"辩论历史：{result_data['full_history']}")
                    else:
                        formatted_results.append(str(result_data))
        else:
            formatted_results.append(str(debate_data))
    
    return "\n".join(formatted_results) if formatted_results else "辩论结果格式异常"
