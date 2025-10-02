from typing import cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState


def create_generator_agent(llm: ChatOpenAI):
    """
    创建最终报告生成智能体，用于综合所有分析信息生成最终的评价报表。
    该智能体会整合所有前置分析的结果，生成一份完整的项目评估报告。
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个专业的项目评估报告生成智能体。你的任务是基于所有收集到的分析信息，生成一份全面、专业、结构化的项目评价报表。

            请按照以下结构生成报告：
            
            ## 项目评估报告
            
            ### 1. 执行摘要
            - 项目概述
            - 主要发现
            - 总体评价结论
            - 核心建议
            
            ### 2. 项目基本信息
            - 研究主题与目标
            - 申请人基本情况
            - 项目团队构成
            - 申请基本信息
            
            ### 3. 学术能力评估
            - 申请人学术背景分析
            - 科研能力评价
            - 学术影响力评估
            - 相关领域经验
            
            ### 4. 社会影响力分析
            - 当前社会影响力评估
            - 公众认知度分析
            - 社会价值潜力
            
            ### 5. 未来发展前景
            - 项目发展潜力
            - 预期影响力
            - 风险因素分析
            - 可持续性评估
            
            ### 6. 跨学科协作评估
            - 涉及学科领域
            - 跨学科整合能力
            - 协作优势分析
            
            ### 7. 可行性与创新性评估
            - 可行性分析总结
            - 创新性评价
            - 技术路线评估
            - 资源配置合理性
            
            ### 8. 综合评价
            - 优势总结
            - 劣势分析
            - 风险评估
            - 改进建议
            
            ### 9. 评分与建议
            - 各维度评分（1-5分制）
            - 总体推荐度
            - 具体改进建议
            - 后续跟进计划
            
            请确保报告内容客观、准确、全面，语言专业规范，结论有据可依。
            """,
            ),
            (
                "human",
                """请基于以下全部分析信息，生成最终的项目评估报告：
 
                    分析报告：
                    学术分析：{academic_analysis_report}
                    社会分析：{social_analysis_report}
                    未来影响分析：{future_influence_report}

                    跨学科分析结果：{interdisciplinary_results}
                    辩论结果：{debate_results}
                    
                    最终分析摘要：{final_analysis_summary}
                    
                    完备性检查结果：{completeness_check_result}
                    
                    人类反馈（如有）：{human_feedback}
                    
                    请生成完整的项目评估报告。""",
            ),
        ]
    )
    #  研究主题：{research_topic}
    #                 研究结构：{research_structure}
    #                 申请人信息：{research_person_info}
    #                 基本信息：{research_basic_info}
    #                 项目团队信息：{research_project_team_info}
    #                 项目申请信息：{research_project_apply_info}
    #                 报告主体摘要：{research_report_body_summary}

    def generator_node(state: AgentState):
        """
        报告生成节点的执行函数
        """
        # 准备输入数据
        input_data = {
            "research_topic": state.get("research_topic", "未提供"),
            "research_structure": state.get("research_structure", "未提供"),
            "research_person_info": state.get("research_person_info", "未提供"),
            "research_basic_info": state.get("research_basic_info", "未提供"),
            "research_project_team_info": state.get(
                "research_project_team_info", "未提供"
            ),
            "research_project_apply_info": state.get(
                "research_project_apply_info", "未提供"
            ),
            "research_report_body_summary": state.get(
                "research_report_body_summary", "未提供"
            ),
            "academic_analysis_report": state.get(
                "academic_analysis_report", "未进行学术分析"
            ),
            "social_analysis_report": state.get(
                "social_analysis_report", "未进行社会分析"
            ),
            "future_influence_report": state.get(
                "future_influence_report", "未进行未来影响分析"
            ),
            "interdisciplinary_results": state.get("interdisciplinary_results", []),
            "debate_results": _format_debate_results(state.get("debate_results", {})),
            "final_analysis_summary": state.get(
                "final_analysis_summary", "未完成最终分析"
            ),
            "completeness_check_result": _format_completeness_result(
                state.get("completeness_check_result", {})
            ),
            "human_feedback": state.get("human_feedback", "无人类反馈"),
        }

        # 调用LLM生成报告
        chain = prompt | llm
        result = chain.invoke(input_data)

        # 更新状态
        final_report_content = (
            result.content if hasattr(result, "content") else str(result)
        )
        state["final_report"] = cast(str, final_report_content)

        # 添加生成消息到消息历史
        state["messages"].append(
            AIMessage(
                content=f"最终评估报告已生成。报告包含{len(final_report_content)}个字符的详细分析内容。"
            )
        )

        print("=== 最终评估报告已生成 ===")
        print(f"报告长度: {len(final_report_content)} 字符")

        return state

    return generator_node


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
                        formatted_results.append(
                            f"裁判总结：{result_text['judge_summary']}"
                        )
                    if "full_history" in result_text:
                        formatted_results.append(
                            f"辩论历史：{result_text['full_history']}"
                        )
                else:
                    formatted_results.append(str(result_text))
        elif isinstance(debate_data, list):
            debate_type_names = ["可行性辩论", "创新性辩论"]
            for i, result_data in enumerate(debate_data):
                if i < len(debate_type_names):
                    formatted_results.append(f"\n【{debate_type_names[i]}】")
                    if isinstance(result_data, dict):
                        if "judge_summary" in result_data:
                            formatted_results.append(
                                f"裁判总结：{result_data['judge_summary']}"
                            )
                        if "full_history" in result_data:
                            formatted_results.append(
                                f"辩论历史：{result_data['full_history']}"
                            )
                    else:
                        formatted_results.append(str(result_data))
        else:
            formatted_results.append(str(debate_data))

    return "\n".join(formatted_results) if formatted_results else "辩论结果格式异常"


def _format_completeness_result(completeness_result) -> str:
    """
    格式化完备性检查结果
    """
    if not completeness_result:
        return "未进行完备性检查"

    try:
        formatted = f"""完备性检查结果：
                    - 完备性：{'通过' if completeness_result.get('is_complete') else '未通过'}
                    - 自洽性：{'通过' if completeness_result.get('is_consistent') else '未通过'}
                    - 质量评分：{completeness_result.get('overall_quality', 'N/A')}/5
                    - 缺失部分：{', '.join(completeness_result.get('missing_parts', []))}
                    - 不一致问题：{', '.join(completeness_result.get('inconsistencies', []))}
                    - 建议：{completeness_result.get('recommendation', 'N/A')}
                    - 原因：{completeness_result.get('reason', 'N/A')}"""
        return formatted
    except (KeyError, TypeError, ValueError) as e:
        return f"完备性检查结果格式异常: {e}"
