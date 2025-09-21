import json
from typing import cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState

def create_feedback_analysis_agent(llm: ChatOpenAI):
    """
    创建反馈分析智能体，用于分析人类反馈并决定下一步执行路径。
    该智能体会分析人类提供的反馈，识别需要改进的具体部分，并决定重新执行哪个节点。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个专业的反馈分析智能体。你的任务是分析人类提供的反馈意见，识别问题所在，并决定下一步的执行路径。

            请从以下几个方面分析反馈：
            
            1. **问题定位**：
            - 识别反馈中提到的具体问题
            - 判断问题属于哪个分析阶段
            - 评估问题的严重程度
            
            2. **缺失内容分析**：
            - 学术分析是否需要补充（申请人背景、学术能力等）
            - 社会分析是否需要加强（影响力、公众认知等）
            - 未来影响预测是否需要深化
            - 跨学科分析是否需要扩展
            - 辩论过程是否需要重新进行
            - 其他遗漏的关键信息
            
            3. **路径决策**：
            根据问题类型，选择以下执行路径之一：
            - "academic_analysis": 重新进行学术分析
            - "social_analysis": 重新进行社会分析  
            - "future_influence": 重新进行未来影响分析
            - "interdisciplinary": 重新进行跨学科分析
            - "debate": 重新进行辩论环节
            - "generate": 直接生成最终报告（如果反馈正面且无需修改）
            
            请输出JSON格式结果，包含以下字段：
            - "feedback_summary": 反馈内容总结
            - "identified_issues": [], 识别的问题列表
            - "missing_content": [], 缺失的内容列表
            - "priority_level": 1-5, 问题优先级（5为最高）
            - "next_step": 选择的执行路径
            - "reason": 选择该路径的详细原因
            - "specific_instructions": 给执行节点的具体指导意见
            
            如果反馈是正面的且没有实质性问题，选择"generate"。
            """
        ),
        (
            "human", 
            """请分析以下人类反馈，并决定下一步执行路径：

            原始分析结果：
            研究主题：{research_topic}
            学术分析：{academic_analysis_report}
            社会分析：{social_analysis_report}
            未来影响分析：{future_influence_report}
            跨学科分析：{interdisciplinary_results}
            辩论结果：{debate_results}
            最终分析摘要：{final_analysis_summary}
            
            完备性检查结果：{completeness_check_result}
            
            人类反馈内容：{human_feedback}
            
            请分析反馈内容，识别需要改进的部分，并选择合适的执行路径。"""
        ),
    ])
    
    def feedback_analysis_node(state: AgentState):
        """
        反馈分析节点的执行函数
        """
        # 准备输入数据
        input_data = {
            "research_topic": state.get("research_topic", "未提供"),
            "academic_analysis_report": state.get("academic_analysis_report", "未进行学术分析"),
            "social_analysis_report": state.get("social_analysis_report", "未进行社会分析"),
            "future_influence_report": state.get("future_influence_report", "未进行未来影响分析"),
            "interdisciplinary_results": state.get("interdisciplinary_results", []),
            "debate_results": _format_debate_results(state.get("debate_results", {})),
            "final_analysis_summary": state.get("final_analysis_summary", "未完成最终分析"),
            "completeness_check_result": _format_completeness_result(state.get("completeness_check_result", {})),
            "human_feedback": state.get("human_feedback", "无人类反馈")
        }
        
        # 调用LLM进行分析
        chain = prompt | llm
        result = chain.invoke(input_data)
        
        # 解析结果
        try:
            content = result.content if hasattr(result, 'content') else str(result)
            feedback_analysis_result = json.loads(content)
            
            # 更新状态
            state["feedback_analysis_result"] = feedback_analysis_result
            state["feedback_routing_decision"] = feedback_analysis_result.get("next_step", "generate")
            state["feedback_instructions"] = feedback_analysis_result.get("specific_instructions", "")
            
            # 添加分析消息到消息历史
            analysis_summary = f"""反馈分析完成：
发现问题：{len(feedback_analysis_result.get('identified_issues', []))}个
缺失内容：{len(feedback_analysis_result.get('missing_content', []))}项
优先级：{feedback_analysis_result.get('priority_level', 'N/A')}/5
决定路径：{feedback_analysis_result.get('next_step', 'generate')}
原因：{feedback_analysis_result.get('reason', '无详细说明')}"""
            
            state["messages"].append(AIMessage(content=analysis_summary))
            
            print(f"反馈分析完成，下一步路径：{feedback_analysis_result.get('next_step', 'generate')}")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析反馈分析结果时出错: {e}")
            # 默认生成报告
            state["feedback_analysis_result"] = {
                "feedback_summary": "解析错误",
                "identified_issues": [f"解析错误: {e}"],
                "missing_content": [],
                "priority_level": 1,
                "next_step": "generate",
                "reason": f"解析反馈分析结果时出错，默认生成报告: {e}",
                "specific_instructions": ""
            }
            state["feedback_routing_decision"] = "generate"
            state["feedback_instructions"] = ""
            
            state["messages"].append(AIMessage(content=f"反馈分析遇到错误，默认生成报告: {e}"))
        
        return state
    
    return feedback_analysis_node

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
    except Exception as e:
        return f"完备性检查结果格式异常: {e}"