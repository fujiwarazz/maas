from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime
from typing import Optional
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.tools.academic_analysis.google_scholar import get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations
from proposalAgent.tools.academic_analysis.wos_util import wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary

def create_academic_agent(llm, toolkit,memory:Optional[EmbeddingMemory]):
    """
    创建学术分析agent，用于分析申请人的学术背景和能力
    
    Args:
        llm: 语言模型实例
        toolkit: 工具包（暂未使用，保留接口兼容性）
    
    Returns:
        academic_agent: 学术分析agent函数
    """
    def academic_agent(state):
        try:
            tools = [get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations,
              #       wos_expanded_search, wos_expanded_citation_fanout
                     ]

            system_message = (
                "你是一个专业的学术分析专家，负责对学术申请书中的项目团队成员进行深度的学术背景调研和能力评估。"
                "你的任务是使用Google Scholar,Web of Science等学术工具，全面分析项目申请人的学术能力、科研背景、学术影响力等关键指标。"
                "Web of science工具一般用于查询文章以及文章的引用关系，google scholar可以用于查询作者。"
                "当你已经获得足够的学术数据（如作者引用信息、文章列表等）后，请停止调用工具，直接生成完整的学术分析报告。"
                "请对申请人进行详细的学术分析，包括但不限于：发表论文质量、被引用情况、学术声誉、研究领域影响力等。"
                "并在报告末尾生成对他的完整的学术分析报告，评价不足和优点。"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一个专业的学术分析助手，与其他助手协作完成学术申请书的评估工作。"
                        "请使用提供的工具来分析项目团队成员的学术背景和能力。"
                        "当你已经获得充足的学术数据（如作者引用信息、h指数、论文列表等）后，请立即停止调用工具，直接基于已有数据生成完整详细的学术分析报告。"
                        "不要尝试调用可能失败的复杂工具，优先生成实用的分析报告。"
                        "如果你或其他助手已经完成了最终的学术分析报告，请在回复前加上'最终学术分析报告：'标识。"
                        "你可以使用以下工具：{tool_names}。\n{system_message}"
                        "项目团队信息：{person_info}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
                        
            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            prompt = prompt.partial(person_info=state["research_person_info"])

            llm_with_tools = llm.bind_tools(tools)
            chain = prompt | llm_with_tools
            result = chain.invoke(state["messages"])

            academic_report = ""
            
            if len(result.tool_calls) == 0:
                academic_report = result.content if result.content else "学术分析已完成，但未生成详细报告内容。"
                print(f"academic_report: {academic_report}")
            else:
                academic_report = "正在使用学术分析工具进行深度调研..."

            return {
                "messages": [result],
                "academic_analysis_report": academic_report,
            }
            
        except Exception as e:
            error_message = f"学术分析过程中发生错误: {str(e)}"
            print(f"Academic agent error: {e}")
            
            return {
                "messages": [],
                "academic_analysis_report": error_message,
            }

    return academic_agent