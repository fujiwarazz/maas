from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime

from proposalAgent.tools.academic_analysis.google_scholar import get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations
from proposalAgent.tools.academic_analysis.wos_util import wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
from proposalAgent.tools.secondary_discipline_rag import secondary_discipline_search

def create_interdis_agent(llm, toolkit):
    """
    创建跨学科分析agent，用于分析申请人的跨学科背景和能力
    
    Args:
        llm: 语言模型实例
        toolkit: 工具包（暂未使用，保留接口兼容性）
    
    Returns:
        interdis_agent: 跨学科分析agent函数
    """
    def interdis_agent(state):
        # 需要 research_basic_info 与 research_report_body_summary
        research_info = state.get("research_basic_info")
        research_body = state.get("research_report_body_summary")
        tools = [secondary_discipline_search]
        if research_info and research_body:
            system_prompt = """
            你是一个跨学科领域识别专家。请根据给定的研究基础信息与正文摘要，判断涉及到的学科/领域，可以是多个。

            输出要求：
            - 仅输出以中文领域名称组成的列表，使用逗号分隔，不要包含其他解释性文字。
            
            必须使用rag工具来获得对齐后的二级学科名称，因此你需要调用工具来获得你认为所属学科的对齐之后的学科名字
            例如：你根据输入认为这是一篇计算机科学的文章，那么你就可以构造输入为"计算机科学"的文本来调用工具，获得对齐后的二级学科名称，输入给工具的文本一定是代表性且不能太长！

            判定依据（仅作参考，不用复述）：关键词、研究方法、数据类型、应用场景、引用领域等。

            研究基础信息：{research_info}
            正文摘要：{research_body}
            工具信息：{tool_names}
            """
            llm_with_tools = llm.bind_tools(tools)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            prompt = prompt.partial(research_info=research_info)
            prompt = prompt.partial(research_body=research_body)
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

            chain = prompt | llm_with_tools
            result = chain.invoke(state.get("messages"))
            # 将输出按逗号分隔并清洗空白
            if len(result.tool_calls) == 0:
                interdis_report = [s.strip() for s in str(result.content).split(",") if s.strip()]
            else:
                interdis_report = "正在搜索所属学科类别..."


            return {
                "messages": [result],
                "interdisciplinary_results": interdis_report,
            }
            
    return interdis_agent

