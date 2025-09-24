"""
未来影响力分析Agent
用于评估研究项目的未来影响力和发展潜力
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime
from typing import Optional
from proposalAgent.agents.utils.memory import EmbeddingMemory

# 导入未来影响力分析工具
from proposalAgent.tools.baidu_util import baidu_search_with_content

def create_future_influence_agent(llm, toolkit):
    """
    创建未来影响力分析agent，用于评估项目的未来发展潜力和影响力
    
    Args:
        llm: 语言模型实例
        toolkit: 工具包（暂未使用，保留接口兼容性）
        memory: 记忆模块，用于存储分析结果
    
    Returns:
        future_influence_agent: 未来影响力分析agent函数
    """
    def future_influence_agent(state):
        try:
            # 获取未来影响力分析工具
            tools = [baidu_search_with_content]


            system_message = (
                "你是一个专业的未来影响力分析专家，负责评估研究项目的未来发展潜力和社会影响力。"
                "你的任务是从多个维度全面分析项目的未来影响，包括学术影响力、技术转化潜力、社会效益等。"
                "请使用提供的分析工具，对项目进行深度的前瞻性评估。"
                "\n\n分析维度包括："
                "\n1. 研究趋势分析 - 评估研究方向的发展趋势和热度"
                "\n2. 学术影响力预测 - 预测研究成果的引用潜力和学术声誉"
                "\n3. 技术成熟度评估 - 分析技术的商业化潜力和市场前景"
                "\n4. 社会影响力评估 - 评估项目对社会问题的解决能力和政策相关性"
                "\n\n请基于分析结果，生成全面的未来影响力评估报告，并提供发展建议。"
            )

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "你是一个专业的未来影响力分析助手，与其他助手协作完成研究项目的全面评估。"
                    "请使用提供的工具来分析项目的未来发展潜力和影响力。"
                    "当你获得足够的分析数据后，请基于多维度评估结果生成完整详细的未来影响力分析报告。"
                    "你的分析应该客观、准确，既要指出项目的潜力，也要识别可能的风险和挑战。"
                    "如果你已经完成了最终的未来影响力分析报告，请在回复前加上'最终未来影响力分析报告：'标识。"
                    "你可以使用以下工具：{tool_names}。\n{system_message}"
                    "\n\n项目信息：{project_info}"
                    "\n\n研究人员信息：{person_info}"
                    "\n\n项目申请信息：{application_info}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ])
                        
            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            
            # 从状态中获取项目相关信息
            project_info = state.get("research_basic_info", "暂无项目基本信息")
            person_info = state.get("research_person_info", "暂无研究人员信息")
            application_info = state.get("research_project_apply_info", "暂无项目申请信息")
            
            prompt = prompt.partial(project_info=project_info)
            prompt = prompt.partial(person_info=person_info)
            prompt = prompt.partial(application_info=application_info)

            llm_with_tools = llm.bind_tools(tools)
            chain = prompt | llm_with_tools
            result = chain.invoke(state["messages"])

            # 处理结果
            future_influence_report = ""
            
            if len(result.tool_calls) == 0:
                future_influence_report = result.content if result.content else "未来影响力分析已完成，但未生成详细报告内容。"
                print(f"future_influence_report: {future_influence_report}")
            else:
                future_influence_report = "正在使用未来影响力分析工具进行深度评估..."

            return {
                "messages": [result],
                "future_influence_analysis_report": future_influence_report,
            }
            
        except Exception as e:
            error_message = f"未来影响力分析过程中发生错误: {str(e)}"
            print(f"Future influence agent error: {e}")
            
            return {
                "messages": [],
                "future_influence_analysis_report": error_message,
            }

    return future_influence_agent


def generate_future_influence_prompt_template():
    """
    生成未来影响力分析的提示词模板，可用于不同类型的项目
    
    Returns:
        Dict[str, str]: 不同类型项目的提示词模板
    """
    
    templates = {
        "基础研究": """
        ## 基础研究项目未来影响力分析指南
        
        ### 关键分析点：
        1. **理论突破潜力** - 评估理论创新的原创性和突破性
        2. **学科发展推动力** - 分析对学科发展的推动作用
        3. **跨学科影响** - 评估对其他学科的启发和影响
        4. **长期学术价值** - 预测长期的学术影响和引用潜力
        5. **应用转化可能性** - 分析基础研究向应用转化的可能性
        
        ### 评估维度：
        - 科学问题的重要性和前沿性
        - 研究方法的创新性和可靠性
        - 预期成果的原创性和影响力
        - 国际竞争力和领先性
        """,
        
        "应用研究": """
        ## 应用研究项目未来影响力分析指南
        
        ### 关键分析点：
        1. **技术成熟度** - 评估技术的发展阶段和成熟程度
        2. **市场应用前景** - 分析市场需求和应用潜力
        3. **产业化可能性** - 评估技术向产业转化的可行性
        4. **社会效益** - 分析对社会问题的解决能力
        5. **经济影响** - 评估潜在的经济效益和市场价值
        
        ### 评估维度：
        - 技术先进性和竞争优势
        - 应用场景的广泛性和需求强度
        - 产业化的技术门槛和市场壁垒
        - 政策支持和监管环境
        """,
        
        "跨学科研究": """
        ## 跨学科研究项目未来影响力分析指南
        
        ### 关键分析点：
        1. **学科融合创新** - 评估不同学科融合的创新潜力
        2. **综合解决方案** - 分析对复杂问题的综合解决能力
        3. **协同效应** - 评估跨学科合作产生的协同效应
        4. **新兴领域开拓** - 分析开创新兴交叉领域的可能性
        5. **多维度影响** - 评估在多个领域的同时影响
        
        ### 评估维度：
        - 学科交叉的深度和广度
        - 团队的跨学科合作能力
        - 方法论的创新性和适用性
        - 成果的多领域应用潜力
        """
    }
    
    return templates

# 辅助函数：生成未来影响力分析报告模板
def generate_influence_report_template():
    """
    生成未来影响力分析报告的模板
    
    Returns:
        str: 报告模板
    """
    
    template = """
# 项目未来影响力分析报告

## 1. 执行摘要
### 1.1 项目概述
- 项目名称：[项目名称]
- 研究领域：[主要研究领域]
- 研究阶段：[基础研究/应用研究/开发阶段]
- 预期周期：[项目周期]

### 1.2 影响力评估总结
- 整体影响力评分：[分数]/100
- 主要优势：[核心优势描述]
- 潜在风险：[主要风险点]
- 推荐等级：[优先/重要/一般]

## 2. 多维度影响力分析

### 2.1 学术影响力评估
- **研究趋势分析**
  - 领域热度评分：[分数]/100
  - 发展趋势：[上升/稳定/下降]
  - 国际竞争态势：[分析结果]

- **引用潜力预测**
  - 预期引用增长率：[百分比]
  - 学术声誉提升潜力：[高/中/低]
  - 重要性指标：[分析结果]

### 2.2 技术转化潜力
- **技术成熟度评估**
  - TRL等级：[1-9级]
  - 商业化时间线：[预计时间]
  - 市场准备度：[高/中/低]

- **产业应用前景**
  - 目标市场规模：[市场分析]
  - 应用场景：[具体场景]
  - 竞争优势：[技术优势]

### 2.3 社会影响力评估
- **社会问题解决能力**
  - 目标问题重要性：[高/中/低]
  - 解决方案有效性：[评估结果]
  - 受益人群规模：[预估数量]

- **政策相关性分析**
  - 国家战略契合度：[高度相关/相关/一般]
  - 政策支持可能性：[分析结果]
  - 资源获取便利性：[评估结果]

## 3. 风险评估与建议

### 3.1 主要风险识别
- **技术风险**：[具体风险点]
- **市场风险**：[市场变化风险]
- **竞争风险**：[竞争态势分析]
- **政策风险**：[政策变化影响]

### 3.2 发展建议
- **短期建议**（1-2年）：[具体建议]
- **中期规划**（3-5年）：[发展路径]
- **长期愿景**（5-10年）：[战略目标]

### 3.3 合作建议
- **学术合作**：[推荐合作机构]
- **产业合作**：[潜在合作伙伴]
- **国际合作**：[国际合作机会]

## 4. 结论与展望

### 4.1 综合评价
[基于多维度分析的综合评价]

### 4.2 投资价值判断
[从投资角度的价值判断]

### 4.3 发展前景展望
[对项目未来发展的展望]

---
*报告生成时间：[时间戳]*
*分析模型版本：[版本信息]*
"""
    
    return template
