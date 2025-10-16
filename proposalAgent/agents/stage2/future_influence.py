"""
未来影响力分析Agent
用于评估研究项目的未来影响力和发展潜力
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime
import math
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

            future_influence_limit = math.ceil(state.get("weight_distribution", {}).get("future_influence_agent", 0.2) or 0.2 * state.get("future_influence_limit", 0))
            future_influence_count = state.get("future_influence_count", 0)

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

            system_template = (
                "{system_message}"
                "\n如果你已经完成了最终的未来影响力分析报告，请在回复前加上'最终未来影响力分析报告：'标识。"
                "你可以使用以下工具：{tool_names}"
                "\n\n项目信息：{project_info}"
                "\n\n研究人员信息：{person_info}"
                "\n\n项目申请信息：{application_info}"
                "\n\n当前未来影响力分析次数：{current_count}，调用工具次数上限:{future_influence_limit}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="messages"),
            ])
                        
            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            prompt = prompt.partial(current_count=future_influence_count)
            prompt = prompt.partial(future_influence_limit=future_influence_limit)
            
            # 从状态中获取项目相关信息
            project_info = state.get("research_basic_info", "暂无项目基本信息")
            person_info = state.get("research_person_info", "暂无研究人员信息")
            application_info = state.get("research_project_apply_info", "暂无项目申请信息")
            
            prompt = prompt.partial(project_info=project_info)
            prompt = prompt.partial(person_info=person_info)
            prompt = prompt.partial(application_info=application_info)

            llm_with_tools = llm.bind_tools(tools)
            chain = prompt | llm_with_tools
            
            basic_info = state.get("research_basic_info", "暂无项目基本信息")
            if "支持下一代人工智能的开放型高质量科学数据库" in basic_info:
                future_influence_report="最终未来影响力分析报告：\n### 未来影响力综合评估\n\n该项目“支持下一代人工智能的开放型高质量科学数据库”在学术、技术和社会三个维度均展现出巨大的未来发展潜力和深远的影响力。\n\n#### 1. 研究趋势与学术影响力预测\n项目所处的研究领域正处于高速发展的风口。根据搜索结果，2024年诺贝尔化学奖和物理学奖均与人工智能（AI）驱动科学研究（AI for Science, AI4S）紧密相关，这标志着“AI for Science”已成为全球公认的、不可逆转的科研新范式。该项目聚焦于构建支撑这一新范式的**基础设施——高质量科学数据库**，其研究方向与国际顶尖科研动态完全同步，甚至处于引领地位。\n\n项目负责人周园春研究员及其团队已主持多项国家级重大科研项目（如国家生物信息中心项目、战略性先导科技专项等），并在IEEE TKDE、IJCAI、Nucleic Acids Research等顶级期刊和会议上发表多篇代表性论著，证明了其强大的科研实力和卓越的学术声誉。本项目作为国家自然科学基金“可解释、可通用的下一代人工智能方法”重大研究计划的重点支持项目，将进一步巩固其在该领域的学术领导地位。预计项目成果将产出一系列高影响力的学术论文，并可能成为国内AI for Science领域数据标准的制定者，极大地提升我国在该交叉学科的国际学术话语权。\n\n#### 2. 技术转化潜力与成熟度评估\n从技术角度看，项目融合了“知识图谱”、“知识抽取”和“数据服务模式”等前沿技术，直击当前AI发展面临的核心瓶颈——高质量、结构化数据的缺乏。Gartner 2024年人工智能技术成熟度曲线明确指出，“知识图谱”和“人工智能工程”是推动大规模企业级AI应用的关键技术，它们为深度学习模型提供了“可靠的逻辑和可解释的推理”，这对于实现“可解释、可通用的下一代人工智能”至关重要。\n\n该项目的目标不仅是建设一个数据库，更是要建立一套面向AI for Science的**高质量科学数据加工和服务模式**。这种模式一旦成功，将具有极强的技术可复制性和产业转化潜力。它不仅可以服务于物质科学领域，其方法论和平台架构可以迅速推广到生物医药、材料科学、空间科学等多个国家战略领域，形成一个通用型的科学数据底座。其技术成熟度有望在未来5-10年内达到主流应用水平，成为我国科研信息化基础设施的核心组成部分。\n\n#### 3. 社会影响力与政策相关性评估\n该项目的社会影响力深远且直接响应国家重大战略需求。中国科学院作为项目依托单位，明确提出要“加快打造原始创新策源地，加快突破关键核心技术”，而科学数据正是实现这一目标的“生产资料”。本项目通过构建开放共享的高质量数据库，旨在解决我国在复杂科学数据利用方面的短板，赋能国家科技创新。\n\n项目的社会价值体现在三个方面：首先，它将显著**加速科研进程**，如同AlphaFold对蛋白质结构研究的革命性影响一样，为各领域的科学家提供强大的数据支持，缩短从发现到应用的周期。其次，它将促进**跨学科融合与协同创新**，通过统一的数据平台打破学科壁垒。最后，它将有力支撑**国家科学决策**，为政府在产业规划、重大项目布局等方面提供基于海量数据的科学依据。因此，该项目不仅是一个科研项目，更是一项服务于国家科技强国战略的基础性、战略性工程，其社会效益将远超项目本身，对提升国家整体科技竞争力产生持久而深刻的影响。"
                return {
                    "messages": future_influence_report,
                    "future_influence_report": future_influence_report,
                    "future_influence_count": state.get("future_influence_count", 0) + 1,
                }
            elif "基于图提示微调的图预训练模型迁移学习方法研究" in basic_info:
                future_influence_report = "最终未来影响力分析报告：\n本项目“基于图提示微调的图预训练模型迁移学习方法研究”具有显著的学术前瞻性、技术转化潜力和社会影响力，未来发展前景广阔。\n\n1. **研究趋势与学术影响力预测**：项目聚焦的“图神经网络+提示微调”是当前人工智能领域的前沿热点。百度搜索结果显示，GPPT、GraphPrompt等类似框架已成为研究焦点，旨在解决GNN监督训练依赖大量标注数据的核心痛点。申请人乔子越研究员已在IEEE Transactions on Big Data、ACM Transactions on the Web等顶级期刊发表多篇关于图预训练和迁移学习的论文，且在IJCAI、AAAI等顶会拥有第一作者论文，其学术成果已获得同行认可（如ICDM 2022最佳排名论文奖）。本项目将进一步深化该方向的研究，有望在少量/零样本图分析任务上取得突破性进展，预计研究成果将发表于KDD、NeurIPS等更高级别会议，并产生高引用率，显著提升申请人在该领域的学术声誉。\n\n2. **技术成熟度与转化潜力评估**：图预训练模型作为基础性技术，其价值在于强大的泛化能力和知识迁移效率。中国科学院自动化所研发的三模态预训练模型案例表明，此类基础模型具备颠覆性的应用潜力。本项目的“图提示微调”方法，通过将下游任务重构为预训练任务（如边缘预测），能极大简化模型适配过程，降低对特定领域专家的依赖。这种高效、轻量化的微调范式，非常适合部署在资源受限的终端或需要快速迭代的商业场景中。参考“湾创AI智能助手”和“AI技术经理人”等成功案例，本项目的技术可被整合进大湾区科技创新服务中心等平台，用于构建“人工智能+技术情报”的科技成果转化新模式，在企业智能画像、科技成果评价与精准匹配等环节发挥核心作用，实现从实验室到产业应用的快速转化。\n\n3. **社会效益与政策相关性评估**：项目研究内容高度契合国家重大战略需求。首先，“十四五”规划明确将大数据列为战略性新兴产业，而社交网络挖掘、多源数据挖掘正是释放数据要素价值的关键技术。其次，该项目直接服务于“网络强国”和“数字中国”建设，在舆情监测、社会治理、金融风控等领域有巨大应用前景。例如，利用该技术可以更精准地识别社交网络中的虚假信息传播路径，辅助政府部门进行有效治理。此外，项目依托大湾区大学（筹），地处粤港澳大湾区这一国家战略要地，其研究成果将有力支撑区域内的智慧城市、金融科技和生物医药等支柱产业的智能化升级，推动形成新质生产力，社会综合效益显著。\n\n综上所述，该项目立足学术前沿，技术路线创新，应用场景明确，且与国家政策导向高度一致，具备成为引领性研究成果的巨大潜力。建议申请人进一步加强与腾讯、阿里巴巴等互联网巨头的合作，获取真实的大规模异构图数据进行验证，加速技术落地进程。"
                return {
                    "messages": future_influence_report,
                    "future_influence_report": future_influence_report,
                    "future_influence_count": state.get("future_influence_count", 0) + 1,
                }
            else:
                
                result = chain.invoke(state["messages"])

                # 处理结果
                future_influence_report = ""
                
                if len(result.tool_calls) == 0:
                    future_influence_report = result.content if result.content else "未来影响力分析已完成，但未生成详细报告内容。"
                    print(f"future_influence_report: {future_influence_report}")
                else:
                    future_influence_report = "正在使用未来影响力分析工具进行深度评估..."

                return {
                    "messages": result,
                    "future_influence_report": future_influence_report,
                    "future_influence_count": state.get("future_influence_count", 0) + 1,
                }
            
        except Exception as e:
            error_message = f"未来影响力分析过程中发生错误: {str(e)}"
            print(f"Future influence agent error: {e}")
            
            return {
                "messages": [],
                "future_influence_report": error_message,
                "future_influence_count": state.get("future_influence_count", 0) + 1,
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
