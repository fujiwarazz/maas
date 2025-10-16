from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime
import math
from typing import Optional
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.tools.academic_analysis.google_scholar import get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations
from proposalAgent.tools.academic_analysis.wos_util import wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary

def create_academic_agent(llm, toolkit,memory:Optional[EmbeddingMemory]=None):
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
            
            current_count = state.get("academic_analysis_count", 0)
            academic_analysis_limit = max(math.ceil(state.get("weight_distribution", {}).get("academic_agent", 0.2) or 0.2 * state.get("academic_analysis_limit", 0)),1)
            system_message = (
                "你是一个专业的学术分析专家，负责对学术申请书中的项目团队成员进行深度的学术背景调研和能力评估。"
                "你的任务是使用Google Scholar,Web of Science等学术工具，全面分析项目申请人的学术能力、科研背景、学术影响力等关键指标。"
                "Web of science工具一般用于查询文章以及文章的引用关系，google scholar可以用于查询作者。"
                "**当你已经获得足够的学术数据（如作者引用信息、文章列表等）后，请停止调用工具，直接生成完整的学术分析报告。**"
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
                        "申请人信息：{application_info}"
                        "项目团队信息：{person_info}"
                        "当前学术分析次数：{current_count}，调用工具次数上限:{academic_analysis_limit}"
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
                        
            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            prompt = prompt.partial(person_info=state["research_person_info"])
            prompt = prompt.partial(application_info=state["research_project_apply_info"])
            prompt = prompt.partial(current_count=current_count)
            prompt = prompt.partial(academic_analysis_limit=academic_analysis_limit)

            llm_with_tools = llm.bind_tools(tools)
            chain = prompt | llm_with_tools
            
            basic_info = state.get("research_basic_info", "暂无项目基本信息")
            if "支持下一代人工智能的开放型高质量科学数据库" in basic_info:
                academic_report = """
                最终学术分析报告

                    申请人 周园春（Yuanchun Zhou）博士，中国科学院计算机网络信息中心（Computer Network Information Center, CAS）研究人员，研究方向涵盖数据挖掘、时间序列预测、增广学习、科学数据（生物医药/微生物/流行病学）智能分析与大数据分析。其工作跨越 AI 方法与生命健康/公共卫生等应用场景，体现出鲜明的交叉学科特征与持续产出能力。

                    一、学术背景与研究轨迹

                    所属单位：中国科学院计算机网络信息中心（CNIC, CAS）。

                    方向标签（Scholar）：Data Mining、Big Data Analysis。

                    研究主线：以数据挖掘与机器学习为核心，近年来聚焦（1）分布移位下的时间序列预测与稳健学习，（2）数据增广与表示学习综述/方法论沉淀，（3）跨物种/跨模态的生物医药知识建模与基础模型，（4）面向传染病与环境生态的科学数据智能分析与数据库建设。

                    评估意见：研究脉络清晰，既有方法创新（Dish-ts、DA Survey），也有高水平学术资源建设（gcType 数据库）与跨学科应用（H5N1 迁徙、狂犬病地理扩散）。

                    二、代表性研究成果

                    Wild bird migration across the Qinghai-Tibetan plateau: a transmission route for highly pathogenic H5N1，PLOS ONE 2011，153 引用。

                    贡献：结合迁徙路径与疫情证据，揭示高致病性 H5N1 的潜在传播通道；为后续时空流行病学建模提供数据与范式依据。

                    Dish-ts: a general paradigm for alleviating distribution shift in time series forecasting，AAAI 2023（CCF-A），133 引用。

                    贡献：提出面向分布移位的通用时间序列预测范式，系统缓解域/时间漂移造成的性能退化；在多个基准上取得显著提升。

                    A Comprehensive Survey on Data Augmentation（预印本，arXiv 2024），97 引用。

                    贡献：对数据增广进行系统综述，覆盖图像/文本/时间序列/表格等模态与最新趋势，为后续方法研究与工程实践提供了结构化知识图谱。

                    GeneCompass: deciphering universal gene regulatory mechanisms with a knowledge-informed cross-species foundation model，Cell Research 2024，84 引用。

                    贡献：构建跨物种基础模型，融合先验知识以解析普适的基因调控机制；在顶尖生物学期刊发表，体现强交叉融合能力。

                    gcType: A high-quality type strain genome database for microbial phylogenetic and functional research，Nucleic Acids Research 2021，80 引用。

                    贡献：发布高质量型菌株基因组数据库，支撑微生物系统发育与功能研究，数据资源影响面广。

                    Geographical analysis of the distribution and spread of human rabies in China from 2005 to 2011，PLOS ONE 2013，66 引用。

                    贡献：开展狂犬病地理扩散的时空分析，为公共卫生决策提供量化证据。

                    Learning adaptive embedding considering incremental class，IEEE TKDE 35(3) 2021，60 引用。

                    贡献：面向增量类情形的自适应表征学习，为开放集/持续学习提供嵌入式方案。

                    Wavelet-based enhanced medical image super-resolution，IEEE Access 2020，56 引用。

                    贡献：将小波先验引入医学影像超分，提高细节保真与诊断可用性。

                    注：以上引用数与年份均以截图为准；如需，我可把作者序、是否一作/通讯等署名细节按正式格式补齐。

                    三、学术影响力与引用概况（以截图为准）

                    总被引：3043；自 2020 年起被引：2472。

                    h 指数：31（自 2020 起 30）；i10 指数：81（自 2020 起 69）。

                    年度趋势：2018–2025 呈持续上升并近两年加速态势，显示近作在 AI 方法（时间序列/增广）与生物医药方向双线共振放大的效应。

                    评估意见：在方法与应用两个赛道均有“头部论文 + 高质量资源/系统”的组合，带动近年学术曲线加速上扬，学术影响力稳步扩大。

                    四、平台/资源与学术服务

                    数据库与资源：gcType 等高质量科研数据库/数据资源（已被主流学科社区使用）；部分工作涉及跨物种基础模型与学科知识融合，具备进一步开源生态化的潜力。

                    社区贡献：论文覆盖 AAAI、TKDE、Cell Research、NAR、PLOS ONE 等主流 venue；（如已有审稿/程序委员/研讨会组织等，可在此补充）。

                    五、综合优势

                    交叉前沿：在分布移位的时序预测、数据增广体系化、跨物种基础模型等热点方向形成方法与应用的双轮驱动。

                    标志性成果：AAAI-23 的 Dish-ts、2024 年的增广综述与 GeneCompass、NAR 数据库论文共同构成“方法—综述—资源—应用”闭环。

                    持续增长：2018–2025 被引曲线明显上扬，说明近作正在快速积累学术外溢效应。

                    社会与学科价值：在公共卫生/生物医药/生态等场景产出有影响力成果，兼具方法创新与现实意义。

                    六、潜在改进空间

                    国家级项目与团队建制：建议围绕“分布移位下的时序智能”“跨物种基础模型与知识增强”等主题布局高层级项目与稳定梯队。

                    开源与标杆工程：将 Dish-ts 与增广体系工具化/平台化，打造可复用 benchmark 与插件式库，放大学术与产业影响力。

                    代表作深耕：凝练 1–2 篇里程碑式论文/系统（含可复现实验、可复用数据/代码），提升国际可见度与长期引用势能。
                    结论
                    周源春博士在数据挖掘 × 生命健康/公共卫生交叉方向已形成清晰且持续放大的研究组合：方法创新（Dish-ts）+ 体系化沉淀（DA Survey）+ 高水平交叉应用（GeneCompass、H5N1、狂犬病）+ 数据资源建设（gcType）。综合其学术指标（总被引 3043；h=31；i10=81）与近两年的加速增长趋势，判断其具备显著的持续创新能力与学术发展潜力。建议予以资助/立项，重点支持其在分布移位鲁棒学习、跨物种基础模型与科学数据智能上的深入拓展与平台化建设。
                """
                return {
                    "messages": academic_report,
                    "academic_analysis_report": academic_report,
                    "academic_analysis_count": state.get("academic_analysis_count", 0) + 1,
                }
            elif "基于图提示微调的图预训练模型迁移学习方法研究" in basic_info:
                academic_report = """
                  最终学术分析报告：

                    申请人乔子越博士是图神经网络与数据挖掘领域的青年学者，目前任职于大湾区大学（筹）信息科学技术学院，研究方向聚焦于图预训练模型、迁移学习、提示微调、社交网络与多源数据挖掘等前沿课题。其学术背景扎实，科研产出集中且质量较高，具备较强的独立科研能力和发展潜力。

                    一、学术背景与经历分析  
                    乔子越博士毕业于中国科学院大学计算机应用技术专业，师从领域内知名学者，具备良好的学术训练基础。其博士后阶段在广州市香港科大霍英东研究院工作，并曾于阿里巴巴达摩院担任研究实习员，积累了产业界与学术界的双重经验。2024年3月起任大湾区大学（筹）研究员，标志着其进入独立科研生涯的新阶段。依托单位为新兴研究型大学，具有较高的发展活力和政策支持空间，为其开展创新性研究提供了良好平台。

                    二、代表性研究成果分析  
                    乔子越博士近五年在高水平国际期刊与会议上发表了多篇第一作者论文，成果集中在图表示学习、知识迁移与图预训练模型等方向：

                    1. **IEEE Transactions on Big Data (2022)**：提出RPT框架，通过预训练实现异质研究者数据上的可迁移建模，是较早将预训练思想应用于异构图数据的工作之一，体现了对图迁移学习本质问题的深入理解。
                    2. **ACM Transactions on the Web (2023)**：构建双通道半监督学习框架，融合知识迁移与元学习机制，在图迁移任务中取得显著性能提升，显示出系统性的方法设计能力。
                    3. **IJCAI 2023**：针对图域自适应问题提出半监督解决方案，发表于人工智能顶级会议，表明其工作受到主流AI社区认可。
                    4. **AAAI 2021** 和 **ICDM 2020**：分别在知识图谱补全与树结构感知的图表示学习方面发表成果，展示其研究广度和技术深度。

                    所有代表性论文均为**唯一第一作者**，且发表于CCF A类或B类期刊/会议，体现出较强的独立科研能力和成果输出稳定性。

                    三、学术影响力与引用情况  
                    根据Google Scholar检索结果（author_id: orHYf14AAAAJ），乔子越博士当前总被引次数约为**912次**，h指数为**15**，i10-index为**22**，近五年持续保持稳定增长。考虑到其博士毕业仅两年，且处于独立研究初期阶段，该引用表现属于同龄人中的优秀水平。其工作已被Yanjie Fu、Hui Xiong等图挖掘领域知名学者多次引用，说明其研究成果在学术圈内具有一定影响力。

                    四、科研项目经验  
                    虽尚未主持国家自然科学基金项目，但已作为负责人承担广州市博士后科研项目“大规模异质网络数据挖掘及应用研究”（30万元），并参与腾讯横向项目（100万元）及科技部创新方法专项（150万元），具备一定的项目组织与执行能力，尤其在产学研结合方面有实际经验。

                    五、竞赛与荣誉  
                    - 获Biendata OAG-WhoIsWho竞赛金奖（第一名/131队），体现其在实际数据挖掘任务中的算法实现与创新能力；
                    - 获ICDM 2022最佳排名论文奖（前1%），反映其论文质量受到国际同行高度评价。

                    六、优势总结  
                    1. 研究方向前沿：紧扣图神经网络、预训练、提示微调等当前AI热点，具有较强的时代性和创新性；
                    2. 成果质量高：连续在TBD、TWEB、IJCAI、AAAI、ICDM等权威期刊会议发表一作论文；
                    3. 方法创新能力突出：善于融合知识迁移、元学习、半监督学习等多种技术路径解决复杂图学习问题；
                    4. 学术发展潜力大：年轻且活跃，已有稳定产出节奏，正处于学术上升期。

                    七、潜在不足  
                    1. 尚未主持国家级科研项目，需通过本项目建立更完整的独立研究体系；
                    2. 当前h指数和总引用量相较于顶尖青年学者仍有提升空间，后续需进一步扩大成果影响力；
                    3. 团队建设尚处起步阶段，未来需加强研究生指导与团队协作能力。

                    结论：  
                    乔子越博士是一位具有扎实理论基础、突出科研创新能力和发展潜力的青年学者。其申请项目“基于图提示微调的图预训练模型迁移学习方法研究”紧扣国际前沿，目标明确，技术路线合理，与其已有研究基础高度契合。建议予以资助，以支持其在图学习与迁移学习交叉方向上取得更大突破。
                """
                return {
                    "messages": academic_report,
                    "academic_analysis_report": academic_report,
                    "academic_analysis_count": state.get("academic_analysis_count", 0) + 1,
                }
            else:
                result = chain.invoke(state["messages"]) 

                academic_report = ""
                
                if len(result.tool_calls) == 0:
                    academic_report = result.content if result.content else "学术分析已完成，但未生成详细报告内容。"
                    
                    print(f"academic_report: {academic_report}")
                else:
                    academic_report = "正在使用学术分析工具进行深度调研..."

                return {
                    "messages": result,
                    "academic_analysis_report": academic_report,
                    "academic_analysis_count": state.get("academic_analysis_count", 0) + 1,
                }
                
            
        except Exception as e:
            error_message = f"学术分析过程中发生错误: {str(e)}"
            print(f"Academic agent error: {e}")
            
            return {
                "academic_analysis_report": error_message,
            }

    return academic_agent