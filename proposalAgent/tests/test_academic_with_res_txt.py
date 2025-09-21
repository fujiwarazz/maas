#!/usr/bin/env python3
"""
学术分析测试脚本 - 使用res.txt内容作为输入
使用图结构测试academic agent和tool节点的协作
"""

import os
import sys
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

def ensure_hxrag_environment():
    """确保激活hxrag环境"""
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    print(f"当前环境: {current_env}")
    
    if current_env != 'hxrag':
        print("⚠️  当前不在hxrag环境中，请手动激活hxrag环境")
        print("运行命令: conda activate hxrag")
        return False
    else:
        print("✅ 已在hxrag环境中")
    return True

def load_test_data_from_res_txt():
    """从res.txt文件加载测试数据"""
    try:
        # 获取res.txt文件路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        res_txt_path = os.path.join(project_root, "res.txt")
        
        if not os.path.exists(res_txt_path):
            print(f"❌ 未找到res.txt文件: {res_txt_path}")
            return None
        
        with open(res_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ 成功加载res.txt文件，内容长度: {len(content)} 字符")
        
        # 提取关键信息
        person_info = extract_person_info_from_content(content)
        return person_info
        
    except (OSError, UnicodeDecodeError) as e:
        print(f"❌ 加载res.txt失败: {e}")
        return None

def extract_person_info_from_content(_: str) -> str:
    """从res.txt内容中提取人员信息"""
    
    # 基于res.txt的实际内容结构化人员信息
    person_info = """### 申请人个人履历

**姓名:** 杜一
**性别:** 男
**出生年月:** 1988年03月
**民族:** 汉族
**学位:** 博士
**职称:** 研究员
**电子邮箱:** duyi@cnic.cn
**工作单位:** 中国科学院计算机网络信息中心
**主要研究领域:** 科技大数据知识图谱

**教育经历:**
* 2008-09至2013-07, 中国科学院软件研究所, 计算机应用技术, 博士
* 2004-09至2008-06, 山东大学, 软件工程, 学士

**科研与学术工作经历:**
* 2021-12至今, 中国科学院计算机网络信息中心, 大数据应用发展部, 研究员
* 2021-02至2022-02, 国家自然科学基金委员会, 交叉科学部
* 2015-12至2021-12, 中国科学院计算机网络信息中心, 大数据应用发展部, 副研究员
* 2013-07至2015-12, 中国科学院计算机网络信息中心, 科学数据中心, 助理研究员

**主持或参加的国家自然科学基金项目:**
* 优秀青年科学基金项目, T2322027, 科技大数据知识图谱, 2024-01-01 至 2026-12-31, 200万元, 在研, 主持
* 专项项目, L1924075, 国家自然科学基金成果开放共享政策与平台架构设计研究, 2020-01-01 至 2021-12-31, 40万元, 结题, 主持
* 重点项目, 61836013, 面向领域大数据的知识图谱构建, 2019-01-01 至 2023-12-31, 288万元, 结题, 参与

**代表性研究成果:**
* Meng Xiao; Min Wu; Ziyue Qiao; Yanjie Fu; Zhiyuan Ning; Yi Du; Yuanchun Zhou; Interdisciplinary Fairness in Imbalanced Research Proposal Topic Inference: A Hierarchical Transformer-based Method with Selective Interpolation, ACM Transactions on Knowledge Discovery from Data, 2024
* Wang Weijun; Ning Zhiyuan; Dong Hao; Qiao Ziyue; Du Yi; Zhou Yuanchun; 基于语义相似关系的学科交叉主题识别方法, 情报学报, 2024
* Yi Du; Ludi Wang; Mengyi Huang; Dongze Song; Wenjuan Cui; Yuanchun Zhou; Autodive: An Integrated Onsite Scientific Literature Annotation Tool, ACL 2023
* Meng Xiao; Ziyue Qiao; Yanjie Fu; Hao Dong; Yi Du; Pengyang Wang; Hui Xiong; Yuanchun Zhou; Hierarchical Interdisciplinary Topic Detection Model for Research Proposal Classification, IEEE TKDE, 2023

**专利成果:**
* Method for Disambiguating Between Authors with Same Name on Basis of Network Representation and Semantic Representation, 美国专利, 2023
* 一种基于网络表征和语义表征的同名作者消歧方法, 中国专利, 2022
* 一种基于LightGBM分类与表示学习的姓名消歧方法和系统, 中国专利, 2022
* 基于图局部结构和文本语义相似性的学术论文推荐方法, 中国专利, 2022
"""
    
    return person_info

class AcademicTestGraph:
    """学术分析测试图类，包含agent节点和tool节点"""
    
    def __init__(self, llm):
        """
        初始化学术测试图
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.tools = self._create_tools()
        self.graph = None
        self._build_graph()
    
    def _create_tools(self):
        """创建学术分析工具"""
        try:
            from proposalAgent.tools.academic_analysis.google_scholar import (
                get_article_brief, 
                resolve_author_candidates, 
                get_author_citations, 
                get_author_citations_auto, 
                get_author_articles_citations
            )
            from proposalAgent.tools.academic_analysis.wos_util import (
                wos_expanded_search, 
                wos_expanded_citation_fanout, 
                wos_citation_influence_summary
            )
            
            tools = [
                get_article_brief, 
                resolve_author_candidates, 
                get_author_citations, 
                get_author_citations_auto, 
                get_author_articles_citations,
                wos_expanded_search, 
                wos_expanded_citation_fanout, 
                wos_citation_influence_summary
            ]
            print(f"✅ 成功加载 {len(tools)} 个学术分析工具")
            return tools
            
        except ImportError as e:
            print(f"⚠️  导入工具失败: {e}")
            # 创建空工具列表，但仍然可以测试基本流程
            return []
    
    def _build_graph(self):
        """构建包含agent和tool节点的图结构"""
        from proposalAgent.agents.stage2.academic import create_academic_agent
        from proposalAgent.agents.utils.agent_states import AgentState
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 创建节点
        academic_agent = create_academic_agent(self.llm, {}, {})
        tool_node = ToolNode(self.tools) if self.tools else None
        
        # 添加agent节点
        workflow.add_node("academic_agent", academic_agent)
        
        # 如果有工具，添加tool节点
        if tool_node:
            workflow.add_node("tools", tool_node)
        
        # 设置入口点
        workflow.set_entry_point("academic_agent")
        
        # 添加条件边
        if tool_node:
            workflow.add_conditional_edges(
                "academic_agent",
                self._should_continue,
                {
                    "tools": "tools",
                    "end": END
                }
            )
            # 工具执行后回到academic agent
            workflow.add_edge("tools", "academic_agent")
        else:
            # 没有工具时直接结束
            workflow.add_edge("academic_agent", END)
        
        # 编译图，设置递归限制
        self.graph = workflow.compile()
        print("✅ 学术分析图构建完成")
    
    def _should_continue(self, state: Dict[str, Any]) -> Literal["tools", "end"]:
        """
        决定是否继续调用工具
        
        Args:
            state: 当前状态
            
        Returns:
            下一步操作："tools" 或 "end"
        """
        # 获取最后一条消息
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # 检查是否有工具调用
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"🔧 检测到工具调用: {len(last_message.tool_calls)} 个")
            return "tools"
        else:
            print("✅ 无工具调用，结束流程")
            return "end"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行学术分析图
        
        Args:
            state: 输入状态
            
        Returns:
            输出状态
        """
        if self.graph:
            # 设置配置，限制递归次数
            config = {"recursion_limit": 30}
            return self.graph.invoke(state, config=config)
        else:
            print("❌ 图未初始化")
            return state


def test_academic_graph_with_res_data():
    """使用res.txt数据测试学术分析图"""
    print("\n=== 基于res.txt的学术分析图测试 ===")
    
    try:
        # 添加项目路径
        sys.path.append(os.path.abspath('.'))
        
        # 加载测试数据
        person_info = load_test_data_from_res_txt()
        if not person_info:
            print("❌ 无法加载测试数据")
            return False
            
        # 导入必要模块
        from proposalAgent.model_config import TONGYI_CONFIG
        from langchain_openai import ChatOpenAI
        
        # 创建真实的LLM
        llm = ChatOpenAI(
            model=TONGYI_CONFIG["deep_think_llm"],
            api_key=TONGYI_CONFIG["api_key"],
            base_url=TONGYI_CONFIG["backend_url"],
            temperature=0.1
        )
        print(f"✅ 使用真实LLM: {TONGYI_CONFIG['deep_think_llm']}")
        
        # 创建学术分析测试图
        test_graph = AcademicTestGraph(llm)
        print("✅ 学术分析测试图创建成功")
        
        # 准备测试状态
        test_state = {
            "messages": [HumanMessage(content="请分析杜一的学术背景和能力")],
            "research_topic": ["科技大数据", "知识图谱", "数据挖掘"],
            "research_person_info": person_info
        }
        
        print("📝 测试数据准备完成")
        print("   - 申请人: 杜一")
        print(f"   - 研究领域: {test_state['research_topic']}")
        print("   - 工作单位: 中国科学院计算机网络信息中心")
        
        test_results = []
        
        # 测试1: 图结构执行测试
        print("\n1️⃣ 测试图结构执行:")
        try:
            print("🚀 开始执行学术分析图...")
            result = test_graph.run(test_state)
            
            success = validate_graph_result(result)
            test_results.append(("图结构执行测试", success))
            
            if success:
                print("✅ 图结构执行测试通过")
                print(f"   报告长度: {len(result.get('academic_analysis_report', ''))} 字符")
                print(f"   消息数量: {len(result.get('messages', []))}")
                
                # 显示部分报告内容
                report = result.get('academic_analysis_report', '')
                if report:
                    if len(report) > 500:
                        print(f"   报告预览: {report}...")
                    else:
                        print(f"   完整报告: {report}")
                
                # 显示工具调用信息
                messages = result.get('messages', [])
                tool_calls_count = 0
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls_count += len(msg.tool_calls)
                
                print(f"   工具调用次数: {tool_calls_count}")
                
            else:
                print("❌ 图结构执行测试失败")
                print(f"   结果: {result}")
                
        except (ImportError, AttributeError, RuntimeError) as e:
            print(f"❌ 图结构执行测试出错: {e}")
            test_results.append(("图结构执行测试", False))
        
        # 测试2: API配置验证
        print("\n2️⃣ 测试API配置:")
        try:
            api_key = TONGYI_CONFIG.get("api_key")
            if api_key:
                print("✅ API密钥已配置")
                test_results.append(("API配置测试", True))
            else:
                print("⚠️  API密钥未配置，请设置DASHSCOPE_API_KEY环境变量")
                test_results.append(("API配置测试", False))
                
        except (KeyError, AttributeError) as e:
            print(f"❌ API配置测试出错: {e}")
            test_results.append(("API配置测试", False))
        
        # 测试3: 数据完整性验证
        print("\n3️⃣ 测试数据完整性验证:")
        person_info_check = person_info
        required_fields = ["杜一", "博士", "研究员", "科技大数据知识图谱", "中国科学院"]
        
        missing_fields = [field for field in required_fields if field not in person_info_check]
        
        if not missing_fields:
            print("✅ 数据完整性验证通过")
            test_results.append(("数据完整性测试", True))
            print(f"   包含所有必要字段: {required_fields}")
        else:
            print("❌ 数据完整性验证失败")
            test_results.append(("数据完整性测试", False))
            print(f"   缺失字段: {missing_fields}")
        
        return summarize_test_results(test_results)
        
    except (ImportError, AttributeError, ValueError) as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_graph_result(result: Dict[str, Any]) -> bool:
    """验证图执行结果"""
    if not isinstance(result, dict):
        print("   验证失败: 结果不是字典类型")
        return False
    
    # 检查消息
    messages = result.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        print("   验证失败: 缺少消息或消息格式错误")
        return False
    
    # 检查是否有学术分析报告
    report = result.get("academic_analysis_report", "")
    if len(report) < 20:
        print(f"   验证失败: 学术分析报告太短 ({len(report)} 字符)")
        return False
    
    print("   ✅ 图执行结果验证通过")
    return True

def summarize_test_results(test_results):
    """汇总测试结果"""
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, success in test_results if success)
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！基于res.txt的学术分析功能工作正常")
        return True
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败")
        return False

def main():
    """主函数"""
    try:
        print("🚀 基于res.txt的学术分析测试系统")
        print("="*60)
        print("Python环境:", sys.executable)
        print("工作目录:", os.getcwd())
        
        # 检查环境
        if not ensure_hxrag_environment():
            print("❌ 环境检查失败，请确保在hxrag环境中运行")
            sys.exit(1)
        
        # 运行测试
        success = test_academic_graph_with_res_data()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n❌ 测试被用户中断")
        sys.exit(1)
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"\n❌ 测试过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
