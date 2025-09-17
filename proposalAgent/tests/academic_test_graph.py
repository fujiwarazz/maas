"""
简化的学术分析图，只包含academic node和tool node
用于测试学术分析功能
"""

from typing import Dict, Any, List, Literal
from langgraph.cache import base
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
import sys
import pathlib
from pydantic import SecretStr
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.stage2.academic import create_academic_agent
from proposalAgent.tools.academic_analysis.google_scholar import (
    get_article_brief, 
    resolve_author_candidates, 
    get_author_citations, 
    get_author_citations_auto, 
    get_author_articles_citations
)


class AcademicTestGraph:
    """简化的学术分析测试图类"""
    
    def __init__(self, llm):
        """
        初始化学术测试图
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.tools = [
            get_article_brief, 
            resolve_author_candidates, 
            get_author_citations, 
            get_author_citations_auto, 
            get_author_articles_citations
        ]
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """构建图结构"""
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 创建节点
        academic_agent = create_academic_agent(self.llm, {})
        tool_node = ToolNode(self.tools)
        
        # 添加节点
        workflow.add_node("academic_agent", academic_agent)
        workflow.add_node("tools", tool_node)
        
        # 设置入口点
        workflow.set_entry_point("academic_agent")
        
        # 添加条件边
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
        
        # 编译图
        self.graph = workflow.compile()
    
    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """
        决定是否继续执行工具
        
        Args:
            state: 当前状态
            
        Returns:
            下一步动作: "tools" 或 "end"
        """
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # 检查是否有工具调用
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行学术分析图
        
        Args:
            initial_state: 初始状态
            
        Returns:
            最终状态
        """
        if not self.graph:
            raise ValueError("图未正确构建")
        
        if "messages" not in initial_state:
            initial_state["messages"] = []
        
        if not initial_state["messages"]:
            person_info = initial_state.get("research_person_info", "未提供")
            
            initial_message = f"""
            请对以下申请人进行学术分析：
                    申请人信息：{person_info}
            
            请使用相关工具进行深入的学术背景调研。
            """
            
            initial_state["messages"] = [HumanMessage(content=initial_message)]
        
        # 运行图
        final_state = self.graph.invoke(initial_state)
        return final_state
    
    def stream_run(self, initial_state: Dict[str, Any]):
        """
        流式运行学术分析图
        
        Args:
            initial_state: 初始状态
            
        Yields:
            状态更新
        """
        if not self.graph:
            raise ValueError("图未正确构建")
        
        # 确保有messages字段
        if "messages" not in initial_state:
            initial_state["messages"] = []
        
        if not initial_state["messages"]:
            research_topic = initial_state.get("research_topic", ["未指定"])
            person_info = initial_state.get("research_person_info", "未提供")
            
            initial_message = f"""
            请对以下申请人进行学术分析：
            
            研究领域：{research_topic if isinstance(research_topic, str) else ', '.join(research_topic)}
            申请人信息：{person_info}
            
            请使用相关工具进行深入的学术背景调研。
            """
            
            initial_state["messages"] = [HumanMessage(content=initial_message)]
        
        # 流式运行图
        for step in self.graph.stream(initial_state):
            yield step


def create_academic_test_graph(llm):
    """
    创建学术分析测试图的便捷函数
    
    Args:
        llm: 语言模型实例
        
    Returns:
        AcademicTestGraph实例
    """
    return AcademicTestGraph(llm)


# 测试示例
def test_academic_graph_example():
    """学术分析图测试示例"""
    from langchain_openai import ChatOpenAI
    import os
    
    print("=== 学术分析图测试示例 ===")
    
    # 创建LLM（需要设置API key）
    try:
        llm = ChatOpenAI(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-0e349a8dc24443988825b69a56d2b868"
        )
        
        test_graph = create_academic_test_graph(llm)
        
        temp_info = """
### **申请人的个人履历 (杜一)**

*   **姓名**: 杜一 [第2页]
*   **性别**: 男 [第2页]
*   **出生年月**: 1988年03月 [第2页]
*   **民族**: 汉族 [第2页]
*   **学位**: 博士 [第2页]
*   **职称**: 研究员 [第2页]
*   **是否在站博士后**: 否 [第2页]
*   **电子邮箱**: duyi@cnic.cn [第2页]
*   **国别或地区**: 中国 [第2页]
*   **申请人类别**: 依托单位全职 [第2页]
*   **工作单位**: 中国科学院计算机网络信息中心 [第2页]
*   **主要研究领域**: 科技大数据知识图谱 [第2页]
*   **教育经历**:
    *   2008-09至2013-07, 中国科学院软件研究所, 计算机应用技术, 博士 [第47页]
    *   2004-09至2008-06, 山东大学, 软件工程, 学士 [第47页]
*   **博士后工作经历**: 无 [第47页]
*   **科研与学术工作经历**:
    *   2021-12至今, 中国科学院计算机网络信息中心, 大数据应用发展部, 研究员 [第47页]
    *   2021-02至2022-02, 国家自然科学基金委员会, 交叉科学部, 无 [第47页]
    *   2015-12至2021-12, 中国科学院计算机网络信息中心, 大数据应用发展部, 副研究员 [第47页]
    *   2013-07至2015-12, 中国科学院计算机网络信息中心, 科学数据中心, 助理研究员 [第47页]
*   **近五年主持或参加的国家自然科学基金项目/课题**:
    *   国家自然科学基金委员会, 优秀青年科学基金项目, T2322027, 科技大数据知识图谱, 200万元, 在研, 主持 [第47页]
    *   国家自然科学基金委员会, 专项项目, L1924075, 国家自然科学基金成果开放共享政策与平台架构设计研究, 40万元, 结题, 主持 [第47页]
    *   国家自然科学基金委员会, 重点项目, 61836013, 面向领域大数据的知识图谱构建, 288万元, 结题, 参与 [第47页]
*   **近五年主持或参加的其他科研项目/课题**:
    *   中国科学院学部工作局, 专项项目, E42Q2302, 学部增选专家库与专家指派系统, 253万元, 在研, 主持 [第47页]
    *   科技部, 重点研发青年科学家项目, 2022YFF0712200, 基于领域知识图谱的光电催化材料挖掘软件, 200万元, 在研, 主持 [第47页]
    *   中国科学院, 院级人才, 2021166, 青年创新促进会, 80万元, 在研, 主持 [第47页]
    *   国家自然科学基金委员会, 委托课题, TC200E024, 国家自然科学基金大数据知识管理服务平台, 478万元, 在研, 主持 [第47页]
    *   中国科学院学部工作局, 中科院专项, E2292304, 科技智库人才系统, 300万元, 结题, 主持 [第47页]
    *   中国科学院国际合作局, 中科院专项, 292021000153, 国际合作知识管理与智能化服务平台, 125万元, 结题, 主持 [第47页]
    *   JKW, 创新特区项目, 1912-2, 基于XX知识图谱的推荐方法研究, 250万元, 结题, 主持 [第47页]
*   **代表性研究成果和学术奖励**:
    *   **代表性论著**:
        *   Meng Xiao et al. (2024). "Interdisciplinary Fairness in Imbalanced Research Proposal Topic Inference: A Hierarchical Transformer-based Method with Selective Interpolation." *ACM Transactions on Knowledge Discovery from Data*, 6(1). (本人标注: 共同通讯作者) [第48页]
    *   **论著之外的代表性研究成果**:
        *   杜一 et al. (2022). "一种基于网络表征和语义表征的同名作者消歧方法." (专利) [第48页]
        *   杜一 et al. (2022). "一种基于LightGBM分类与表示学习的姓名消歧方法和系统." (专利) [第48页]
        *   杜一 et al. (2022). "基于图局部结构和文本语义相似性的学术论文推荐方法." (专利) [第48页]
        *   杜一 et al. (2022). "一种科技资源汇聚与持续服务方法及装置." (专利) [第48页]
        *   杜一 et al. (2022). "一种可利用专家知识的申请书多标签层次分类方法." (专利) [第48页]
        *   杜一 et al. (2022). "一种基于异质图卷积神经网络嵌入的作者名字消歧方法." (专利) [第48页]
        *   杜一 et al. (2022). "一种基于作者著作树和图神经网络的论文合作者推荐方法." (专利) [第49页]
        *   杜一 et al. (2021). "无监督的基于表示学习的同名作者消歧方法及装置." (专利) [第49页]
        *   杜一 et al. (2021). "无监督的基于表示学习的同名作者消歧方法及装置." (专利) [第49页]

"""
        test_state = {
            "research_person_info": temp_info,
            "messages": [{"role":"user","content":"请对以下申请人进行学术分析"}]
        }
        
        print("初始状态:")
        print(f"申请人信息: {test_state['research_person_info']}")
        
        print("\n开始流式执行:")
        for i, step in enumerate(test_graph.stream_run(test_state)):
            print(f"\n{'='*50} 步骤 {i+1} {'='*50}")
            for node_name, node_output in step.items():
                print(f"\n🔹 节点 '{node_name}' 输出:")
                print(f"   输出键: {list(node_output.keys())}")
                
                # 显示学术分析报告
                if "academic_analysis_report" in node_output:
                    report = node_output['academic_analysis_report']
                    print(f"\n📊 学术分析报告:")
                    print(f"{report}")
                
                # 显示消息内容
                if "messages" in node_output:
                    messages = node_output['messages']
                    print(f"\n💬 消息内容 (共{len(messages)}条):")
                    for j, msg in enumerate(messages):
                        print(f"   消息 {j+1}:")
                        print(f"   类型: {type(msg).__name__}")
                        
                        # 显示消息内容
                        if hasattr(msg, 'content') and msg.content:
                            content = str(msg.content)[:300] + "..." if len(str(msg.content)) > 300 else str(msg.content)
                            print(f"   内容: {content}")
                        
                        # 显示工具调用信息
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"   🔧 工具调用 (共{len(msg.tool_calls)}个):")
                            for k, tool_call in enumerate(msg.tool_calls):
                                print(f"      工具调用 {k+1}:")
                                print(f"      - 工具名: {tool_call.get('name', 'Unknown')}")
                                print(f"      - 调用ID: {tool_call.get('id', 'Unknown')}")
                                if 'args' in tool_call:
                                    print(f"      - 参数: {tool_call['args']}")
                        
                        # 显示工具消息内容
                        if hasattr(msg, 'name') and hasattr(msg, 'content'):
                            print(f"   🛠️  工具响应:")
                            print(f"   工具名: {msg.name}")
                            tool_content = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
                            print(f"   响应内容: {tool_content}")
                        
                        print("   " + "-"*60)
                
                print("\n" + "="*100)
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("这可能是由于缺少API密钥或网络问题导致的")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_academic_graph_example()
