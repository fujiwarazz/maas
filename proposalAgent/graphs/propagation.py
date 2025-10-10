# TradingAgents/graph/propagation.py

# 导入必要的类型提示，增强代码的可读性和健壮性
from typing import Dict, Any
from datetime import datetime
# 从项目内部导入定义好的状态类，确保数据结构的一致性
from proposalAgent.agents.utils.agent_states import (
    AgentState,
    DebateState,
)
from typing import List

class Propagator:
    """
    处理状态的初始化以及在图（Graph）中的传播。
    这个类是一个辅助工具，主要负责准备图运行所需的初始数据结构和配置参数。
    它将图的“启动”逻辑与图本身的结构定义分离开来，使代码更清晰、更易于管理。
    """

    def __init__(self, max_recur_limit=100):
        """
        使用配置参数初始化传播器。
        
        Args:
            max_recur_limit (int, optional): 设置图的最大递归深度限制。
                                             这是一个安全机制，用于防止图中出现无限循环，
                                             导致程序崩溃或资源耗尽。默认为 100 次。
        """
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, user_prompt: str, user_interest: List[str], filepath: str = ""
    ) -> Dict[str, Any]:
        """
        为项目评估代理图（Agent Graph）创建一个初始状态字典。
        这个函数就像一个"重置按钮"，确保每次启动一个新的项目评估流程时，
        所有的状态变量都被设置到一个干净、预定义的初始值。

        Args:
            user_prompt (str): 用户输入的项目评估请求或指令。
            user_interest (List[str]): 用户关注的评估重点列表。
            filepath (str, optional): 项目文档的文件路径。

        Returns:
            Dict[str, Any]: 一个符合 `AgentState` 结构的字典，作为图的起始输入。
        """
        from langchain_core.messages import HumanMessage
        
        # 构建初始消息
        prompt_content = f"""项目评估请求：{user_prompt}

                用户关注的评估重点：{', '.join(user_interest) if user_interest else '全面评估'}

                文件路径：{filepath if filepath else '未提供'}

                请对此项目进行全面的多维度评估分析。"""
        
        return {
            # 消息历史 - 使用正确的消息格式
            "messages": [HumanMessage(content=prompt_content)],

            # 基本信息字段
            "filepath": filepath,
            "research_topic": user_interest,
            "intention_decision": "",

            # 项目结构化信息（Stage 1 输出占位）
            "research_structure": "",
            "research_person_info": "",
            "research_basic_info": "",
            "research_project_team_info": "",
            "research_project_apply_info": "",
            "research_report_body_summary": "",

            # 权重分布 & 节点迭代配置
            "weight_distribution": {
                "academic_agent": 0.3,
                "future_influence_agent": 0.3,
                "interdisciplinary_agent": 0.2,
                "debate_agent": 0.2,
            },
            "academic_analysis_limit": 3,
            "academic_analysis_count": 0,
            "social_analysis_limit": 2,
            "social_analysis_count": 0,
            "future_influence_limit": 3,
            "future_influence_count": 0,

            # 各维度分析报告（Stage 2 输出占位）
            "academic_analysis_report": "",
            "social_analysis_report": "",
            "future_influence_report": "",

            # 跨学科与辩论（Stage 2/辩论阶段）
            "interdisciplinary_results": [],
            "current_discipline": None,
            "debate_results": {},

            # Stage 3: 综合分析与输出
            "final_analysis_summary": "",
            "completeness_check_result": {},
            "is_analysis_complete": None,
            "is_analysis_consistent": None,
            "completeness_recommendation": None,
            "skip_human_review": None,
            "reflection_decision": None,
            "human_feedback": "",
            "feedback_analysis_result": {},
            "feedback_routing_decision": None,
            "feedback_instructions": "",
            "feedback_target": None,
            "feedback_pending": False,
            "final_report": "",
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """
        获取用于调用（invoke）项目评估图的参数。
        这个函数将一些通用的、与图运行机制相关的配置打包起来，
        方便在调用图时直接传入。

        Returns:
            Dict[str, Any]: 一个包含图调用所需配置的字典。
        """
        return {
           
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }

    def create_evaluation_context(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        为项目评估创建上下文信息。
        
        Args:
            project_info (Dict[str, Any]): 包含项目基本信息的字典
            
        Returns:
            Dict[str, Any]: 格式化的评估上下文
        """
        return {
            "evaluation_timestamp": str(datetime.now()),
            "project_metadata": {
                "title": project_info.get("title", "未提供"),
                "category": project_info.get("category", "未分类"),
                "funding_amount": project_info.get("funding_amount", "未提供"),
                "duration": project_info.get("duration", "未提供"),
                "keywords": project_info.get("keywords", [])
            },
            "evaluation_criteria": {
                "academic_weight": 0.25,
                "social_impact_weight": 0.20,
                "future_potential_weight": 0.20,
                "feasibility_weight": 0.20,
                "innovation_weight": 0.15
            }
        }

    def validate_state_completeness(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证状态的完整性，返回缺失字段的报告。
        
        Args:
            state (Dict[str, Any]): 当前状态字典
            
        Returns:
            Dict[str, Any]: 包含验证结果的字典
        """
        required_fields = [
            "research_topic",
            "academic_analysis_report", 
            "social_analysis_report",
            "future_influence_report",
            "interdisciplinary_results",
            "debate_results",
            "final_analysis_summary"
        ]
        
        missing_fields = []
        incomplete_fields = []
        
        for field in required_fields:
            if field not in state:
                missing_fields.append(field)
            elif not state[field] or state[field] == "":
                incomplete_fields.append(field)
        
        return {
            "is_complete": len(missing_fields) == 0 and len(incomplete_fields) == 0,
            "missing_fields": missing_fields,
            "incomplete_fields": incomplete_fields,
            "completion_rate": (len(required_fields) - len(missing_fields) - len(incomplete_fields)) / len(required_fields)
        }

    def extract_evaluation_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        从状态中提取评估摘要信息。
        
        Args:
            state (Dict[str, Any]): 完整的状态字典
            
        Returns:
            Dict[str, Any]: 评估摘要字典
        """
        return {
            "project_title": state.get("research_topic", ["未知项目"])[0] if isinstance(state.get("research_topic"), list) else state.get("research_topic", "未知项目"),
            "evaluation_status": "已完成" if state.get("final_report") else "进行中",
            "analysis_dimensions": {
                "academic": "已完成" if state.get("academic_analysis_report") else "未完成",
                "social": "已完成" if state.get("social_analysis_report") else "未完成", 
                "future_impact": "已完成" if state.get("future_influence_report") else "未完成",
                "interdisciplinary": "已完成" if state.get("interdisciplinary_results") else "未完成"
            },
            "debate_status": {
                "conducted": bool(state.get("debate_results")),
                "disciplines_count": len(state.get("interdisciplinary_results", []))
            },
            "human_review": {
                "required": not state.get("skip_human_review", True),
                "completed": bool(state.get("human_feedback")),
                "feedback_analyzed": bool(state.get("feedback_analysis_result"))
            },
            "final_output": {
                "analysis_completed": bool(state.get("final_analysis_summary")),
                "report_generated": bool(state.get("final_report"))
            }
        }