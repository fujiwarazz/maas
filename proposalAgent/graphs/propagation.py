# TradingAgents/graph/propagation.py

# 导入必要的类型提示，增强代码的可读性和健壮性
from typing import Dict, Any
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
        self, user_prompt: str,user_interest:List[str],
    ) -> Dict[str, Any]:
        """
        为代理图（Agent Graph）创建一个初始状态字典。
        这个函数就像一个“重置按钮”，确保每次启动一个新的决策流程时，
        所有的状态变量都被设置到一个干净、预定义的初始值。

        Args:
            user_prompt (str): 用户输入的问题或指令。
            user_interest (List[str]): 用户的兴趣列表。

        Returns:
            Dict[str, Any]: 一个符合 `AgentState` 结构的字典，作为图的起始输入。
        """
        prompt = f"""
            用户输入：{user_prompt}
            用户侧重的分析重点：{', '.join(user_interest)}
            """
        
        return {
            
            "messages": [("human", prompt.format(user_prompt=user_prompt,user_interest=user_interest))],
            "research_topic":prompt,
            "intention_decision":"",
            "research_structure":"",
            "execution_plan":"",
            "academic_analysis_report":"",
            "social_analysis_report":"",
            "future_influence_report":"",
            "sentiment_analysis_report":"",
            "interdisciplinary_results":[],
            "current_discipline":"",
            "debate_results":[],
            "final_analysis_summary":"",
            "reflection_decision":"",
            "human_feedback":"",
            "feedback_routing_decision":"",
            "final_report":""
           
           
            # # 初始化“投资辩论”的子状态，所有字段都设为空或零
            # "investment_debate_state": InvestDebateState(
            #     **{"history": "", "current_response": "", "count": 0}
            # ),
            # # 初始化“风险评估”的子状态，所有字段都设为空或零
            # "risk_debate_state": RiskDebateState(
            #     **{
            #         "history": "",
            #         "current_risky_response": "",
            #         "current_safe_response": "",
            #         "current_neutral_response": "",
            #         "count": 0,
            #     }
            # ),
            # # 将所有需要由代理生成的报告内容初始化为空字符串
            # "market_report": "",
            # "fundamentals_report": "",
            # "sentiment_report": "",
            # "news_report": "",
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """
        获取用于调用（invoke）图的参数。
        这个函数将一些通用的、与图运行机制相关的配置打包起来，
        方便在调用图时直接传入。

        Returns:
            Dict[str, Any]: 一个包含图调用所需配置的字典。
        """
        return {
            # "stream_mode": "values" 指定了图的流式输出模式。
            # "values" 模式意味着每当图中的一个节点执行完毕，
            # 就会立即将该节点产生的状态更新流式地返回给调用者。
            # 这对于实时观察图的执行过程非常有用。
            "stream_mode": "values",
            # "config" 字段用于传递一些运行时的配置
            "config": {"recursion_limit": self.max_recur_limit},
        }