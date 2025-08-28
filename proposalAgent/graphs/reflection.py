
from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """
    处理对决策的反思并将学习到的经验更新到记忆中。
    这个类是整个代理系统实现自我学习和迭代优化的核心。
    它通过调用一个大语言模型（LLM）来扮演一个专家分析师的角色，
    对过去的决策进行审查，并生成改进建议。
    """

    def __init__(self, quick_thinking_llm: Any):
        """
        使用一个大语言模型（LLM）来初始化反思器。
        
        Args:
            quick_thinking_llm (ChatOpenAI): 一个用于快速生成分析和反思的语言模型实例。
                                            "Quick thinking" 暗示这个模型可能配置为响应速度较快，
                                            适用于这种内部循环和分析任务。
        """
        self.quick_thinking_llm = quick_thinking_llm
        # 初始化时，预先加载用于反思的系统提示（System Prompt）
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """
        获取用于指导大语言模型进行反思的系统提示。
        这个提示是整个反思质量的灵魂，它为 LLM 设定了详细的角色、目标和输出格式。
        """
        return """
你是一名专家级的金融分析师，任务是审查交易决策/分析，并提供一个全面的、分步走的分析报告。
你的目标是就投资决策提供深刻的见解，并强调改进的机会，同时严格遵守以下准则：

1. 推理 (Reasoning):
   - 对于每一个交易决策，判断它是正确还是错误。正确的决策会增加回报，反之亦然。
   - 分析导致每次成功或失败的因素。请考虑：
     - 市场情报 (Market intelligence)。
     - 技术指标 (Technical indicators)。
     - 技术信号 (Technical signals)。
     - 价格走势分析 (Price movement analysis)。
     - 整体市场数据分析 (Overall market data analysis)。
     - 新闻分析 (News analysis)。
     - 社交媒体和情绪分析 (Social media and sentiment analysis)。
     - 基本面数据分析 (Fundamental data analysis)。
   - 评估每个因素在决策过程中的重要性权重。

2. 改进 (Improvement):
   - 对于任何错误的决策，提出修正方案以最大化回报。
   - 提供一个详细的纠正措施或改进清单，包括具体的建议（例如：在某个特定日期将决策从“持有”改为“买入”）。

3. 总结 (Summary):
   - 总结从成功和失败中学到的经验教训。
   - 强调这些教训如何应用于未来的交易场景，并通过连接相似情况来应用所学知识。

4. 查询 (Query):
   - 将总结中的核心见解提取成一个不超过1000个 token 的简洁句子。
   - 确保这个浓缩后的句子能抓住经验和推理的精髓，以便于快速参考。

请严格遵守这些指示，并确保你的输出是详细、准确且可操作的。
为了给你的分析提供更多背景信息，你还会获得关于市场价格走势、技术指标、新闻和情绪的客观描述。
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """
        从全局状态字典中提取当前的市场状况信息。
        这个函数负责将分散在状态中的各个分析报告整合成一个统一的“市场快照”。
        
        Args:
            current_state (Dict[str, Any]): 包含所有信息的全局状态字典。

        Returns:
            str: 一个包含了所有客观市场报告的字符串，作为反思时的上下文。
        """
        # 从状态中分别获取市场、情绪、新闻和基本面分析报告
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        # 将所有报告拼接成一个大的字符串，用换行符分隔
        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses: float
    ) -> str:
        """
        针对一个特定的决策组件（如某个代理的分析报告）生成反思。
        这是一个通用的调用 LLM 的函数。
        
        Args:
            component_type (str): 组件的类型（如 "BULL", "TRADER"），主要用于日志或调试。
            report (str): 需要被审查的分析报告或决策内容。
            situation (str): 当前的市场状况，由 _extract_current_situation 生成。
            returns_losses (float): 该决策导致的实际盈亏结果。

        Returns:
            str: 大语言模型生成的详细反思报告。
        """
        # 构建发送给 LLM 的消息列表，包含系统提示和用户输入
        messages = [
            ("system", self.reflection_system_prompt), # 角色和指令
            (
                "human",
                # 人类（用户）输入部分，提供了所有必要信息：盈亏、要审查的决策、以及当时的市场背景
                f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]

        # 调用 LLM 并获取其生成的内容
        result = self.quick_thinking_llm.invoke(messages).content
        return result

    # --- 以下是针对不同代理的具体反思方法 ---
    # 每个方法的逻辑都类似：
    # 1. 从全局状态中提取当前的市场状况。
    # 2. 从全局状态中提取该代理当初的分析或决策。
    # 3. 调用通用的 _reflect_on_component 方法让 LLM 生成反思。
    # 4. 将生成的反思结果存入该代理专属的记忆（memory）对象中，供其未来决策时参考。

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """反思看涨研究员的分析，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 从投资辩论状态中获取看涨方的历史发言
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        # 将（当时的市场状况, 反思结果）作为一个经验对，添加到看涨研究员的记忆中
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """反思看跌研究员的分析，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 从投资辩论状态中获取看跌方的历史发言
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        # 将经验对添加到看跌研究员的记忆中
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """反思交易员的决策，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 获取交易员最终制定的投资计划
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        # 将经验对添加到交易员的记忆中
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """反思投资“法官”的决策，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 从投资辩论状态中获取法官的最终决策
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        # 将经验对添加到投资法官的记忆中
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """反思风险管理“法官”的决策，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 从风险辩论状态中获取法官的最终决策
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        # 将经验对添加到风险管理法官的记忆中
        risk_manager_memory.add_situations([(situation, result)])