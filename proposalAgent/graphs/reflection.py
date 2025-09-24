
from typing import Dict, Any, Optional


class Reflector:
    """
    处理对项目评估决策的反思并将学习到的经验更新到记忆中。
    这个类是整个项目评估代理系统实现自我学习和迭代优化的核心。
    它通过调用一个大语言模型（LLM）来扮演一个专家评审员的角色，
    对过去的评估决策进行审查，并生成改进建议。
    """

    def __init__(self, quick_thinking_llm):
        """
        使用一个大语言模型（LLM）来初始化反思器。
        """
        self.quick_thinking_llm = quick_thinking_llm
        # 初始化时，预先加载用于反思的系统提示（System Prompt）
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """
        获取用于指导大语言模型进行项目评估反思的系统提示。
        这个提示是整个反思质量的灵魂，它为 LLM 设定了详细的角色、目标和输出格式。
        """
        return """
            你是一名专家级的项目评估专家和学术审查员，任务是审查项目评估决策/分析，并提供一个全面的、分步走的反思报告。
            你的目标是对项目评估决策提供深刻的见解，并强调改进的机会，同时严格遵守以下准则：

            1. 推理分析 (Reasoning Analysis):
            - 对于每一个评估决策，判断它是合理还是存在偏差。合理的评估有助于正确识别项目价值，偏差的评估可能误导决策。
            - 分析导致评估成功或失败的关键因素，请考虑：
                - 学术分析的深度和准确性 (Academic analysis depth and accuracy)
                - 社会影响力评估的全面性 (Social impact assessment comprehensiveness)
                - 未来影响预测的合理性 (Future impact prediction reasonableness)
                - 跨学科分析的覆盖面 (Interdisciplinary analysis coverage)
                - 可行性评估的客观性 (Feasibility assessment objectivity)
                - 创新性判断的准确性 (Innovation judgment accuracy)
                - 人类反馈的质量和相关性 (Human feedback quality and relevance)
            - 评估每个分析维度在最终决策中的重要性权重。

            2. 改进建议 (Improvement Recommendations):
            - 对于任何有偏差的评估，提出具体的修正方案以提高评估准确性。
            - 提供详细的改进清单，包括具体建议：
                - 学术分析：需要补充哪些学术指标或数据源
                - 社会分析：应该关注哪些被忽视的社会影响维度
                - 未来预测：如何提高预测的科学性和可信度
                - 跨学科：需要纳入哪些额外的学科视角
                - 辩论质量：如何提升正反方论证的深度和平衡性

            3. 经验总结 (Experience Summary):
            - 总结从成功和失败的评估案例中学到的经验教训。
            - 强调这些教训如何应用于未来的项目评估场景。
            - 识别评估过程中的常见陷阱和最佳实践。

            4. 核心洞察 (Core Insights):
            - 将总结中的核心见解提取成一个不超过500字的精炼总结。
            - 确保这个总结能够抓住评估经验和推理的精髓，便于快速参考和应用。

            5. 质量评估 (Quality Assessment):
            - 对整个评估过程的质量进行1-5分的评分。
            - 识别评估过程中的强项和薄弱环节。
            - 提供具体的质量改进路径。

            请严格遵守这些指示，并确保你的输出是详细、准确且可操作的。
            为了给你的分析提供更多背景信息，你还会获得关于项目基本情况、各维度分析结果、辩论过程和最终评估结果的客观描述。
            """

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """
        从全局状态字典中提取当前的项目评估情况信息。
        这个函数负责将分散在状态中的各个分析报告整合成一个统一的"项目评估快照"。
        
        Args:
            current_state (Dict[str, Any]): 包含所有信息的全局状态字典。

        Returns:
            str: 一个包含了所有项目评估报告的字符串，作为反思时的上下文。
        """
        # 从状态中分别获取各个维度的分析报告
        research_topic = current_state.get("research_topic", "未提供")
        research_basic_info = current_state.get("research_basic_info", "未提供")
        academic_report = current_state.get("academic_analysis_report", "未进行学术分析")
        social_report = current_state.get("social_analysis_report", "未进行社会分析")
        future_influence_report = current_state.get("future_influence_report", "未进行未来影响分析")
        interdisciplinary_results = current_state.get("interdisciplinary_results", [])
        debate_results = current_state.get("debate_results", {})
        final_analysis = current_state.get("final_analysis_summary", "未完成最终分析")
        
        # 格式化跨学科结果
        interdisciplinary_str = "涉及学科：" + ", ".join(interdisciplinary_results) if interdisciplinary_results else "未识别跨学科领域"
        
        # 格式化辩论结果
        debate_summary = "辩论结果摘要：\n"
        if isinstance(debate_results, dict):
            for discipline, debates in debate_results.items():
                debate_summary += f"  {discipline}学科：\n"
                if isinstance(debates, dict):
                    for debate_type, result in debates.items():
                        debate_summary += f"    {debate_type}：{str(result)[:500]}...\n"
                else:
                    debate_summary += f"    {str(debates)[:500]}...\n"
        else:
            debate_summary += "  无辩论数据\n"

        # 将所有报告拼接成一个大的字符串，用换行符分隔
        return f"""
                着重关心点：{research_topic}

                项目基本信息：{research_basic_info}

                学术分析报告：{academic_report}

                社会分析报告：{social_report}

                未来影响分析报告：{future_influence_report}

                {interdisciplinary_str}

                {debate_summary}

                最终分析摘要：{final_analysis}"""

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, evaluation_outcome:Optional[Dict[str, Any]]=None
    ) -> str:
        """
        针对一个特定的评估组件（如某个代理的分析报告）生成反思。
        这是一个通用的调用 LLM 的函数。
        
        Args:
            component_type (str): 组件的类型（如 "ACADEMIC", "SOCIAL", "FEASIBILITY"），主要用于日志或调试。
            report (str): 需要被审查的分析报告或决策内容。
            situation (str): 当前的项目评估情况，由 _extract_current_situation 生成。
            evaluation_outcome (Dict[str, Any]): 该评估导致的实际结果（如人类反馈、完备性检查结果等）。

        Returns:
            str: 大语言模型生成的详细反思报告。
        """
        # 由人类添加
        outcome_str = f"""
        评估结果：
        完备性检查：{evaluation_outcome.get('completeness_check', '未进行')}
        人类反馈：{evaluation_outcome.get('human_feedback', '无反馈')}
        最终评分：{evaluation_outcome.get('final_score', '未评分')}
        改进建议：{evaluation_outcome.get('improvement_suggestions', '无建议')}
"""

        # 构建发送给 LLM 的消息列表，包含系统提示和用户输入
        messages = [
            ("system", self.reflection_system_prompt), # 角色和指令
            (
                "human",
                # 人类（用户）输入部分，提供了所有必要信息：评估结果、要审查的分析、以及项目背景
                f"分析组件类型: {component_type}\n\n待审查的分析/决策: {report}\n\n项目评估背景信息: {situation}",
            ),
        ]

        # 调用 LLM 并获取其生成的内容
        result = self.quick_thinking_llm.invoke(messages)
        content = result.content if hasattr(result, 'content') else str(result)
        return str(content)

    # --- 以下是针对不同评估代理的具体反思方法 ---
    # 每个方法的逻辑都类似：
    # 1. 从全局状态中提取当前的项目评估情况。
    # 2. 从全局状态中提取该代理当初的分析或决策。
    # 3. 调用通用的 _reflect_on_component 方法让 LLM 生成反思。
    # 4. 将生成的反思结果存入该代理专属的记忆（memory）对象中，供其未来决策时参考。

    def reflect_academic_analyst(self, current_state,academic_memory, evaluation_outcome=None):
        """反思学术分析代理的分析，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 获取学术分析报告
        academic_report = current_state.get("academic_analysis_report", "未进行学术分析")

        result = self._reflect_on_component(
            "ACADEMIC", academic_report, situation, evaluation_outcome
        )
        # 将（当时的项目情况, 反思结果）作为一个经验对，添加到学术分析代理的记忆中
        academic_memory.add_situations([(situation, result)])

    def reflect_social_analyst(self, current_state, social_memory, evaluation_outcome=None):
        """反思社会分析代理的分析，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 获取社会分析报告
        social_report = current_state.get("social_analysis_report", "未进行社会分析")

        result = self._reflect_on_component(
            "SOCIAL", social_report, situation, evaluation_outcome
        )
        # 将经验对添加到社会分析代理的记忆中
        social_memory.add_situations([(situation, result)])

    def reflect_future_influence_analyst(self, current_state, future_memory, evaluation_outcome=None):
        """反思未来影响分析代理的分析，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 获取未来影响分析报告
        future_report = current_state.get("future_influence_report", "未进行未来影响分析")

        result = self._reflect_on_component(
            "FUTURE_INFLUENCE", future_report, situation, evaluation_outcome
        )
        # 将经验对添加到未来影响分析代理的记忆中
        future_memory.add_situations([(situation, result)])

    def reflect_interdisciplinary_analyst(self, current_state, interdis_memory, evaluation_outcome=None):
        """反思跨学科分析代理的分析，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 获取跨学科分析结果
        interdis_results = current_state.get("interdisciplinary_results", [])
        interdis_report = "识别的学科领域：" + ", ".join(interdis_results) if interdis_results else "未识别跨学科领域"

        result = self._reflect_on_component(
            "INTERDISCIPLINARY", interdis_report, situation, evaluation_outcome
        )
        # 将经验对添加到跨学科分析代理的记忆中
        interdis_memory.add_situations([(situation, result)])

    def reflect_feasibility_debate(self, current_state, feasibility_memory, evaluation_outcome=None):
        """反思可行性辩论的质量，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 从辩论结果中获取可行性辩论的历史
        debate_results = current_state.get("debate_results", {})
        feasibility_debate_summary = self._extract_debate_summary(debate_results, "可行性")

        result = self._reflect_on_component(
            "FEASIBILITY_DEBATE", feasibility_debate_summary, situation, evaluation_outcome
        )
        # 将经验对添加到可行性辩论的记忆中
        feasibility_memory.add_situations([(situation, result)])

    def reflect_innovation_debate(self, current_state, innovation_memory, evaluation_outcome=None):
        """反思创新性辩论的质量，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 从辩论结果中获取创新性辩论的历史
        debate_results = current_state.get("debate_results", {})
        innovation_debate_summary = self._extract_debate_summary(debate_results, "创新性")

        result = self._reflect_on_component(
            "INNOVATION_DEBATE", innovation_debate_summary, situation, evaluation_outcome
        )
        # 将经验对添加到创新性辩论的记忆中
        innovation_memory.add_situations([(situation, result)])

    def reflect_final_analyst(self, current_state, final_memory, evaluation_outcome=None):
        """反思最终分析代理的决策，并更新其记忆。"""
        situation = self._extract_current_situation(current_state)
        # 获取最终分析摘要
        final_analysis = current_state.get("final_analysis_summary", "未完成最终分析")

        result = self._reflect_on_component(
            "FINAL_ANALYST", final_analysis, situation, evaluation_outcome
        )
        # 将经验对添加到最终分析代理的记忆中
        final_memory.add_situations([(situation, result)])

    def _extract_debate_summary(self, debate_results: Dict[str, Any], debate_type: str) -> str:
        """
        从辩论结果中提取特定类型的辩论摘要
        
        Args:
            debate_results: 辩论结果字典
            debate_type: 辩论类型（"可行性" 或 "创新性"）
            
        Returns:
            str: 格式化的辩论摘要
        """
        summary_parts = []
        
        for discipline, debates in debate_results.items():
            if isinstance(debates, dict) and debate_type in debates:
                debate_data = debates[debate_type]
                summary_parts.append(f"{discipline}学科{debate_type}辩论：")
                
                if isinstance(debate_data, dict):
                    if "judge_summary" in debate_data:
                        summary_parts.append(f"  裁判总结：{debate_data['judge_summary']}")
                    if "full_history" in debate_data:
                        history = debate_data['full_history']
                        if isinstance(history, list):
                            summary_parts.append(f"  辩论历史：{'; '.join(history[:3])}...")  # 只取前3轮
                        else:
                            summary_parts.append(f"  辩论历史：{str(history)[:200]}...")
                else:
                    summary_parts.append(f"  {str(debate_data)[:200]}...")
        
        return "\n".join(summary_parts) if summary_parts else f"未找到{debate_type}辩论数据"