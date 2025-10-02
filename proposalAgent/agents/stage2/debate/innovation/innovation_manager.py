from typing import Any, Dict, List

from proposalAgent.agents.utils.agent_states import AgentState, DebateState
from proposalAgent.agents.utils.memory import EmbeddingMemory


def create_innovation_manager(llm, memory: EmbeddingMemory):
    """
    综合创新性正反方观点，输出最终评审与结论的裁判节点。

    期望输入（state）关键字段：
    - research_basic_info, research_report_body_summary, academic_analysis_report, research_project_apply_info
    - current_discipline: tuple(code, name)
    - debate_results: Dict[str, Dict[str, DebateState]]
    """

    def innovation_manager(state: AgentState) -> Dict[str, Any]:
        research_info = state.get("research_basic_info", "")
        academic_report = state.get("academic_analysis_report", "")
        research_project_apply_info = state.get("research_project_apply_info", "")
        research_body = state.get("research_report_body_summary", "")

        curr = state.get("current_discipline")
        # disc_code = curr[0] if isinstance(curr, tuple) and len(curr) >= 2 else ""
        # disc_name = curr[1] if isinstance(curr, tuple) and len(curr) >= 2 else ""

        debate_kind = "创新性"
        debate_results: Dict[str, Dict[str, DebateState]] = state.get("debate_results", {}) or {}
        disc_bucket: Dict[str, DebateState] = debate_results.get(curr, {}) or {}
        inno_state: DebateState = disc_bucket.get(debate_kind, {}) or {}

        good_history: List[str] = list(inno_state.get("good_agent_history", []))
        bad_history: List[str] = list(inno_state.get("bad_agent_history", []))
        full_history: List[str] = list(inno_state.get("full_history", []))
        prev_rounds = int(inno_state.get("debate_rounds", 1))

        # 组织当前情境与记忆
        curr_situation = f"{research_info}\n\n{academic_report}\n\n{research_project_apply_info}\n\n{research_body}"
        #past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memories = []
        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec.get("recommendation", "") + "\n\n"

        debate_transcript = []
        if good_history:
            debate_transcript.append("【正方观点】\n" + "\n".join(good_history))
        if bad_history:
            debate_transcript.append("【反方观点】\n" + "\n".join(bad_history))
        if full_history:
            debate_transcript.append("【完整对话历史】\n" + "\n".join(full_history))
        debate_text = "\n\n".join(debate_transcript) or "(暂无历史辩论记录)"

        prompt = f"""
            你是该学科 {curr} 的项目创新性裁判。请基于双方观点与项目信息，给出清晰、可执行的最终结论。

            要求：
            - 先各用不超过3点总结正反双方对“创新性”的最有力论据。
            - 给出最终判定：高创新性 / 中等创新性 / 低创新性（必须三选一）。
            - 从以下维度说明判定理由：方法创新、数据与资源、应用/场景、理论/算法贡献、系统工程、潜在影响。
            - 给出改进建议与下一步关键实验（可执行、可验证）。
            - 若涉及相关工作，请指出差异化点与最小可复现证据。

            可参考过往反思：
            {past_memory_str}

            本轮辩论记录：
            {debate_text}
            """

        response = llm.invoke(prompt)
        judge_summary = str(getattr(response, "content", response)).strip()

        # 更新状态拷贝
        new_inno_state: DebateState = {
            "good_agent_history": good_history,
            "bad_agent_history": bad_history,
            "full_history": full_history + [f"裁判结论：{judge_summary}"] if judge_summary else full_history,
            "judge_summary": judge_summary,
            "debate_rounds": prev_rounds,
        }

        new_disc_bucket = dict(disc_bucket)
        new_disc_bucket[debate_kind] = new_inno_state

        new_debate_results = dict(debate_results)
        new_debate_results[curr] = new_disc_bucket

        return {
            "debate_results": new_debate_results,
            "innovation_decision": judge_summary,
        }

    return innovation_manager
