from typing import Any, Dict, List

from proposalAgent.agents.utils.agent_states import AgentState, DebateState
from proposalAgent.agents.utils.memory import EmbeddingMemory


def create_feasible_manager(llm, memory: EmbeddingMemory):
    """
    综合可行性正反方观点，输出最终评审与结论的裁判节点。

    期望输入（state）关键字段：
    - research_basic_info, research_report_body_summary, academic_analysis_report, research_project_apply_info
    - current_discipline: tuple(code, name)
    - debate_results: Dict[str, Dict[str, DebateState]]
    """

    def feasible_manager(state: AgentState) -> Dict[str, Any]:
        research_info = state.get("research_basic_info", "")
        academic_report = state.get("academic_analysis_report", "")
        research_project_apply_info = state.get("research_project_apply_info", "")
        research_body = state.get("research_report_body_summary", "")

        curr = state.get("current_discipline")
        disc_code = curr[0] if isinstance(curr, tuple) and len(curr) >= 2 else ""
        disc_name = curr[1] if isinstance(curr, tuple) and len(curr) >= 2 else ""

        debate_kind = "可行性"
        debate_results: Dict[str, Dict[str, DebateState]] = state.get("debate_results", {}) or {}
        disc_bucket: Dict[str, DebateState] = debate_results.get(disc_name, {}) or {}
        feas_state: DebateState = disc_bucket.get(debate_kind, {}) or {}

        good_history: List[str] = list(feas_state.get("good_agent_history", []))
        bad_history: List[str] = list(feas_state.get("bad_agent_history", []))
        full_history: List[str] = list(feas_state.get("full_history", []))
        prev_rounds = int(feas_state.get("debate_rounds", 1))

        # 组织当前情境与记忆
        curr_situation = f"{research_info}\n\n{academic_report}\n\n{research_project_apply_info}\n\n{research_body}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
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
            你是该学科（{disc_code} {disc_name}）的项目可行性裁判。请基于双方观点与项目信息，给出清晰、可执行的最终结论。

            要求：
            - 先简要总结正反双方的最有力观点（各不超过3点）。
            - 给出最终判定：可行 / 部分可行 / 不可行（必须三选一）。
            - 提供判定理由（关键证据与逻辑）。
            - 给出落实建议（需可执行，含短期行动清单）。
            - 标注关键风险与缓解措施（若有）。

            可参考过往反思：
            {past_memory_str}

            本轮辩论记录：
            {debate_text}
            """

        response = llm.invoke(prompt)
        judge_summary = str(getattr(response, "content", response)).strip()

        # 更新状态拷贝
        new_feas_state: DebateState = {
            "good_agent_history": good_history,
            "bad_agent_history": bad_history,
            "full_history": full_history + [f"裁判结论：{judge_summary}"] if judge_summary else full_history,
            "judge_summary": judge_summary,
            "debate_rounds": prev_rounds,
        }

        new_disc_bucket = dict(disc_bucket)
        new_disc_bucket[debate_kind] = new_feas_state

        new_debate_results = dict(debate_results)
        new_debate_results[disc_name] = new_disc_bucket

        return {
            "debate_results": new_debate_results,
            "feasibility_decision": judge_summary,
        }

    return feasible_manager
