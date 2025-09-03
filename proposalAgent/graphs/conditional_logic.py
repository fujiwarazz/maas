# TradingAgents/graph/conditional_logic.py

from proposalAgent.agents.utils.agent_states import AgentState
import json
"""
这个文件使用Agent state中的message[-1]来判断llm是否调用工具，如果调用了工具那么就继续走分析师那一步，
"""
class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""
    
    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds
    
    def should_route_to_output(self,state:AgentState):
        """Determine if the output tool should be called."""
        return True

    # ========== Stage 0 / Intention ==========
    def _is_finalize_signal(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        tokens = ["<FINALIZE>", "FINALIZE", "DONE", "完成", "结束", "生成报告", "生成结果"]
        text_upper = text.upper()
        return any(t in text_upper for t in tokens)

    def _last_message(self, state: AgentState):
        try:
            return state["messages"][-1]
        except Exception:
            return None

    def _has_tool_calls(self, message) -> bool:
        try:
            return bool(getattr(message, "tool_calls", None))
        except Exception:
            return False

    def _route_three_way(self, state: AgentState, tools_key: str, msg_clear_key: str, final_key: str) -> str:
        last = self._last_message(state)
        if last is None:
            return msg_clear_key
        if self._is_finalize_signal(getattr(last, "content", "")):
            return final_key
        if self._has_tool_calls(last):
            return tools_key
        return msg_clear_key

    def _route_debate(self, state: AgentState) -> str | None:
        last = self._last_message(state)
        if last is None:
            return None
        content = getattr(last, "content", "") or ""
        if self._is_finalize_signal(content):
            return "end"
        if self._has_tool_calls(last):
            return "tools"
        return "continue"

        # 独立函数用于 setup.add_conditional_edges 中的 "should_output"
    def should_output(self,state: AgentState) -> str:
        """Decide whether to go directly to output or continue to structure.
        返回值需在 setup 中的映射键内："output_node" 或 "structure_node"。
        规则：
        - 若 state["should_output"] 为真或消息包含明显终止信号，则走 "output_node"
        - 否则默认走 "structure_node"
        """
        try:
            last = state["messages"][-1]
            content = getattr(last, "content", "") or ""
        except Exception:
            content = ""
        should = False
        if isinstance(state, dict) and state.get("should_output") is True:
            should = True
        tokens = ["<FINALIZE>", "FINALIZE", "DONE", "完成", "直接输出", "生成结果", "生成报告"]
        cu = content.upper()
        if any(t in cu for t in tokens):
            should = True
        return "output_node" if should else "structure_node"
    
    
    def route_after_planning(self,state: AgentState) -> str | list[str]:
        """
        This function now directly decides the next step(s).
        It returns either a single string (node name) or a list of strings (parallel nodes).
        """
        print("--- Deciding next step after planning ---")
        # Add the self. prefix to the method call
        if self.should_route_to_output(state):
            print(">>> Routing to: output_tool")
            return "output_node"
        else:
            print(">>> Routing to: parallel analysis nodes")
            parallel_nodes = [
                "impact analyst",
                "future influence analyst",
                "risk analyst",
                "interdisciplinary analyst",
                "academic analyst",
                "feasibility analyst",
                "innovation analyst",
            ]
            return parallel_nodes



    # ========== Stage 2 / 收集阶段 ==========
    def should_continue_academic_analysis(self, state: AgentState) -> str:
        return self._route_three_way(state, "tools_academic", "msg_clear_academic", "final_analyst_node")

    def should_continue_social_analysis(self, state: AgentState) -> str:
        return self._route_three_way(state, "tools_social", "msg_clear_social", "final_analyst_node")

    def should_continue_future_influence(self, state: AgentState) -> str:
        return self._route_three_way(state, "tools_future_influence", "msg_clear_future_influence", "final_analyst_node")

    def should_continue_interdisciplinary(self, state: AgentState) -> str:
        return self._route_three_way(state, "tools_interdisciplinary", "msg_clear_interdisciplinary", "final_analyst_node")

    def should_request_human_review(self, state: AgentState) -> str:
        """
        Determines whether to proceed with generation or request human review based on the reflection agent's output.
        """
        print("--- Checking if human review is needed ---")
        last_message = state["messages"][-1]
        try:
            content = getattr(last_message, "content", "")
            reflection_output = {}
            if isinstance(content, dict):
                reflection_output = content
            elif isinstance(content, (str, bytes, bytearray)):
                reflection_output = json.loads(content)  # may raise
            recommendation = reflection_output.get("recommendation")
            print(f">>> Reflection agent recommended: {recommendation}")
            if recommendation == "review":
                return "review"
            else:
                return "generate"
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Error parsing reflection output: {e}. Defaulting to generation.")
            # Fallback in case of parsing errors
            return "generate"

    def route_after_feedback(self, state: AgentState) -> str:
        """
        Routes the workflow to the appropriate node based on the feedback analysis agent's output.
        """
        print("--- Routing after human feedback ---")
        last_message = state["messages"][-1]
        try:
            content = getattr(last_message, "content", "")
            feedback_analysis_output = {}
            if isinstance(content, dict):
                feedback_analysis_output = content
            elif isinstance(content, (str, bytes, bytearray)):
                feedback_analysis_output = json.loads(content)  # may raise
            next_step = feedback_analysis_output.get("next_step")
            print(f">>> Feedback analysis recommends routing to: {next_step}")
            
            # Validate that the next_step is a valid node name before returning
            valid_routes = [
                "academic_analysis",
                "social_analysis",
                "future_influence",
                "interdisciplinary",
                "debate",
                "generate",
            ]
            if next_step in valid_routes:
                return next_step
            else:
                print(f"Invalid route '{next_step}' recommended. Defaulting to generation.")
                return "generate"

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Error parsing feedback analysis output: {e}. Defaulting to generation.")
            # Fallback in case of parsing errors
            return "generate"


    # ========== 辩论阶段 ==========
    def should_continue_feasibility(self, state: AgentState) -> str:
        res = self._route_debate(state)
        return res if res is not None else "continue"

    def should_continue_innovation(self, state: AgentState) -> str:
        res = self._route_debate(state)
        return res if res is not None else "continue"

    # 额外的风控/影响判断函数已删除（未在当前流程中使用且类型与 AgentState 不一致）


# ========== 模块级别函数，兼容 setup.py 的直接引用 ==========
_default_logic = ConditionalLogic()

def should_output(state: AgentState) -> str:
    return _default_logic.should_output(state)

def should_continue_academic_analysis(state: AgentState) -> str:
    return _default_logic.should_continue_academic_analysis(state)

def should_continue_social_analysis(state: AgentState) -> str:
    return _default_logic.should_continue_social_analysis(state)

def should_continue_future_influence(state: AgentState) -> str:
    return _default_logic.should_continue_future_influence(state)

def should_continue_interdisciplinary(state: AgentState) -> str:
    return _default_logic.should_continue_interdisciplinary(state)
