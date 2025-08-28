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
    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"


    def should_continue_impact(self,state:AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_impact"
        return "Msg Clear Impact"
    
    def should_continue_impact_analysis(self, state: AgentState):
        """Determine if impact analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_impact"
        return "Msg Clear Impact"
    
    def should_continue_interdisciplinary(self, state: AgentState):
        """Determine if interdisciplinary analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_interdisciplinary"
        return "Msg Clear Interdisciplinary"
    
    def should_continue_academic(self, state: AgentState):
        """Determine if academic analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_academic"
        return "Msg Clear Academic"

    def should_request_human_review(self, state: AgentState) -> str:
        """
        Determines whether to proceed with generation or request human review based on the reflection agent's output.
        """
        print("--- Checking if human review is needed ---")
        last_message = state["messages"][-1]
        try:
            # The reflection agent is expected to return a JSON object in its content.
            reflection_output = json.loads(last_message.content)
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
            # The feedback analysis agent is expected to return a JSON object in its content.
            feedback_analysis_output = json.loads(last_message.content)
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

    
    def should_continue_feasible_debate(self, state: AgentState):
        """Determine if feasible debate should continue."""
        if(
            state["feasibility_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):
            return "Feasibility Judge"
        if state["feasibility_debate_state"]["current_response"].startswith("Feasibility Good"):
            return "Feasibility Good Analyst"
        
        return "Feasibility Bad Analyst"

    def should_continue_innovation_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["innovation_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Innovation Judge"
        if state["innovation_debate_state"]["current_response"].startswith("Innovation Good"):
            return "Innovation Good Analyst"
        return "Innovation Bad Analyst"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risk Risky"):
            return "Risk Risky Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risk Conservative"):
            return "Risk Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risk Neutral"):
            return "Neutral Analyst"
        return "Risky Analyst"
    
    def should_continue_influence_analysis(self, state: AgentState) -> str:
        """Determine if influence analysis should continue."""
        if (
            state["future_influence_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Future Influence Judge"
        if state["future_influence_debate_state"]["latest_speaker"].startswith("Future Good"):
            return "Future Good Analyst"
        if state["future_influence_debate_state"]["latest_speaker"].startswith("Future Bad"):
            return "Future Bad Analyst"
        if state["future_influence_debate_state"]["latest_speaker"].startswith("Future Neutral"):
            return "Future Neutral Analyst"
        return "No Influence Analyst"
    
    