from proposalAgent.graphs.proposal_graph import ProposalAgentGraph
from proposalAgent.model_config import TONGYI_CONFIG
import asyncio

# Create a custom config
config = TONGYI_CONFIG.copy()
# config["llm_provider"] = "google"  # Use a different model
# config["backend_url"] = "https://generativelanguage.googleapis.com/v1"  # Use a different backend
# config["deep_think_llm"] = "gemini-2.5-flash"  # Use a different model
# config["quick_think_llm"] = "gemini-2.5-flash"  # Use a different model
# config["max_debate_rounds"] = 1  # Increase debate rounds


async def main():
    # Initialize with custom config
    ta = ProposalAgentGraph(config=config)

    decision = await ta.evaluate_project(user_prompt="请分析这个项目",user_interests=["技术可行性","社会影响","创新性","学术性"],filepath="/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/提交版本.pdf")
    print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns

if __name__ == "__main__":
    asyncio.run(main())