from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState

def create_reflection_agent(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a reflection agent. Your role is to analyze the work done by the other agents and provide a confidence score.
                Based on the analysis, decide if the result is good enough to be sent to the generator or if it needs human review.
                The confidence score should be an integer between 0 and 100.
                - A score of 80 or higher means the analysis is robust and can proceed to generation.
                - A score below 80 means there are potential issues, and it should be reviewed by a human.

                Output a JSON object with two keys: 'confidence_score' and 'recommendation'.
                'recommendation' should be either 'generate' or 'review'.
                """
            ),
            ("human", "{messages}"),
        ]
    )
    return prompt | llm