from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState

def create_feedback_analysis_agent(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a feedback analysis agent. Your role is to analyze the human feedback and determine which part of the process needs to be re-run.
                The user will provide feedback on the analysis. Your job is to understand the feedback and map it to one of the following stages:
                - academic_analysis
                - social_analysis
                - future_influence
                - interdisciplinary
                - debate
                - generate

                If the feedback is positive and no changes are needed, output 'generate'.
                Otherwise, output the name of the stage that needs to be revisited.
                Output a JSON object with one key: 'next_step'.
                """
            ),
            ("human", "{messages}"),
        ]
    )
    return prompt | llm