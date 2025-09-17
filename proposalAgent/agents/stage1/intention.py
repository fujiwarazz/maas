import sys
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_utils import get_user_query

def create_intention_agent(llm):

    def intention_agent(state):
        intentions = [
            "通用模型能力（如：日常问答、常规推理、代码生成、知识检索等，模型本身即可胜任的任务",
            "申请书/论文评估（如：涉及对学术申请书、科研论文、项目计划书等内容的结构、创新性、可行性、学术价值等方面的分析与评估）"
        ]
        intentions_flag = [
            "structure",
            "output"
        ]
        
        
        system_prompt = """
        你是一个智能意图识别助手，负责分析用户的输入内容，并判断其意图属于以下哪一类：
        {intentions}
        请根据用户输入内容，准确判断其意图类型,
        他们对应的返回标识为：
        {intentions_flag}
        请根据用户输入内容，准确判断其意图类型,并返回对应的返回标识,
        用户输入:{user_question}
        """
       
        
        user_question = get_user_query(state)
        system_prompt = system_prompt.format(user_question=user_question,intentions="\n".join(intentions),intentions_flag="\n".join(intentions_flag))
        result = llm.invoke(system_prompt)

        return {
            "messages": [result],
            "intention_decision":result.content
        }

    return intention_agent

if __name__ == "__main__":
    llm = ChatOpenAI(model="qwen-plus",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 api_key="sk-0e349a8dc24443988825b69a56d2b868"
                 )
    intention_agent = create_intention_agent(llm)
    result = intention_agent({"messages": [("user", "分析这篇文章")]})
    print(result)