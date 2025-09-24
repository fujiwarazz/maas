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
            "output",
            "structure"
        ]
        
        
        system_prompt = """
        你是一个智能意图识别助手，负责分析用户的输入内容，并判断其意图属于以下哪一类：
        
        1. 通用模型能力（如：日常问答、常规推理、代码生成、知识检索等，模型本身即可胜任的任务）-> 返回：output
        2. 申请书/论文评估（如：涉及对学术申请书、科研论文、项目计划书等内容的结构、创新性、可行性、学术价值等方面的分析与评估）-> 返回：structure
        
        用户输入: {user_question}
        
        请根据用户输入内容，准确判断其意图类型，并只返回对应的标识（output 或 structure）。
        """
       
        
        user_question = get_user_query(state)
        system_prompt = system_prompt.format(user_question=user_question)
        result = llm.invoke(system_prompt)

        should_output = False
        if result.content and "output" in result.content.lower():
            should_output = True
        
        print(f"intention_decision: {result.content}")
        return {
            "messages": [result],
            "intention_decision": result.content,
            "should_output": should_output
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