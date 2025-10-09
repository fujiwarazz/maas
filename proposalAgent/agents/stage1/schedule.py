import sys
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from langchain_openai import ChatOpenAI
import pathlib
import os

current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from proposalAgent.agents.utils.agent_utils import get_user_query, get_agents_info

def create_schedule_agent(llm):

    def schedule_agent(state):
        
        user_topic = ",".join(state["research_topic"])
        
        agents_info = get_agents_info(state)
        
        user_question = get_user_query(state)
        
        # 创建默认权重字典
        weight = {
            k: 0.2 for k, _ in agents_info.items()
        }
        
        system_prompt = f"""
        ### 角色定义
        你是一个智能的流程调度助手，负责根据用户输入的主题，分配相应的智能体对应的权重完成任务。
        权重表述的是智能体或者任务节点可以消耗的资源，如果权重越大，节点agent就会花费更多的时间。
        ### 任务定义
        根据用户输入的主题，分配相应的智能体对应的权重完成任务。
        用户输入：{user_question}
        用户输入的主题：{user_topic}
        智能体信息：{str(agents_info)}
       
        输出格式：
        {{
            "agent_name":"weight(float)",
            "agent_name":"weight(float)"
        }}
        """
        
        result = llm.invoke(system_prompt)
        
        weight_distribution = json.loads(result.content)
        for k,v in weight_distribution.items():
            if k in weight.keys():
                if k == "academic_agent":
                    weight[k] = 0.5
                else: 
                    weight[k] = v
            
        print(weight)
        return {
            "messages": [result],
            "weight_distribution":weight
        }

    return schedule_agent

if __name__ == "__main__":
    llm = ChatOpenAI(model="qwen-plus",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 api_key="sk-0e349a8dc24443988825b69a56d2b868"
                 )
    schedule_agent = create_schedule_agent(llm)
    result = schedule_agent({"messages": [("user", "分析这篇文章")],"research_topic":["交叉性","未来影响力"]})
    print(result)