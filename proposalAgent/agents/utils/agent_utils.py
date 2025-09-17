from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
# from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
# import proposalAgent.tools.tool_interface as interface
from proposalAgent.model_config import TONGYI_CONFIG
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class Toolkit:
    
    _config = TONGYI_CONFIG

    def __init__(self, config: dict = None):
        self._config = config or self._config

    def get_config(self) -> dict:
        return self._config
    
    def set_config(self, config: dict):
        self._config = config
        
        
def get_user_query(state):
    user_input = None
    for msg in state["messages"]:
        if isinstance(msg, tuple) and msg[0] == "user":
            user_input = msg
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_input = msg
            break
    
    if user_input is None:
        user_input = state["messages"][0] if state["messages"] else ("user", "默认问题")
    
    # 提取用户问题
    if isinstance(user_input, tuple):
        user_question = user_input[1]
    elif hasattr(user_input, 'content'):
        # 处理AIMessage对象
        user_question = user_input.content
    elif isinstance(user_input, dict):
        user_question = user_input['content']
    else:
        user_question = str(user_input)
    
    print("user questoin",user_question)
    return user_question


def get_agents_info(state):
    agents_info = {
        
        "academic_agent":"使用学术调研工具，如同wos api和google scholar调查申请人的学术能力以及科研背景",
        "social_agent":"使用社会调研工具，如同百度指数调查申请人的社会影响力以及科研背景",
        "future_influence_agent":"使用未来影响调研工具，判断申请人申请项目可能的未来影响力大小",
        "feasible_good_agent":"使用可行性辩论工具，提出申请书的可行性正方观点",
        "feasible_bad_agent":"使用可行性辩论工具，提出申请书的可行性反方观点",
        "feasible_judge_agent":"使用可行性辩论工具，判断申请书的可行性正方和反方观点的优劣以及综合结果",
        "innovation_good_agent":"使用创新性辩论工具，提出申请书的创新性正方观点",
        "innovation_bad_agent":"使用创新性辩论工具，提出申请书的创新性反方观点",
        "innovation_judge_agent":"使用创新性辩论工具，判断申请书的创新性正方和反方观点的优劣以及综合结果",
        "interdisciplinary_agent":"使用跨学科调研工具，如同百度指数调查申请人的跨学科影响力以及科研背景",
    }
    return agents_info
    