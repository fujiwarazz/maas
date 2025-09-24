

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END,START
from typing import Dict, Any, Generator, AsyncGenerator
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import json
import asyncio
from proposalAgent.agents.utils.agent_utils import get_user_query


def create_stream_generator(user_question: str) -> Generator[str, None, None]:
    """
    创建用于 FastAPI 流式响应的生成器
    """
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868"),
        streaming=True
    )
    
    prompt = f"""
        ### 角色描述
        你是一个专业的对话机器人，能够很好的回复用户的信息。
        ### 任务描述
        根据用户输入的内容，回复用户的信息。
        用户输入:{user_question}
        注意：请使用中文回复用户的信息、注意回答边界问题，如果用户输入的内容不属于你的能力范围或者询问的问题涉及政治、暴力等非法内容，请回复："我无法回答这个问题，请重新提问。"
        """
    
    # 发送开始信号
    yield f"data: {json.dumps({'type': 'start', 'content': 'AI回复：'}, ensure_ascii=False)}\n\n"
    
    # 流式输出内容
    for chunk in llm.stream(prompt):
        if hasattr(chunk, 'content') and chunk.content:
            content = chunk.content
            if isinstance(content, str):
                yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        yield f"data: {json.dumps({'type': 'content', 'content': item}, ensure_ascii=False)}\n\n"
    
    # 发送结束信号
    yield f"data: {json.dumps({'type': 'end', 'content': ''}, ensure_ascii=False)}\n\n"


async def create_async_stream_generator(user_question: str) -> AsyncGenerator[str, None]:
    """
    创建用于 FastAPI 异步流式响应的生成器
    """
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868"),
        streaming=True
    )
    
    prompt = f"""
        ### 角色描述
        你是一个专业的对话机器人，能够很好的回复用户的信息。
        ### 任务描述
        根据用户输入的内容，回复用户的信息。
        用户输入:{user_question}
        注意：请使用中文回复用户的信息、注意回答边界问题，如果用户输入的内容不属于你的能力范围或者询问的问题涉及政治、暴力等非法内容，请回复："我无法回答这个问题，请重新提问。"
        """
    
    # 发送开始信号
    yield f"data: {json.dumps({'type': 'start', 'content': 'AI回复：'}, ensure_ascii=False)}\n\n"
    
    # 流式输出内容
    for chunk in llm.stream(prompt):
        if hasattr(chunk, 'content') and chunk.content:
            content = chunk.content
            if isinstance(content, str):
                yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # 让出控制权
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        yield f"data: {json.dumps({'type': 'content', 'content': item}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0)  # 让出控制权
    
    # 发送结束信号
    yield f"data: {json.dumps({'type': 'end', 'content': ''}, ensure_ascii=False)}\n\n"



def create_output_node(llm):
    def get_output_node(state):
        
        # llm = ChatOpenAI(model="qwen-plus",
        #          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #          api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868"),
        #          streaming=True  # 启用流式输出
        #          )
        prompt = """
            ### 角色描述
            你是一个专业的对话机器人，能够很好的回复用户的信息。
            ### 任务描述
            根据用户输入的内容，回复用户的信息。
            用户输入:{user_question}
            注意：请使用中文回复用户的信息、注意回答边界问题，如果用户输入的内容涉及政治、暴力等非法内容，请回复："我无法回答这个问题，请重新提问。"
            """
        # 用户消息
        user_question = get_user_query(state)
        prompt = prompt.format(user_question=user_question)
        
        result = llm.invoke(prompt)

        
        current_messages = state.get("messages", [])
        new_messages = current_messages + [result.content]
        
        return {
            "messages": new_messages
        }
    node = get_output_node
    return node

if __name__ == "__main__":
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-0e349a8dc24443988825b69a56d2b868")
    )
    output_agent = create_output_node(llm)
    result = output_agent({"messages": [("user","你好")]})
    print(result)