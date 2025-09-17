#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangGraph 节点与 FastAPI 流式响应集成示例
展示如何将图节点的输出通过 FastAPI 进行流式返回
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import sys
import os
import json
import asyncio

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from proposalAgent.agents.stage1.output import create_output_node

app = FastAPI(title="LangGraph流式节点API", description="将LangGraph节点输出进行流式返回")


class NodeRequest(BaseModel):
    """节点请求模型"""
    message: str
    user_id: Optional[str] = None


async def stream_node_output(state: dict) -> AsyncGenerator[str, None]:
    """
    将LangGraph节点的输出转换为流式响应
    这是一个适配器函数，将原有的节点逻辑包装为流式输出
    """
    try:
        # 发送开始信号
        yield f"data: {json.dumps({'type': 'start', 'content': '节点开始处理...'}, ensure_ascii=False)}\n\n"
        
        # 创建原有的输出节点
        output_node = create_output_node()
        
        # 这里我们需要修改原有节点以支持流式输出
        # 为了演示，我们先获取完整结果，然后模拟流式输出
        result = output_node(state)
        
        if 'messages' in result and result['messages']:
            message_content = result['messages'][0].content
            
            # 模拟逐字符流式输出
            yield f"data: {json.dumps({'type': 'content', 'content': 'AI回复：'}, ensure_ascii=False)}\n\n"
            
            # 将内容分块流式输出
            chunk_size = 5  # 每次输出5个字符
            for i in range(0, len(message_content), chunk_size):
                chunk = message_content[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)  # 模拟网络延迟
        
        # 发送结束信号
        yield f"data: {json.dumps({'type': 'end', 'content': '节点处理完成'}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'节点处理错误: {str(e)}'}, ensure_ascii=False)}\n\n"


@app.post("/node/output/stream")
async def node_output_stream(request: NodeRequest):
    """
    流式输出节点处理结果
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        # 构造节点状态
        state = {
            "messages": [{"role": "user", "content": request.message}]
        }
        
        # 使用流式生成器
        generator = stream_node_output(state)
        
        return StreamingResponse(
            generator,
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


class StreamableOutputNode:
    """
    可流式输出的节点包装器
    将原有的节点逻辑包装为支持流式输出的形式
    """
    
    def __init__(self):
        self.original_node = create_output_node()
    
    async def stream_execute(self, state: dict) -> AsyncGenerator[dict, None]:
        """
        流式执行节点逻辑
        """
        try:
            # 发送处理开始事件
            yield {"type": "start", "content": "开始处理节点逻辑..."}
            
            # 执行原有节点逻辑
            result = self.original_node(state)
            
            # 发送处理结果
            yield {"type": "result", "content": result}
            
            # 发送处理完成事件
            yield {"type": "end", "content": "节点处理完成"}
            
        except Exception as e:
            yield {"type": "error", "content": f"节点处理错误: {str(e)}"}


@app.post("/node/wrapper/stream")
async def wrapper_node_stream(request: NodeRequest):
    """
    使用包装器节点进行流式处理
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        # 构造节点状态
        state = {
            "messages": [{"role": "user", "content": request.message}]
        }
        
        # 创建可流式输出的节点
        streamable_node = StreamableOutputNode()
        
        async def generate():
            async for event in streamable_node.stream_execute(state):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LangGraph流式节点API服务正在运行",
        "endpoints": {
            "/node/output/stream": "原有节点的流式输出",
            "/node/wrapper/stream": "包装器节点的流式输出",
            "/docs": "API文档"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("启动 LangGraph 流式节点服务...")
    print("访问 http://localhost:8001/docs 查看 API 文档")
    print("\n测试流式接口:")
    print("curl -X POST 'http://localhost:8001/node/output/stream' \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"message\": \"你好，请介绍一下自己\"}'")
    
    uvicorn.run(
        "graph_stream_integration:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
