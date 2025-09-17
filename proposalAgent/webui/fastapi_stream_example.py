#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI 流式响应示例
展示如何将 output.py 中的流式输出集成到 FastAPI 中
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from proposalAgent.agents.stage1.output import create_stream_generator, create_async_stream_generator

app = FastAPI(title="AI流式对话API", description="基于LangGraph节点的流式AI对话服务")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    success: bool
    message: str


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口 - 使用 Server-Sent Events (SSE)
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        # 使用异步生成器
        generator = create_async_stream_generator(request.message)
        
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


@app.post("/chat/stream/sync")
def chat_stream_sync(request: ChatRequest):
    """
    同步流式聊天接口 - 使用 Server-Sent Events (SSE)
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        
        # 使用同步生成器
        generator = create_stream_generator(request.message)
        
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


@app.get("/")
async def root():
    """根路径"""
    return {"message": "AI流式对话API服务正在运行"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "AI流式对话API"}


if __name__ == "__main__":
    import uvicorn
    
    print("启动 FastAPI 流式对话服务...")
    print("访问 http://localhost:8000/docs 查看 API 文档")
    print("\n测试流式接口:")
    print("curl -X POST 'http://localhost:8000/chat/stream' \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"message\": \"你好，请介绍一下自己\"}'")
    
    uvicorn.run(
        "fastapi_stream_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
