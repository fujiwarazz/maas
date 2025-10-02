from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="角色，例如 user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatMessageRequest(BaseModel):
    thread_id: Optional[str] = Field(None, description="会话线程ID，为空则新建")
    message: str = Field(..., description="用户输入")
    history_limit: Optional[int] = Field(None, description="最多保留的历史条数")


class ChatMessageResponse(BaseModel):
    thread_id: str
    reply: ChatMessage
    history: List[ChatMessage]
