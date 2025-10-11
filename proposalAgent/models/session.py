from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    file_id: str = Field(..., description="上传文件的唯一标识")
    filename: str = Field(..., description="原始文件名")
    content_type: Optional[str] = Field(None, description="文件类型")
    size: int = Field(..., description="文件大小，单位字节")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    file_ids: List[str] = Field(..., description="关联的文件标识")
    query: str = Field(..., description="用户提问/任务描述")
    user_interest: List[str] = Field(default_factory=list, description="用户关注点列表")
    thread_id: Optional[str] = Field(None, description="用于恢复会话的线程ID")


class ChatMessageRequest(BaseModel):
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    message: str = Field(..., description="用户输入的消息")
    history_limit: Optional[int] = Field(
        None, description="可选的历史截断窗口，控制上下文长度"
    )


class SSEEvent(BaseModel):
    event: str = Field(..., description="事件类型，例如 message/complete/error")
    data: str = Field(..., description="事件携带的数据，通常为JSON字符串")
    id: Optional[str] = Field(None, description="可选的事件ID，用于断点续传")


class SessionControl(BaseModel):
    thread_id: str = Field(..., description="会话线程ID")
    action: str = Field(..., description="控制动作，支持 resume/cancel")
    feedback: Optional[str] = Field(None, description="当 action=resume 时的反馈内容")
    file_id: Optional[str] = Field(
        None,
        description="可选：当存在多文件子会话时，用于指定目标文件ID",
    )


