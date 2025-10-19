from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import os
import logging
from tavily import TavilyClient
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TavilySearchInput(BaseModel):
    query: str = Field(..., description="搜索关键词，建议尽量具体")
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="控制 Tavily 搜索深度：basic 或 advanced",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回结果条数上限 (1-20)",
    )
    include_raw_content: bool = Field(
        default=False,
        description="是否在结果中包含页面原始内容",
    )
    include_images: bool = Field(
        default=False,
        description="是否返回 Tavily 提供的图片摘要",
    )
    include_answer: bool = Field(
        default=False,
        description="是否返回 Tavily 聚合的答案",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="可选的 Tavily API key，未提供时会读取环境变量",
    )

    @model_validator(mode="after")
    def _strip_query(self) -> "TavilySearchInput":
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query 不能为空字符串")
        return self


def _get_client(api_key: Optional[str] = None) -> TavilyClient:
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY 未配置，请设置环境变量或传入 api_key。")
    return TavilyClient(key)


def _format_result(item: Dict[str, Any], include_raw_content: bool) -> Dict[str, Any]:
    return {
        "title": item.get("title"),
        "url": item.get("url"),
        "content": item.get("content"),
        "score": item.get("score"),
        "published_date": item.get("published"),
        "raw_content": item.get("raw_content") if include_raw_content else None,
    }


@tool("tavily_search", args_schema=TavilySearchInput)
def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_raw_content: bool = False,
    include_images: bool = False,
    include_answer: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """基于 Tavily API 的通用网页搜索工具。"""
    try:
        client = _get_client(api_key)
        raw = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_raw_content=include_raw_content,
            include_images=include_images,
            include_answer=include_answer,
        )
    except Exception as exc:
        logger.error("Tavily 搜索失败: %s", exc)
        return {"error": f"Tavily 搜索失败: {exc}"}

    results = raw.get("results") or []
    formatted: List[Dict[str, Any]] = [
        _format_result(r, include_raw_content=include_raw_content) for r in results
    ]

    return {
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_raw_content": include_raw_content,
        "include_images": include_images,
        "include_answer": include_answer,
        "results": formatted,
        "answer": raw.get("answer"),
        "images": raw.get("images"),
        "follow_up_questions": raw.get("follow_up_questions"),
        "raw_response": raw,
    }


