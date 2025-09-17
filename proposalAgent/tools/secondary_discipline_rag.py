"""
Secondary discipline RAG tool for LangChain.

This tool embeds the input text using a DashScope-compatible OpenAI endpoint and
searches a Milvus collection (stored locally via uri like './discipline.db') for
the most relevant secondary disciplines. It returns a comma-separated list of
Chinese secondary discipline names suitable for your interdis agent output.
"""

from __future__ import annotations

import os
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool,tool
from openai import OpenAI
from pymilvus import MilvusClient


def _get_sync_embedding(
    text: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "text-embedding-v4",
    dimensions: int = 1024,
) -> List[float]:
    """Get a single embedding vector synchronously."""
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url=base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    resp = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions,
        encoding_format="float",
    )
    return resp.data[0].embedding  # type: ignore


def _search_secondary_disciplines(
    *,
    text: str,
    top_k: int = 10,
    return_codes: bool = False,
    milvus_uri: str = "./discipline.db",
    collection_name: str = "second_level_disciplines",
) -> str:
    """
    Embed the text and search Milvus for matching secondary disciplines.
    Returns a comma-separated string of names (or code + name if return_codes=True).
    """
    emb = _get_sync_embedding(text)
    client = MilvusClient(uri=milvus_uri)

    try:
        results = client.search(
            collection_name=collection_name,
            data=[emb],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=["discipline_code", "discipline_name"],
        )
    except Exception as e:
        # Return empty to avoid breaking agent flow
        return ""

    names: List[str] = []
    seen = set()
    for hits in results:
        for h in hits:
            code = h.get('entity',{}).get("discipline_code", None)
            name = h.get('entity',{}).get("discipline_name", None)
            if not name:
                continue
            key = (code or "", name)
            if key in seen:
                continue
            seen.add(key)
            names.append(f"{code} {name}".strip() if return_codes and code else name)

    return ", ".join(names)


# def build_secondary_discipline_tool(
#     *,
#     milvus_uri: str = "./discipline.db",
#     collection_name: str = "second_level_disciplines",
# ) -> StructuredTool:
#     """
#     Create a LangChain StructuredTool for secondary discipline retrieval.

#     Usage:
#         tool = build_secondary_discipline_tool()
#         tools = [tool]
#         # Bind to an agent that supports tool execution (e.g., AgentExecutor)
#     """

#     def _tool_impl(text: str, top_k: int = 10, return_codes: bool = False) -> str:
#         return _search_secondary_disciplines(
#             text=text,
#             top_k=top_k,
#             return_codes=return_codes,
#             milvus_uri=milvus_uri,
#             collection_name=collection_name,
#         )

#     return StructuredTool.from_function(
#         name="secondary_discipline_rag",
#         description=(
#             "根据输入的研究文本，检索Milvus中的二级学科并返回最匹配的中文名称列表（逗号分隔）。"
#             "可用于在识别出领域后，对齐标准二级学科名称。"
#         ),
#         func=_tool_impl,
#         args_schema=_SecondaryDisciplineArgs,
#         return_direct=False,
#     )


@tool
def secondary_discipline_search(
    text: str,
    top_k: int = 10,
    return_codes: bool = False,
) -> str:
    """ 根据输入的研究文本，检索Milvus中的二级学科并返回最匹配的中文名称列表（逗号分隔）。可用于在识别出领域后，对齐标准二级学科名称。

    Args:
        text (str): 输入文本
        top_k (int, optional): 召回的二级学科数目. Defaults to 10.
        return_codes (bool, optional): 返回的代码. Defaults to False.

    Returns:
        str: 逗号分隔的二级学科名称列表
    """
    return _search_secondary_disciplines(
        text=text,
        top_k=top_k,
        return_codes=return_codes,
        milvus_uri= "./discipline.db",
        collection_name='second_level_disciplines',
    )
