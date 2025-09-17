"""
Google Scholar 工具（SerpAPI 版），用于 LangGraph/Agent 调用。

功能：
1) scholar_article_brief：查看某篇文章的简要内容（标题、摘要片段、作者、年份、引用数、链接、PDF）。
2) scholar_author_citations：查看某位学者的总被引、近5年被引、h-index、i10-index 与按年引用数。

依赖：
- 环境变量 SERPAPI_API_KEY
- pip install google-search-results pydantic

文档提示：
- Google Scholar 搜索：/search?engine=google_scholar
- Author 详情（含 cited_by 表）：/search?engine=google_scholar_author
- Profiles 搜索（已官方标注为 deprecated，可能不可用）：/search?engine=google_scholar_profiles
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
import os
from pydantic import BaseModel, Field

try:
    from serpapi import GoogleSearch  # pip install google-search-results
except Exception as e:
    raise RuntimeError(
        "未安装 google-search-results，请先执行：pip install google-search-results"
    ) from e


# -----------------------------
# SerpAPI 基础客户端封装
# -----------------------------
class SerpAPIScholar:
    """SerpAPI 的 Google Scholar 客户端封装。"""

    def __init__(self, api_key: Optional[str] = None, hl: str = "en"):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise RuntimeError("SERPAPI_API_KEY 未设置。请在环境变量中配置 SerpApi 的 API Key。")
        self.hl = hl

    def search_scholar(
        self,
        q: str,
        as_ylo: Optional[int] = None,
        as_yhi: Optional[int] = None,
        num: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "engine": "google_scholar",
            "q": q,
            "api_key": self.api_key,
            "hl": self.hl,
            "num": max(1, min(int(num), 20)),  # Scholar 单页最多 20 条
        }
        if as_ylo:
            params["as_ylo"] = int(as_ylo)
        if as_yhi:
            params["as_yhi"] = int(as_yhi)
        params.update(kwargs)
        return GoogleSearch(params).get_dict()

    def author(self, author_id: str, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "engine": "google_scholar_author",
            "author_id": author_id,
            "api_key": self.api_key,
            "hl": self.hl,
        }
        params.update(kwargs)
        return GoogleSearch(params).get_dict()

    def try_profiles(self, name: str, organization: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        """
        可选：通过 Profiles 搜索尝试解析 author_id（注意官方标注为 deprecated，可能不可用）。
        成功时返回 author_id（即 Profile 链接中 ?user= 的值），失败返回 None。
        """
        params: Dict[str, Any] = {
            "engine": "google_scholar_profiles",
            "mauthors": name,
            "api_key": self.api_key,
            "hl": self.hl,
        }
        if organization:
            # 将机构拼进查询，提升消歧效果
            params["mauthors"] = f"{name} {organization}"
        params.update(kwargs)
        try:
            resp = GoogleSearch(params).get_dict()
            profiles = resp.get("profiles", []) or []
            # 优先匹配机构
            if organization:
                for p in profiles:
                    aff = (p.get("affiliations") or "").lower()
                    if organization.lower() in aff:
                        link = p.get("link") or ""
                        auth = _extract_author_id_from_profile_link(link)
                        if auth:
                            return auth
            # 退化为第一个 Profile
            if profiles:
                link = profiles[0].get("link") or ""
                return _extract_author_id_from_profile_link(link)
        except Exception:
            return None
        return None


def _extract_author_id_from_profile_link(link: str) -> Optional[str]:
    """从 profile 链接中提取 ?user= 的 author_id。"""
    try:
        import urllib.parse as up

        parsed = up.urlparse(link)
        qs = up.parse_qs(parsed.query)
        user = qs.get("user")
        if user and isinstance(user, list) and user[0]:
            return user[0]
    except Exception:
        return None
    return None


# -----------------------------
# 文章简要内容提取
# -----------------------------

def _extract_article_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    pub_info = item.get("publication_info") or {}
    inline = item.get("inline_links") or {}
    cited_by = (inline.get("cited_by") or {}) if isinstance(inline, dict) else {}
    resources = item.get("resources") or []

    # PDF 链接（若存在）
    pdf = None
    for r in resources:
        try:
            if (r.get("file_format") or "").upper() == "PDF":
                pdf = r.get("link")
                break
        except Exception:
            continue

    # 作者列表
    authors: List[str] = []
    try:
        if isinstance(pub_info.get("authors"), list):
            authors = [
                a.get("name")
                for a in pub_info["authors"]
                if isinstance(a, dict) and a.get("name")
            ]
    except Exception:
        authors = []

    year = item.get("year")
    if not year and isinstance(pub_info, dict):
        year = pub_info.get("year")

    return {
        "title": item.get("title"),
        "snippet": item.get("snippet"),  # Scholar 提供的摘要片段（非完整摘要）
        "link": item.get("link"),
        "pdf": pdf,
        "authors": authors,
        "publication": pub_info.get("summary"),  # 期刊/会议+页码等摘要信息
        "year": year,
        "cited_by": cited_by.get("total"),
        "cited_by_link": cited_by.get("link"),
        "versions_link": (inline.get("versions") or {}).get("link") if isinstance(inline, dict) else None,
        "result_id": item.get("result_id"),
    }


class ArticleBriefInput(BaseModel):
    """查看文章简要内容的入参。"""

    query: str = Field(..., description="论文标题（建议加引号）或关键词")
    year_from: Optional[int] = Field(None, description="起始年份过滤 as_ylo")
    year_to: Optional[int] = Field(None, description="结束年份过滤 as_yhi")
    top_k: int = Field(1, ge=1, le=20, description="返回条数 [1,20]")
    hl: str = Field("en", description="界面语言，如 zh-CN / en")


def get_article_brief(input: ArticleBriefInput) -> List[Dict[str, Any]]:
    """根据标题或关键词，返回 Google Scholar 的前若干条结果简要信息。"""
    client = SerpAPIScholar(hl=input.hl)
    resp = client.search_scholar(
        q=input.query,
        as_ylo=input.year_from,
        as_yhi=input.year_to,
        num=input.top_k,
    )
    items = resp.get("organic_results", [])[: input.top_k]
    return [_extract_article_brief(it) for it in items]


# -----------------------------
# 作者被引 / h-index / i10-index
# -----------------------------

def _first_recent_value(d: Dict[str, Any]) -> Optional[int]:
    """从 {all: x, since_2016: y, ...} 里取任意 since_* 的值（键名会随年份滚动）。"""
    if not isinstance(d, dict):
        return None
    # 优先取含 since_ 的键
    for k, v in d.items():
        if isinstance(k, str) and k.startswith("since_"):
            return v
    return None


class AuthorCitationsInput(BaseModel):
    """查看作者引用指标的入参。优先提供 author_id。"""

    author_id: Optional[str] = Field(
        None, description="Google Scholar author_id，例如 EicYvbwAAAAJ。强烈建议直接提供。"
    )
    name: Optional[str] = Field(None, description="作者姓名。当没有 author_id 时用于尝试解析")
    organization: Optional[str] = Field(None, description="可选：机构，用于消歧")
    hl: str = Field("en", description="界面语言，如 zh-CN / en")


def get_author_citations(input: AuthorCitationsInput) -> Dict[str, Any]:
    """返回作者的总被引、近5年被引、h-index、i10-index 与按年引用数。"""
    client = SerpAPIScholar(hl=input.hl)
    author_id = input.author_id

    # 尝试通过 Profiles 解析 author_id（可能失败）
    if not author_id and input.name:
        author_id = client.try_profiles(input.name, input.organization)

    if not author_id:
        raise ValueError(
            "未提供 author_id，且无法通过 name/organization 解析。请直接传入 author_id（见 Scholar 个人主页 URL 的 ?user= 值）。"
        )

    data = client.author(author_id)
    author = data.get("author") or {}
    cited_by = data.get("cited_by") or {}

    # 表格汇总指标
    totals = {
        "citations_all": None,
        "citations_5y": None,
        "h_index_all": None,
        "h_index_5y": None,
        "i10_all": None,
        "i10_5y": None,
    }
    table = cited_by.get("table") or []
    for row in table:
        if "citations" in row:
            totals["citations_all"] = (row["citations"] or {}).get("all")
            totals["citations_5y"] = _first_recent_value(row["citations"])  # 动态 since_*
        if "h_index" in row:
            totals["h_index_all"] = (row["h_index"] or {}).get("all")
            totals["h_index_5y"] = _first_recent_value(row["h_index"])  # 动态 since_*
        if "i10_index" in row:
            totals["i10_all"] = (row["i10_index"] or {}).get("all")
            totals["i10_5y"] = _first_recent_value(row["i10_index"])  # 动态 since_*

    graph = cited_by.get("graph") or []  # [{year: 2018, citations: 123}, ...]

    return {
        "author_id": author_id,
        "name": author.get("name"),
        "affiliations": author.get("affiliations"),
        "thumbnail": author.get("thumbnail"),
        **totals,
        "citations_by_year": graph,
    }


# -----------------------------
# LangChain / LangGraph 工具包装（可选）
# -----------------------------
try:
    from langchain_core.tools import StructuredTool

    def to_langchain_tools():
        """将本模块函数导出为 LangChain 工具，方便在 LangGraph 的 ToolNode 中使用。"""
        return [
            StructuredTool.from_function(
                func=get_article_brief,
                name="scholar_article_brief",
                description="查看某篇文章的简要内容（标题、摘要片段、作者、年份、引用次数、链接）。",
                args_schema=ArticleBriefInput,
                return_direct=False,
            ),
            StructuredTool.from_function(
                func=get_author_citations,
                name="scholar_author_citations",
                description="查看某位学者的总被引、h-index、i10-index 与按年引用数。",
                args_schema=AuthorCitationsInput,
                return_direct=False,
            ),
        ]
except Exception:
    # 不强依赖 LangChain
    def to_langchain_tools():  # type: ignore
        raise RuntimeError("未安装 langchain-core，或版本不兼容。可忽略此函数，直接调用纯 Python 接口。")


# -----------------------------
# 快速命令行测试
# -----------------------------
if __name__ == "__main__":
    import json

    print("== Article brief demo ==")
    ab = get_article_brief(
        ArticleBriefInput(query='Meng Xiao; Min Wu; Ziyue Qiao; Yanjie Fu; Zhiyuan Ning; Yi Du; Yuanchun Zhou; Interdisciplinary Fairness in Imbalanced Research Proposal Topic Inference: A Hierarchical Transformer-based Method with Selective Interpolation', top_k=1, hl="en")
    )
    print(json.dumps(ab, ensure_ascii=False, indent=2))

    print("\n== Author citations demo ==")
    # 示例 author_id：请替换为真实的（如从个人主页 ?user= 后复制）
    try:
        ac = get_author_citations(AuthorCitationsInput(author_id="VjJkNh8AAAAJ", hl="en"))
        print(json.dumps(ac, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Author demo failed:", e)
