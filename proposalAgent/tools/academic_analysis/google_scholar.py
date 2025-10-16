from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from serpapi import GoogleSearch 
from typing import Annotated

class SerpAPIScholar:
    def __init__(self, api_key: Optional[str] = None, hl: str = "en"):
        self.api_key = api_key or os.getenv("SERP_API_KEY")
        self.hl = hl
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY 未设置。请在环境变量中配置 SerpApi 的 API Key。")

    def search_scholar(
        self,
        q: str,
        *,
        as_ylo: Optional[int] = None,
        as_yhi: Optional[int] = None,
        num: int = 10,
        start: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "engine": "google_scholar",
            "q": q,
            "api_key": self.api_key,
            "hl": self.hl,
            "num": max(1, min(int(num), 20)),  # Scholar 单页最多 20 条
            "start": max(0, int(start)),
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

    def profiles(self, name: str, organization: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """注意：profiles 引擎官方标注 deprecated，这里仅作兜底使用。"""
        params: Dict[str, Any] = {
            "engine": "google_scholar_profiles",
            "mauthors": f"{name} {organization}" if organization else name,
            "api_key": self.api_key,
            "hl": self.hl,
        }
        params.update(kwargs)
        return GoogleSearch(params).get_dict()



def _extract_author_id_from_profile_link(link: str) -> Optional[str]:
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


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _initials(name: str) -> str:
    parts = [p for p in name.replace(".", " ").split() if p]
    return "".join(p[0] for p in parts).lower()


def _name_match_score(target: str, candidate: str, aliases: List[str]) -> int:
    t, c = _norm(target), _norm(candidate)
    if not c:
        return 0
    score = 0
    if t and (t == c):
        score += 3
    if aliases:
        alias_set = {_norm(a) for a in aliases}
        if c in alias_set:
            score += 3
        if _initials(candidate) in {_initials(a) for a in alias_set}:
            score += 1
    if t and (t in c or c in t):
        score += 1
    return score


def _org_match_score(org_hint: Optional[str], affiliations: Optional[str]) -> int:
    if not org_hint or not affiliations:
        return 0
    oh, af = _norm(org_hint), _norm(affiliations)
    score = 0
    if oh and oh in af:
        score += 4
    cas_alias = ["chinese academy of sciences", "中国科学院", "cas"]
    if any(a in oh for a in cas_alias) and any(a in af for a in cas_alias):
        score += 2
    return score

# =============================
# 论文简要信息
# =============================
class ArticleBriefInput(BaseModel):
    query: str = Field(..., description="论文标题（建议加引号）或关键词")
    year_from: Optional[int] = Field(None, description="起始年份过滤 as_ylo")
    year_to: Optional[int] = Field(None, description="结束年份过滤 as_yhi")
    top_k: int = Field(1, ge=1, le=20, description="返回条数 [1,20]")
    hl: str = Field("en", description="界面语言，如 zh-CN / en")
    
# =============================
# 指标获取（有/无 author_id）
# =============================
class AuthorCitationsInput(BaseModel):
    author_id: Optional[str] = Field(None, description="Scholar author_id（优先）")
    name: Optional[str] = Field(None, description="无 author_id 时的姓名兜底")
    organization: Optional[str] = Field(None, description="机构线索")
    hl: str = Field("en", description="界面语言")
    
# =============================
# 解析 author_id（核心）
# =============================
class ResolveAuthorInput(BaseModel):
    name: str = Field(..., description="目标学者姓名（可中文）")
    organization: Optional[str] = Field(None, description="机构线索（有助于消歧）")
    alias_names: Optional[List[str]] = Field(None, description="姓名别名/英文名/拼音")
    publication_titles: Optional[List[str]] = Field(None, description="若干代表作题名（建议提供英文题名）")
    hl: str = Field("en", description="界面语言，如 zh-CN / en")

class AuthorCitationsAutoInput(BaseModel):
    name: str = Field(..., description="目标学者姓名（可中文）")
    organization: Optional[str] = Field(None, description="机构线索（强烈建议提供）")
    alias_names: Optional[List[str]] = Field(None, description="姓名别名/英文名/拼音")
    publication_titles: Optional[List[str]] = Field(None, description="代表作题名（强烈建议至少 1 篇）")
    hl: str = Field("en", description="界面语言")

# =============================
#  作者每篇文章被引
# =============================
class AuthorArticlesInput(BaseModel):
    author_id: str = Field(..., description="Scholar author_id")
    hl: str = Field("en", description="界面语言")
    
    
def _extract_author_id_from_authors(authors: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for a in authors:
        link = (a or {}).get("profile") or (a or {}).get("link")
        if not link:
            continue
        aid = _extract_author_id_from_profile_link(link)
        if aid:
            pairs.append(((a or {}).get("name") or "", aid))
    return pairs

def _extract_article_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    pub_info = item.get("publication_info") or {}
    inline = item.get("inline_links") or {}
    cited_by = (inline.get("cited_by") or {}) if isinstance(inline, dict) else {}
    resources = item.get("resources") or []

    pdf = None
    for r in resources:
        try:
            if (r.get("file_format") or "").upper() == "PDF":
                pdf = r.get("link")
                break
        except Exception:
            pass

    authors: List[Dict[str, Optional[str]]] = []
    if isinstance(pub_info.get("authors"), list):
        for a in pub_info["authors"]:
            if isinstance(a, dict) and a.get("name"):
                authors.append({
                    "name": a.get("name"),
                    "profile": a.get("link"),  # 可能含有 ?user= 的 profile 链接
                })

    year = item.get("year") or pub_info.get("year")

    return {
        "title": item.get("title"),
        "snippet": item.get("snippet"),
        "link": item.get("link"),
        "pdf": pdf,
        "authors": authors,
        "publication": pub_info.get("summary"),
        "year": year,
        "cited_by": cited_by.get("total"),
        "cited_by_link": cited_by.get("link"),
        "versions_link": (inline.get("versions") or {}).get("link") if isinstance(inline, dict) else None,
        "result_id": item.get("result_id"),
    }

def _first_recent_value(d: Dict[str, Any]) -> Optional[int]:
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        if isinstance(k, str) and k.startswith("since_"):
            return v
    return None




# tool1
@tool
def get_article_brief(input: Annotated[ArticleBriefInput,"查询论文简要信息的入参,包括query、year_from、year_to、top_k、hl"]) -> List[Dict[str, Any]]:
    """根据文章标题标题，返回 Google Scholar 的前若干条结果简要信息。

    Args:
        input (Annotated[ArticleBriefInput,): 查询论文简要信息的入参,包括query、year_from、year_to、top_k、hl

    Returns:
        List[Dict[str, Any]]: 返回 Google Scholar 的前若干条结果简要信息。
    """
    client = SerpAPIScholar(hl=input.hl)
    resp = client.search_scholar(
        q=input.query,
        as_ylo=input.year_from,
        as_yhi=input.year_to,
        num=input.top_k,
    )
    items = resp.get("organic_results", [])[: input.top_k]
    return [_extract_article_brief(it) for it in items]


# tool 2
@tool
def resolve_author_candidates(input: Annotated[ResolveAuthorInput,"解析作者信息的入参,包括name、organization、alias_names、publication_titles、hl"]) -> List[Dict[str, Any]]:
    """根据姓名、机构、代表作题名解析 Google Scholar author_id，返回候选作者的列表与打分。

    Args:
        input (Annotated[ResolveAuthorInput,): 解析作者信息的入参,包括name、organization、alias_names、publication_titles、hl

    Returns:
        List[Dict[str, Any]]: 返回候选列表与打分。
    """
    client = SerpAPIScholar(hl=input.hl)
    aliases = list({input.name, *(input.alias_names or [])})

    candidate_map: Dict[str, Dict[str, Any]] = {}
    matched_titles_map: Dict[str, List[str]] = {}

    # 1) 代表作命中 → 从作者列表提取 profile → author_id
    for title in (input.publication_titles or []):
        if not title:
            continue
        resp = client.search_scholar(q=f'"{title}"', num=3)
        for it in resp.get("organic_results", [])[:3]:
            brief = _extract_article_brief(it)
            for cand_name, aid in _extract_author_id_from_authors(brief.get("authors") or []):
                s_name = _name_match_score(input.name, cand_name, aliases)
                cand = candidate_map.setdefault(aid, {"name": cand_name, "score": 0})
                cand["score"] += (5 + s_name)
                matched_titles_map.setdefault(aid, []).append(brief.get("title"))

    # 2) profiles 兜底（低权重）
    try:
        resp = client.profiles(input.name, input.organization)
        for p in (resp.get("profiles") or []):
            aid = _extract_author_id_from_profile_link(p.get("link") or "")
            if not aid:
                continue
            cand = candidate_map.setdefault(aid, {"name": p.get("name") or input.name, "score": 0})
            cand["score"] += 2
    except Exception:
        pass

    # 3) 拉 author 详情，机构加分 & 指标提供
    results: List[Dict[str, Any]] = []
    for aid, meta in candidate_map.items():
        try:
            data = client.author(aid)
            author = data.get("author") or {}
            aff = author.get("affiliations")
            meta_score = meta.get("score", 0) + _org_match_score(input.organization, aff)

            cited_by = data.get("cited_by") or {}
            table = cited_by.get("table") or []
            citations_all = h_all = i10_all = None
            for row in table:
                if "citations" in row:
                    citations_all = (row["citations"] or {}).get("all")
                if "h_index" in row:
                    h_all = (row["h_index"] or {}).get("all")
                if "i10_index" in row:
                    i10_all = (row["i10_index"] or {}).get("all")

            results.append({
                "author_id": aid,
                "name": author.get("name") or meta.get("name"),
                "affiliations": aff,
                "citations_all": citations_all,
                "h_index_all": h_all,
                "i10_all": i10_all,
                "matched_titles": matched_titles_map.get(aid, []),
                "score": meta_score,
            })
        except Exception:
            results.append({
                "author_id": aid,
                "name": meta.get("name"),
                "affiliations": None,
                "citations_all": None,
                "h_index_all": None,
                "i10_all": None,
                "matched_titles": matched_titles_map.get(aid, []),
                "score": meta.get("score", 0),
            })

    results.sort(key=lambda x: (x.get("score") or 0, x.get("citations_all") or 0), reverse=True)
    return results

# tool 3
@tool
def get_author_citations(input: Annotated[AuthorCitationsInput,"查询作者被引信息的入参,包括author_id、name、organization、hl"]) -> Dict[str, Any]:
    """返回作者的总被引、近5年被引、h-index、i10-index 与按年引用数。

    Args:
        input (Annotated[AuthorCitationsInput,): 查询作者被引信息的入参,包括author_id、name、organization、hl

    Raises:
        ValueError: 缺少 author_id，且无法仅凭姓名解析。建议使用 scholar_author_citations_auto 并提供代表作题名。
        ValueError: 缺少 author_id。

    Returns:
        Dict[str, Any]: 返回作者的总被引、近5年被引、h-index、i10-index 与按年引用数。
    """
    client = SerpAPIScholar(hl=input.hl)
    author_id = input.author_id

    if not author_id and input.name:
        cands = resolve_author_candidates.invoke({
            "input": ResolveAuthorInput(name=input.name, organization=input.organization, hl=input.hl)
        })
        if cands:
            author_id = cands[0]["author_id"]
        else:
            raise ValueError("缺少 author_id，且无法仅凭姓名解析。建议使用 scholar_author_citations_auto 并提供代表作题名。")

    if not author_id:
        raise ValueError("缺少 author_id。")

    data = client.author(author_id)
    author = data.get("author") or {}
    cited_by = data.get("cited_by") or {}

    totals = {
        "citations_all": None,
        "citations_5y": None,
        "h_index_all": None,
        "h_index_5y": None,
        "i10_all": None,
        "i10_5y": None,
    }
    for row in (cited_by.get("table") or []):
        if "citations" in row:
            totals["citations_all"] = (row["citations"] or {}).get("all")
            totals["citations_5y"] = _first_recent_value(row["citations"])  # since_XXXX
        if "h_index" in row:
            totals["h_index_all"] = (row["h_index"] or {}).get("all")
            totals["h_index_5y"] = _first_recent_value(row["h_index"])      # since_XXXX
        if "i10_index" in row:
            totals["i10_all"] = (row["i10_index"] or {}).get("all")
            totals["i10_5y"] = _first_recent_value(row["i10_index"])       # since_XXXX

    graph = cited_by.get("graph") or []  # [{year: 2018, citations: 123}, ...]

    return {
        "author_id": author_id,
        "name": author.get("name"),
        "affiliations": author.get("affiliations"),
        "thumbnail": author.get("thumbnail"),
        **totals,
        "citations_by_year": graph
    }
    
# tool 4
@tool
def get_author_citations_auto(input: Annotated[AuthorCitationsAutoInput,"查询作者被引信息的入参,不需要知道author_id,包括name、organization、alias_names、publication_titles、hl"]) -> Dict[str, Any]:
    """根据姓名、机构、代表作题名解析 Google Scholar author_id，返回作者的总被引、近5年被引、h-index、i10-index 与按年引用数。

    Raises:
        ValueError: 未能解析任何候选 author_id，请补充英文题名或机构别名后重试。

    Returns:
        Dict[str, Any]: 返回候选列表与打分。
    """
    cands = resolve_author_candidates.invoke({
        "input": ResolveAuthorInput(
            name=input.name,
            organization=input.organization,
            alias_names=input.alias_names,
            publication_titles=input.publication_titles,
            hl=input.hl,
        )
    })
    if not cands:
        raise ValueError("未能解析任何候选 author_id，请补充英文题名或机构别名后重试。")

    best = cands[0]
    metrics = get_author_citations.invoke({
        "input": AuthorCitationsInput(author_id=best["author_id"], hl=input.hl)
    })
    return {"resolution": {"selected": best, "candidates": cands}, "metrics": metrics}


# tool 5
@tool
def get_author_articles_citations(input: Annotated[AuthorArticlesInput,"查询作者每篇文章被引信息的入参,包括author_id、hl"]) -> List[Dict[str, Any]]:
    """返回作者每篇文章被引信息。

    Args:
        input (Annotated[AuthorArticlesInput,): 查询作者每篇文章被引信息的入参,包括author_id、hl

    Returns:
        List[Dict[str, Any]]: 返回作者每篇文章被引信息。
    """
    client = SerpAPIScholar(hl=input.hl)
    data = client.author(input.author_id)
    articles = data.get("articles") or []
    out: List[Dict[str, Any]] = []
    for a in articles:
        out.append({
            "title": a.get("title"),
            "year": a.get("year"),
            "link": a.get("link"),
            "citation_id": a.get("citation_id"),
            "cited_by": ((a.get("cited_by") or {}).get("value")),
        })
    out.sort(key=lambda x: (x.get("cited_by") or 0), reverse=True)
    return out