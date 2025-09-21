
"""
Web of Science 工具（Starter API + Expanded API）

用途：作为 LangGraph/Agent 的工具节点，统一封装 Clarivate Web of Science 的两套 REST API：
- Starter API（轻量、基础元数据与 times-cited 统计）
- Expanded API（完整记录、引用/被引/相关文献等高级操作）

依赖：
- 环境变量：CLARIVATE_API_KEY  或在构造 Client 时显式传入 api_key。
- pip install requests pydantic

注意：
- Starter API 端点（示例）：https://api.clarivate.com/apis/wos-starter/v1
  - /documents
  - /documents/{uid}
- Expanded API 端点（示例）：https://api.clarivate.com/api/wos
  - 根搜索：       GET /api/wos
  - 被引参考文献： GET /api/wos/references
  - 施引文献：     GET /api/wos/citing
  - 相关文献：     GET /api/wos/related

参数命名提示：
- Starter：q（高级检索语法，如 DT=Article）、db（库，如 WOS）、limit、page、sortField（例如 LD+D）、detail 等。
- Expanded：usrQuery（高级检索语法，如 ts=cadmium 或 UT=(WOS:001...)）、databaseId、count、firstRecord、viewField（如 UID pub_info keywords）、optionView（FS=Full，SR=Short）。

下方同时提供 LangChain StructuredTool 包装（可选）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
import os
from langchain_core.tools import tool
import requests
from pydantic import BaseModel, Field


class _HTTPError(RuntimeError):
    pass


def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    key = api_key or os.getenv("CLARIVATE_API_KEY")
    if not key:
        raise _HTTPError("缺少 API Key。请设置环境变量 CLARIVATE_API_KEY 或在构造函数中传入 api_key。")
    return {
        "accept": "application/json",
        "X-ApiKey": key,
    }


def _request_json(method: str, url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    try:
        resp = requests.request(method, url, headers=headers, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise _HTTPError(f"请求失败：{e}") from e

    if resp.status_code // 100 != 2:
        snippet = resp.text[:400]
        raise _HTTPError(f"HTTP {resp.status_code}: {snippet}")

    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}



STARTER_BASE = "https://api.clarivate.com/apis/wos-starter/v1"

class StarterDocumentsInput(BaseModel):
    q: str = Field(..., description="高级检索表达式，如 DT=Article")
    db: str = Field("WOS", description="数据库代码，示例：WOS")
    limit: int = Field(10, ge=1, le=50, description="每页条数，1..50，默认 10")
    page: int = Field(1, ge=1, description="页码，从 1 开始")
    sort_field: Optional[str] = Field(None, description="排序，形如 PY+D 或 LD+D（字段+方向，A=升序，D=降序）")
    detail: Optional[bool] = Field(None, description="是否返回更详细字段（True/False）")
    modified_time_span: Optional[str] = Field(None, description="过滤元数据更新时间范围，如 P7D/P30D")
    tc_modified_time_span: Optional[str] = Field(None, description="过滤 times-cited 更新时间范围")


EXPANDED_BASE = os.getenv("WOS_EXPANDED_BASE", "https://api.clarivate.com/api/wos")

class ExpandedSearchInput(BaseModel):
    usr_query: str = Field(..., description="高级检索语法，如 ts=cadmium 或 UT=(WOS:001...)")
    database_id: str = Field("WOS", description="数据库，如 WOS")
    count: int = Field(10, ge=1, le=100, description="返回条数（每次返回记录数）")
    first_record: int = Field(1, ge=1, description="起始记录序号（从 1 开始）")
    view_field: Optional[List[str]] = Field(None, description="仅返回所需字段，例如 ['UID','pub_info','title']")
    option_view: Optional[str] = Field(None, description="FS=Full，SR=Short（Short Record）")
    lang: Optional[str] = Field(None, description="界面语言/记录语言偏好，例如 en")



@tool
def wos_expanded_search(input: ExpandedSearchInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """根据Wos的语法进行查询，可以通过这个方法来查询文献的WOS编号,从而进行后续操作
     例如：TI=("Hierarchical Interdisciplinary Topic Detection Model for Research Proposal Classification") AND AU=(Du Yi OR "Yi Du")
     参数如:TI:标题 AU:作者 DT:文献类型 PY:年份 PYD:年份降序

    Args:
        input (ExpandedSearchInput): 查询的如惨，包括usr_query、database_id、count、first_record、view_field、option_view、lang。其中user_query是高级检索语法，database_id是数据库，count是返回条数，first_record是起始记录序号，view_field是仅返回所需字段，option_view是视图，lang是界面语言/记录语言偏好。
        api_key (Optional[str], optional): _description_. Defaults to None.
        base_url (str, optional): _description_. Defaults to EXPANDED_BASE.

    Returns:
        Dict[str, Any]: _description_
    """
    headers = _build_headers(api_key)
    url = base_url
    params: Dict[str, Any] = {
        "databaseId": input.database_id,
        "usrQuery": input.usr_query,
        "count": input.count,
        "firstRecord": input.first_record,
    }
    if input.view_field:
        params["viewField"] = "+".join(input.view_field)
    if input.option_view:
        params["optionView"] = input.option_view
    if input.lang:
        params["lang"] = input.lang

    return _request_json("GET", url, headers=headers, params=params)


class ExpandedGetByUTInput(BaseModel):
    ut: str = Field(..., description="WOS 的 UT/UID，如 WOS:001321517500001")
    database_id: str = Field("WOS", description="数据库，如 WOS")
    view_field: Optional[List[str]] = Field(None, description="需要的字段列表")
    option_view: Optional[str] = Field(None, description="FS 或 SR")


@tool
def wos_expanded_get_by_ut(input: ExpandedGetByUTInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """按 UT 精确获取记录（内部仍调用根搜索）。

    Args:
        input (ExpandedGetByUTInput): _description_
        api_key (Optional[str], optional): _description_. Defaults to None.
        base_url (str, optional): _description_. Defaults to EXPANDED_BASE.

    Returns:
        Dict[str, Any]: _description_
    """
    
    q = f"UT=({input.ut})" if not input.ut.startswith("UT=") else input.ut
    return wos_expanded_search(
        ExpandedSearchInput(
            usr_query=q,
            database_id=input.database_id,
            count=1,
            first_record=1,
            view_field=input.view_field,
            option_view=input.option_view,
        ),
        api_key=api_key,
        base_url=base_url,
    )


class ExpandedByUTBase(BaseModel):
    ut: str = Field(..., description="父记录的 UT/UID，如 WOS:001321517500001")
    database_id: str = Field("WOS", description="数据库，如 WOS")
    count: int = Field(100, ge=1, le=100, description="每次返回条数")
    first_record: int = Field(1, ge=1, description="起始序号（从 1 开始）")
    view_field: Optional[List[str]] = Field(None, description="需要的字段列表")
    option_view: Optional[str] = Field(None, description="FS 或 SR")
    lang: Optional[str] = Field(None, description="语言，如 en")
    by: Literal['usrQuery','uniqueId'] = Field('uniqueId', description="调用子端点时使用 uniqueId 还是 usrQuery")
    created_time_span: Optional[str] = Field(None, description="可选：限制施引/相关的创建时间范围，如 2024-01-01 2025-09-10")


_ROUTE_MAP = {
    "citedReferences": "references",
    "citingArticles": "citing",
    "related": "related",
}


def _expanded_child_endpoint(
    endpoint: str,
    input: ExpandedByUTBase,
    *,
    api_key: Optional[str],
    base_url: str,
) -> Dict[str, Any]:
    headers = _build_headers(api_key)
    real_endpoint = _route = _ROUTE_MAP.get(endpoint, endpoint).strip('/')
    url = f"{base_url}/{real_endpoint}"

    params: Dict[str, Any] = {
        "databaseId": input.database_id,
        "firstRecord": input.first_record,
        "count": input.count,
    }
    if input.by == 'uniqueId':
        params["uniqueId"] = input.ut
    else:
        params["usrQuery"] = f"UT=({input.ut})" if not input.ut.startswith("UT=") else input.ut

    if input.view_field:
        params["viewField"] = "+".join(input.view_field)
    if input.option_view:
        params["optionView"] = input.option_view
    if input.lang:
        params["lang"] = input.lang
    if input.created_time_span and real_endpoint in {"citing", "related"}:
        params["createdTimeSpan"] = input.created_time_span

    return _request_json("GET", url, headers=headers, params=params)


class ExpandedCitedReferencesInput(ExpandedByUTBase):
    pass


def wos_expanded_cited_references(input: ExpandedCitedReferencesInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """获取指定 UT 的参考文献。等价于 GET /api/wos/references。"""
    return _expanded_child_endpoint("citedReferences", input, api_key=api_key, base_url=base_url)


class ExpandedCitingArticlesInput(ExpandedByUTBase):
    pass


def wos_expanded_citing_articles(input: ExpandedCitingArticlesInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """获取指定 UT 的施引文献。等价于 GET /api/wos/citing。"""
    return _expanded_child_endpoint("citingArticles", input, api_key=api_key, base_url=base_url)


class ExpandedRelatedRecordsInput(ExpandedByUTBase):
    pass


def wos_expanded_related_records(input: ExpandedRelatedRecordsInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """获取指定 UT 的相关文献。等价于 GET /api/wos/related。"""
    return _expanded_child_endpoint("related", input, api_key=api_key, base_url=base_url)



from collections import defaultdict, deque

class CitationFanoutInput(BaseModel):
    ut: str = Field(..., description="起始论文的 UT/UID，如 WOS:001321517500001,参数可从wos_expanded_search的返回对象中获取")
    database_id: str = Field("WOS", description="数据库，如 WOS")
    max_depth: int = Field(1, ge=1, le=3, description="向外展开的层数（1=仅直接施引文献）")
    per_request: int = Field(100, ge=1, le=100, description="每次请求条数（Expanded 上限 100）")
    max_per_level: int = Field(500, ge=1, description="每一层最多抓取多少条，避免爆炸")
    option_view: str = Field("SR", description="视图：SR=Short Record（省配额），FS=Full Record")
    view_field: List[str] = Field(
        default_factory=lambda: ["UID","titles","pub_info","source","categories"],
        description="尽量包含 UID/标题/期刊/年份/学科等摘要用字段",
    )
    lang: Optional[str] = Field("en")
    created_time_span: Optional[str] = Field(None, description="可选：限制施引/相关的创建时间范围，如 2024-01-01 2025-09-10")


def _extract_meta_from_record(rec):
    uid = rec.get("UID") or rec.get("uid") or rec.get("ut")
    title = journal = None
    year = None
    cats = None

    # 先尝试扁平字段
    titles = rec.get("titles") or rec.get("title") or []
    if isinstance(titles, list):
        for t in titles:
            if isinstance(t, dict) and t.get("type") in {"item","title"}:
                title = t.get("content") or t.get("value") or t.get("title")
                if title: break
    elif isinstance(titles, dict):
        title = titles.get("content") or titles.get("value") or titles.get("title")

    src = rec.get("source")
    if isinstance(src, dict):
        journal = src.get("title") or src.get("source")

    pub_info = rec.get("pub_info") or {}
    year = pub_info.get("pubyear") or pub_info.get("year")

    # —— 兜底：从 static_data.summary.* 抽
    sd = rec.get("static_data") or {}
    sm = sd.get("summary") or {}
    if not year and isinstance(sm.get("pub_info"), dict):
        year = sm["pub_info"].get("pubyear") or sm["pub_info"].get("year")

    if not (title and journal) and isinstance(sm.get("titles"), dict):
        for t in sm["titles"].get("title", []):
            if t.get("type") in {"item","title"} and not title:
                title = t.get("content") or t.get("value") or t.get("title")
            if t.get("type") in {"source","source_title","journal"} and not journal:
                journal = t.get("content") or t.get("value") or t.get("title")

    # 学科类别如果需要也可从 static_data 里另行解析（此处略）
    return {"UT": uid, "title": title, "journal": journal, "year": year, "categories": cats}


@tool
def wos_expanded_citation_fanout(input: CitationFanoutInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """基于 Expanded API 的施引文献多跳展开（简易引用“路径/网络”）

    Args:
        input (CitationFanoutInput): 输入的入参，包括ut、database_id、max_depth、per_request、max_per_level、option_view、view_field、lang。其中ut是起始论文的UT，database_id是数据库，max_depth是向外展开的层数，per_request是每次请求条数，max_per_level是每一层最多抓取多少条，option_view是视图，view_field是尽量包含UID/标题/期刊/年份/学科等摘要用字段，lang是界面语言/记录语言偏好。
        api_key (Optional[str], optional): _description_. Defaults to None.
        base_url (str, optional): _description_. Defaults to EXPANDED_BASE.

    Returns:
        Dict[str, Any]: _description_
    """
    seed_ut = input.ut if input.ut.startswith("WOS:") else f"WOS:{input.ut}"
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[tuple] = []
    levels: Dict[int, List[str]] = defaultdict(list)

    frontier = deque([(seed_ut, 0)])
    seen = {seed_ut}

    while frontier:
        parent_ut, depth = frontier.popleft()
        if depth >= input.max_depth:
            continue

        batch_fetched = 0
        first = 1
        while batch_fetched < input.max_per_level:
            chunk = wos_expanded_citing_articles(
                ExpandedCitingArticlesInput(
                    ut=parent_ut,
                    database_id=input.database_id,
                    count=min(input.per_request, input.max_per_level - batch_fetched),
                    first_record=first,
                    view_field=input.view_field,
                    option_view=input.option_view,
                    lang=input.lang,
                    created_time_span=input.created_time_span,
                    by='uniqueId',
                ),
                api_key=api_key,
                base_url=base_url,
            )

            recs = (chunk.get("Data") or chunk.get("data") or {}).get("Records") \
                or chunk.get("records") or chunk

            items = []

            if isinstance(recs, list):
                # 罕见：Records 直接就是列表
                items = recs

            elif isinstance(recs, dict):
                # 常见 1：{"REC": [ ... ]}
                if isinstance(recs.get("REC"), list):
                    items = recs["REC"]

                # 常见 2：{"records": {"REC": [ ... ]}}
                elif isinstance(recs.get("records"), dict) and isinstance(recs["records"].get("REC"), list):
                    items = recs["records"]["REC"]

                # 常见 3：{"records": [ ... ]}
                elif isinstance(recs.get("records"), list):
                    items = recs["records"]

            # 若还是拿不到，就判空退出
            if not items:
                break

            for r in items:
                meta = _extract_meta_from_record(r)
                child_ut = meta.get("UT")
                if not child_ut:
                    continue
                nodes.setdefault(child_ut, meta)
                edges.append((parent_ut, child_ut))
                if child_ut not in seen and depth + 1 <= input.max_depth:
                    seen.add(child_ut)
                    frontier.append((child_ut, depth + 1))
                    levels[depth + 1].append(child_ut)

            got = len(items)
            batch_fetched += got
            if got < min(input.per_request, input.max_per_level - (batch_fetched - got)):
                break
            first += got


        if parent_ut not in nodes:
            try:
                parent_full = wos_expanded_get_by_ut(ExpandedGetByUTInput(
                    ut=parent_ut,
                    database_id=input.database_id,
                    view_field=input.view_field,
                    option_view=input.option_view,
                ), api_key=api_key, base_url=base_url)
                recs = (parent_full.get("Data") or parent_full.get("data") or {}).get("Records") or parent_full.get("records") or parent_full
                if isinstance(recs, dict):
                    cand = (recs.get("records") or [None])[0]
                elif isinstance(recs, list):
                    cand = recs[0] if recs else None
                else:
                    cand = None
                if isinstance(cand, dict):
                    nodes[parent_ut] = _extract_meta_from_record(cand)
            except Exception:
                nodes.setdefault(parent_ut, {"UT": parent_ut})

    return {"seed": seed_ut, "nodes": nodes, "edges": edges, "levels": dict(levels)}


class CitationSummaryInput(BaseModel):
    graph: Dict[str, Any] = Field(..., description="wos_expanded_citation_fanout 的返回对象")


# todo： need to be modified
@tool
def wos_citation_influence_summary(input: CitationSummaryInput) -> Dict[str, Any]:
    """对引用网络做轻量级影响力摘要：期刊、年份、学科的分布。

    Args:
        input (CitationSummaryInput): 输入的图谱,可从wos_expanded_citation_fanout的返回对象中获取

    Returns:
        _type_: _description_
    """
    nodes = input.graph.get("nodes", {})
    seed = input.graph.get("seed")
    edges = input.graph.get("edges", [])

    direct_citations = len([1 for (p, _) in edges if p == seed])

    by_journal = defaultdict(int)
    by_year = defaultdict(int)
    by_category = defaultdict(int)

    for ut, meta in nodes.items():
        if ut == seed:
            continue
        j = (meta.get("journal") or "").strip() or "(unknown)"
        y = meta.get("year") or "(unknown)"
        cats = meta.get("categories") or []
        by_journal[j] += 1
        by_year[y] += 1
        if isinstance(cats, list):
            for c in cats:
                by_category[c] += 1

    top_journals = sorted(by_journal.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top_years = sorted(by_year.items(), key=lambda kv: (str(kv[0]), kv[1]), reverse=True)
    top_categories = sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)[:20]

    return {
        "direct_citations": direct_citations,
        "top_journals": top_journals,
        "year_distribution": top_years,
        "top_categories": top_categories,
    }


# =============================
# 命令行示例（可删除）
# =============================
if __name__ == "__main__":


    seed = wos_expanded_search.invoke({"input":ExpandedSearchInput(
        usr_query='TI=("Hierarchical Interdisciplinary Topic Detection Model for Research Proposal Classification") AND AU=(Du Yi OR "Yi Du")',  # 你已验证 OK
        database_id='WOS',
        count=1, first_record=1,
        view_field=['UID'], option_view='SR', lang='en'
    )})

    seed_ut = seed['Data']['Records']['records']['REC'][0]['UID']
    print(seed_ut)
    # 2) 展开“施引网络”1~2跳（内部已用 /citing + uniqueId）
    
    graph = wos_expanded_citation_fanout.invoke(
        {"input":CitationFanoutInput(
        ut=seed_ut,
        database_id='WOS',
        max_depth=2,                 # 1=直接施引；2=二跳（注意规模）
        per_request=100, max_per_level=500,
        option_view='SR',            # 先用 SR 省配额；要跨库总TC改为 'FS'
        view_field=['UID','titles','source','pub_info','categories'],
        lang='en',
        created_time_span='2024-01-01 2025-09-10'
    )})

    print(graph)
    print("\n\n")
    # 3) 影响力摘要（直接被引数、期刊/年份/学科分布）
    summary = wos_citation_influence_summary.invoke({"input":CitationSummaryInput(graph=graph)})
    print('direct_citations =', summary['direct_citations'])
    print('top_journals =', summary['top_journals'][:10])
    print('year_distribution =', summary['year_distribution'][:5])
    print('top_categories =', summary['top_categories'][:10])
