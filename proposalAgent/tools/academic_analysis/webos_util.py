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
- Expanded API 端点（示例）：https://wos-api.clarivate.com/api/wos
  - 根搜索：       GET /api/wos
  - 被引参考文献： GET /api/wos/citedReferences
  - 施引文献：     GET /api/wos/citingArticles
  - 相关文献：     GET /api/wos/related

参数命名提示：
- Starter：q（高级检索语法，如 DT=Article）、db（库，如 WOS）、limit、page、sortField（例如 LD+D）、detail 等。
- Expanded：usrQuery（高级检索语法，如 ts=cadmium 或 UT=(WOS:001...)）、databaseId、count、firstRecord、viewField（如 UID pub_info keywords）、optionView（FS=Full，SR=Short）。

下方同时提供 LangChain StructuredTool 包装（可选）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import requests
from pydantic import BaseModel, Field

# =============================
# 通用：HTTP 工具 & 错误
# =============================
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
        # 尝试附带服务端返回
        snippet = resp.text[:400]
        raise _HTTPError(f"HTTP {resp.status_code}: {snippet}")

    try:
        return resp.json()
    except Exception:
        # 有时 Expanded 也可返回 XML；此处仅返回原始文本（由上层自行处理）
        return {"raw": resp.text}


# =============================
# Starter API
# =============================
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


def wos_starter_documents(input: StarterDocumentsInput, *, api_key: Optional[str] = None, base_url: str = STARTER_BASE) -> Dict[str, Any]:
    """查询 Web of Science 文献（Starter API）。等价于 GET /documents。
    返回原始 JSON（已解析为 dict）。
    """
    headers = _build_headers(api_key)
    url = f"{base_url}/documents"
    params: Dict[str, Any] = {
        "q": input.q,
        "db": input.db,
        "limit": input.limit,
        "page": input.page,
    }
    if input.sort_field:
        params["sortField"] = input.sort_field
    if input.detail is not None:
        params["detail"] = str(bool(input.detail)).lower()
    if input.modified_time_span:
        params["modifiedTimeSpan"] = input.modified_time_span
    if input.tc_modified_time_span:
        params["tcModifiedTimeSpan"] = input.tc_modified_time_span

    return _request_json("GET", url, headers=headers, params=params)


class StarterGetByUIDInput(BaseModel):
    uid: str = Field(..., description="WOS 记录的 UID/UT，如 WOS:001321517500001")
    db: str = Field("WOS", description="数据库代码，示例：WOS")


def wos_starter_get_by_uid(input: StarterGetByUIDInput, *, api_key: Optional[str] = None, base_url: str = STARTER_BASE) -> Dict[str, Any]:
    """通过 UID 获取单条记录（Starter API）。等价于 GET /documents/{uid}。
    返回原始 JSON（已解析为 dict）。
    """
    headers = _build_headers(api_key)
    url = f"{base_url}/documents/{input.uid}"
    params = {"db": input.db}
    return _request_json("GET", url, headers=headers, params=params)


# =============================
# Expanded API
# =============================
EXPANDED_BASE = "https://wos-api.clarivate.com/api/wos"  # 常见网关域名；如需改为 https://api.clarivate.com/api/wos 亦可

class ExpandedSearchInput(BaseModel):
    usr_query: str = Field(..., description="高级检索语法，如 ts=cadmium 或 UT=(WOS:001...)")
    database_id: str = Field("WOS", description="数据库，如 WOS")
    count: int = Field(10, ge=1, le=100, description="返回条数（每次返回记录数）")
    first_record: int = Field(1, ge=1, description="起始记录序号（从 1 开始）")
    view_field: Optional[List[str]] = Field(None, description="仅返回所需字段，例如 ['UID','pub_info','title']")
    option_view: Optional[str] = Field(None, description="FS=Full，SR=Short（Short Record）")
    lang: Optional[str] = Field(None, description="界面语言/记录语言偏好，例如 en")


def wos_expanded_search(input: ExpandedSearchInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """Expanded API 根搜索。等价于 GET /api/wos。
    注意：Expanded 支持 JSON 或 XML；本函数优先尝试 JSON，若非 JSON 将返回 {"raw": 原始文本}。
    """
    headers = _build_headers(api_key)
    url = base_url  # 根搜索
    params: Dict[str, Any] = {
        "databaseId": input.database_id,
        "usrQuery": input.usr_query,
        "count": input.count,
        "firstRecord": input.first_record,
    }
    if input.view_field:
        params["viewField"] = " ".join(input.view_field)
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


def wos_expanded_get_by_ut(input: ExpandedGetByUTInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """按 UT 精确获取记录，实质是构造 usrQuery=UT=(...) 执行根搜索。"""
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


def _expanded_child_endpoint(
    endpoint: str,
    input: ExpandedByUTBase,
    *,
    api_key: Optional[str],
    base_url: str,
) -> Dict[str, Any]:
    headers = _build_headers(api_key)
    url = f"{base_url}/{endpoint.strip('/')}"
    params: Dict[str, Any] = {
        "databaseId": input.database_id,
        "firstRecord": input.first_record,
        "count": input.count,
        # 官方示例常见把父记录 UT 写进 usrQuery=UT=(WOS:...)
        "usrQuery": f"UT=({input.ut})" if not input.ut.startswith("UT=") else input.ut,
    }
    if input.view_field:
        params["viewField"] = " ".join(input.view_field)
    if input.option_view:
        params["optionView"] = input.option_view

    return _request_json("GET", url, headers=headers, params=params)


class ExpandedCitedReferencesInput(ExpandedByUTBase):
    pass


def wos_expanded_cited_references(input: ExpandedCitedReferencesInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """获取指定 UT 的参考文献（被引参考文献列表）。等价于 GET /api/wos/citedReferences。"""
    return _expanded_child_endpoint("citedReferences", input, api_key=api_key, base_url=base_url)


class ExpandedCitingArticlesInput(ExpandedByUTBase):
    pass


def wos_expanded_citing_articles(input: ExpandedCitingArticlesInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """获取指定 UT 的施引文献。等价于 GET /api/wos/citingArticles。"""
    return _expanded_child_endpoint("citingArticles", input, api_key=api_key, base_url=base_url)


class ExpandedRelatedRecordsInput(ExpandedByUTBase):
    pass


def wos_expanded_related_records(input: ExpandedRelatedRecordsInput, *, api_key: Optional[str] = None, base_url: str = EXPANDED_BASE) -> Dict[str, Any]:
    """获取指定 UT 的相关文献。等价于 GET /api/wos/related。"""
    return _expanded_child_endpoint("related", input, api_key=api_key, base_url=base_url)


# =============================
# LangChain / LangGraph 工具包装（可选）
# =============================
try:
    from langchain_core.tools import StructuredTool

    def to_langchain_tools():
        return [
            StructuredTool.from_function(
                func=wos_starter_documents,
                name="wos_starter_documents",
                description="Starter API：检索文献（/documents）",
                args_schema=StarterDocumentsInput,
            ),
            StructuredTool.from_function(
                func=wos_starter_get_by_uid,
                name="wos_starter_get_by_uid",
                description="Starter API：按 UID 获取单条记录（/documents/{uid}）",
                args_schema=StarterGetByUIDInput,
            ),
            StructuredTool.from_function(
                func=wos_expanded_search,
                name="wos_expanded_search",
                description="Expanded API：根搜索（/api/wos）",
                args_schema=ExpandedSearchInput,
            ),
            StructuredTool.from_function(
                func=wos_expanded_get_by_ut,
                name="wos_expanded_get_by_ut",
                description="Expanded API：按 UT 精确获取（封装到根搜索 usrQuery=UT=(...)）",
                args_schema=ExpandedGetByUTInput,
            ),
            StructuredTool.from_function(
                func=wos_expanded_cited_references,
                name="wos_expanded_cited_references",
                description="Expanded API：被引参考文献（/api/wos/citedReferences）",
                args_schema=ExpandedCitedReferencesInput,
            ),
            StructuredTool.from_function(
                func=wos_expanded_citing_articles,
                name="wos_expanded_citing_articles",
                description="Expanded API：施引文献（/api/wos/citingArticles）",
                args_schema=ExpandedCitingArticlesInput,
            ),
            StructuredTool.from_function(
                func=wos_expanded_related_records,
                name="wos_expanded_related_records",
                description="Expanded API：相关文献（/api/wos/related）",
                args_schema=ExpandedRelatedRecordsInput,
            ),
        ]
except Exception:
    def to_langchain_tools():  # type: ignore
        raise RuntimeError("未安装 langchain-core；如需绑定 LangGraph 工具节点，请先安装。")


# =============================
# 命令行示例（可删除）
# =============================
if __name__ == "__main__":
    import json

    # —— Starter：/documents ——
    heads = {"X-ApiKey": os.getenv("CLARIVATE_API_KEY", "xxx")}
    print("== Starter documents demo ==")
    try:
        data = wos_starter_documents(StarterDocumentsInput(q="DT=Article", db="WOS", limit=3, page=1, sort_field="LD+D"))
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
    except Exception as e:
        print("Starter error:", e)

    print("\n== Expanded root search demo ==")
    try:
        data = wos_expanded_search(ExpandedSearchInput(
            usr_query="UT=(WOS:001321517500001)",
            database_id="WOS",
            count=1,
            first_record=1,
            view_field=["UID", "pub_info", "titles", "names", "citations"],
            option_view="FS",
        ))
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
    except Exception as e:
        print("Expanded error:", e)
