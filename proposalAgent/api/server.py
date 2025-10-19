from __future__ import annotations

import asyncio
import io
import json
import os
import logging
import textwrap
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from proposalAgent.graphs.proposal_graph import ProposalAgentGraph
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.models import (
    ChatMessage,
    ChatMessageRequest,
    ChatRequest,
    SessionControl,
)
from proposalAgent.agents.stage3.generator import (
    _format_completeness_result,
    _format_debate_results,
)
from proposalAgent.utils.logger import get_logger

try:
    import oss2
    from oss2.credentials import EnvironmentVariableCredentialsProvider
    import alibabacloud_oss_v2 as oss
    from alibabacloud_oss_v2.credentials import (
        EnvironmentVariableCredentialsProvider as V2EnvProvider,
    )
except ImportError:  
    oss2 = None
    EnvironmentVariableCredentialsProvider = None
    oss = None
    V2EnvProvider = None

try: 
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except ImportError: 
    A4 = None
    pdfmetrics = None
    UnicodeCIDFont = None
    canvas = None
    REPORTLAB_AVAILABLE = False
logger = get_logger("api")  


app = FastAPI(title="ProposalAgent Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_SESSIONS: Dict[str, List[ChatMessage]] = defaultdict(list)
SESSION_QUEUES: Dict[str, SessionQueues] = {}
UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
OSS_REGION = os.getenv("OSS_REGION", "cn-hangzhou")
OSS_BUCKET = os.getenv("OSS_BUCKET", "evaluatoin-pdfs")


class OssNotConfigured(RuntimeError):
    """Raised when OSS dependencies or credentials are missing."""

def _ensure_oss_available() -> None:
    if not all([oss2, EnvironmentVariableCredentialsProvider, oss, V2EnvProvider]):
        raise OssNotConfigured("OSS SDK 未安装，请安装 oss2 和 alibabacloud_oss_v2 后重试")

    required_envs = ["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET"]
    missing = [env for env in required_envs if not os.getenv(env)]
    if missing:
        raise OssNotConfigured(f"缺少 OSS 访问凭证环境变量: {', '.join(missing)}")


def upload_bytes_to_oss(object_key: str, content: bytes, content_type: Optional[str] = None) -> None:
    _ensure_oss_available()

    credentials_provider = EnvironmentVariableCredentialsProvider()
    bucket = oss2.Bucket(
        oss2.ProviderAuthV4(credentials_provider),
        OSS_ENDPOINT,
        OSS_BUCKET,
        region=OSS_REGION,
    )

    headers = {}
    if content_type:
        headers["Content-Type"] = content_type

    bucket.put_object(object_key, content, headers=headers)


def generate_presigned_get_url(object_key: str, expires: int = 900) -> str:
    _ensure_oss_available()

    cfg = oss.config.load_default()
    cfg.credentials_provider = V2EnvProvider()
    cfg.region = OSS_REGION
    cfg.endpoint = OSS_ENDPOINT

    client = oss.Client(cfg)
    pre_result = client.presign(oss.GetObjectRequest(bucket=OSS_BUCKET, key=object_key))
    return pre_result.url


def extract_judge_summaries(debate_results: Any) -> Dict[str, Dict[str, str]]:
    summaries: Dict[str, Dict[str, str]] = {}
    if not isinstance(debate_results, dict):
        return summaries

    for discipline, debate_data in debate_results.items():
        discipline_key = (
            "-".join(str(part) for part in discipline)
            if isinstance(discipline, (tuple, list))
            else str(discipline)
        )

        discipline_summary: Dict[str, str] = {}
        if isinstance(debate_data, dict):
            feas_summary = debate_data.get("可行性", {}).get("judge_summary")
            innov_summary = debate_data.get("创新性", {}).get("judge_summary")
            if feas_summary:
                discipline_summary["feasibility"] = str(feas_summary)
            if innov_summary:
                discipline_summary["innovation"] = str(innov_summary)

        if discipline_summary:
            summaries[discipline_key] = discipline_summary

    return summaries


def create_and_upload_reports(final_state: Optional[Dict[str, Any]], thread_id: str) -> Optional[Dict[str, str]]:
    if not final_state:
        return None

    try:
        _ensure_oss_available()
    except OssNotConfigured:
        logger.warning("OSS 未配置，跳过文件上传")
        return None

    if not REPORTLAB_AVAILABLE:
        logger.warning("reportlab 未安装，无法生成 PDF，跳过文件上传")
        return None

    if REPORTLAB_AVAILABLE:
        try:
            pdfmetrics.getFont("STSong-Light")
        except KeyError:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    flat_state: Dict[str, Any] = (
        flatten_state(final_state) if isinstance(final_state, dict) else {}
    )

    def _normalize(value: Any) -> str:
        if value is None:
            return "无"
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                return str(value)
        text = str(value).strip()
        return text or "无"

    academic_report = _normalize(
        flat_state.get("academic_analysis_report")
        or final_state.get("academic_analysis_report")
        or "未提供学术分析报告"
    )
    future_report = _normalize(
        flat_state.get("future_influence_report")
        or final_state.get("future_influence_report")
        or "未提供未来影响力分析"
    )
    social_report = _normalize(
        flat_state.get("social_analysis_report")
        or final_state.get("social_analysis_report")
        or "未提供社会影响分析"
    )
    final_analysis_summary = _normalize(
        flat_state.get("final_analysis_summary")
        or final_state.get("final_analysis_summary")
        or "未提供最终分析摘要"
    )
    final_report_text = _normalize(
        flat_state.get("final_report")
        or final_state.get("final_report")
        or "未提供综合评审报告"
    )
    completeness_info = _normalize(
        flat_state.get("completeness_check_result")
        or final_state.get("completeness_check_result")
        or "未提供完备性检查结果"
    )
    human_feedback = _normalize(
        flat_state.get("human_feedback")
        or final_state.get("human_feedback")
        or "无"
    )

    missing_markers = {
        "未提供学术分析报告",
        "未提供未来影响力分析",
        "未提供社会影响分析",
        "未提供最终分析摘要",
        "未提供综合评审报告",
        "未提供完备性检查结果",
        "未提供辩论裁判总结",
        "无",
        "{}",
    }

    handled_report_keys = {
        "academic_analysis_report",
        "future_influence_report",
        "social_analysis_report",
        "final_analysis_summary",
        "final_report",
    }

    extra_reports: List[Tuple[str, str]] = []
    for key, value in flat_state.items():
        if not value:
            continue
        if key in handled_report_keys:
            continue
        if key.endswith("_report"):
            extra_reports.append((key, _normalize(value)))

    debate_results = flat_state.get("debate_results", {}) or final_state.get("debate_results", {})
    judge_summaries = extract_judge_summaries(debate_results)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    PAGE_MARGIN = 40
    LINE_HEIGHT = 20
    TITLE_FONT = "STSong-Light"
    BODY_FONT = "STSong-Light"
    FONT_SIZE = 16
    usable_width = width - PAGE_MARGIN * 2

    pdf.setTitle("ProposalAgent Analysis Report")

    y_position = height - PAGE_MARGIN
    page_number = 1

    def start_new_page() -> None:
        nonlocal y_position, page_number
        pdf.showPage()
        page_number += 1
        pdf.setFont(TITLE_FONT, FONT_SIZE)
        pdf.drawString(PAGE_MARGIN, height - PAGE_MARGIN, "ProposalAgent 评估输出摘要（续）")
        y_position = height - PAGE_MARGIN - LINE_HEIGHT * 2

    def ensure_space(line_count: int = 1) -> None:
        nonlocal y_position
        if y_position - line_count * LINE_HEIGHT < PAGE_MARGIN:
            start_new_page()

    def render_lines(text: str) -> List[str]:
        raw_lines = text.splitlines() if text else [""]
        wrapped: List[str] = []

        for raw_line in raw_lines:
            stripped = raw_line.rstrip()
            if stripped == "":
                wrapped.append("")
                continue

            current = ""
            for char in stripped:
                candidate = current + char
                if pdf.stringWidth(candidate, BODY_FONT, FONT_SIZE) <= usable_width:
                    current = candidate
                else:
                    if current:
                        wrapped.append(current)
                    # 如果单字符超过宽度，直接单独成行
                    if pdf.stringWidth(char, BODY_FONT, FONT_SIZE) > usable_width:
                        wrapped.append(char)
                        current = ""
                    else:
                        current = char

            if current:
                wrapped.append(current)

        return wrapped or [""]

    def write_paragraph(title: str, content: str) -> None:
        nonlocal y_position
        if not content:
            return
        normalized = content.strip()
        if not normalized:
            return
        if normalized in missing_markers:
            return
        wrapped_lines = render_lines(normalized)
        ensure_space(len(wrapped_lines) + 2)

        pdf.setFont(TITLE_FONT, FONT_SIZE)
        pdf.drawString(PAGE_MARGIN, y_position, title)
        y_position -= LINE_HEIGHT

        pdf.setFont(BODY_FONT, FONT_SIZE)
        for line in wrapped_lines:
            if y_position - LINE_HEIGHT < PAGE_MARGIN:
                start_new_page()
                pdf.setFont(BODY_FONT, FONT_SIZE)
            if line:
                pdf.drawString(PAGE_MARGIN, y_position, line)
            y_position -= LINE_HEIGHT

        y_position -= LINE_HEIGHT // 2

    pdf.setFont(TITLE_FONT, FONT_SIZE)
    pdf.drawString(PAGE_MARGIN, y_position, "ProposalAgent 评估输出摘要")
    y_position -= LINE_HEIGHT * 2

    write_paragraph("线程 ID", str(thread_id))
    write_paragraph("最终分析摘要", final_analysis_summary)
    write_paragraph("学术分析报告", academic_report)
    write_paragraph("社会影响分析报告", social_report)
    write_paragraph("未来影响力分析报告", future_report)
    write_paragraph("综合评审报告", final_report_text)

    report_title_map = {
        "social_analysis_report": "社会影响分析报告",
        "future_influence_report": "未来影响力分析报告",
        "academic_analysis_report": "学术分析报告",
        "final_report": "综合评审报告",
    }

    for key, value in extra_reports:
        title = report_title_map.get(key, key)
        write_paragraph(title, value)

    write_paragraph("完备性检查结果", completeness_info)
    if human_feedback not in missing_markers:
        write_paragraph("人类反馈", human_feedback)

    judge_lines: List[str] = []
    if judge_summaries:
        for discipline, summary in judge_summaries.items():
            judge_lines.append(f"学科：{discipline}")
            feasibility = summary.get("feasibility") or "无可行性结论"
            innovation = summary.get("innovation") or "无创新性结论"
            judge_lines.append(f"  可行性裁判结论：{feasibility}")
            judge_lines.append(f"  创新性裁判结论：{innovation}")
            judge_lines.append("")
    else:
        judge_lines.append("未提供辩论裁判总结")

    write_paragraph("辩论裁判最终结论", "\n".join(judge_lines))

    debate_details = _normalize(debate_results)
    write_paragraph("辩论详情", debate_details)

    pdf.setFont(BODY_FONT, FONT_SIZE)
    pdf.drawRightString(width - PAGE_MARGIN, PAGE_MARGIN / 2, "Powered by ProposalAgent")

    pdf.save()

    buffer.seek(0)
    pdf_bytes = buffer.read()

    object_key = f"reports/{thread_id}/{uuid.uuid4().hex}.pdf"
    upload_bytes_to_oss(object_key, pdf_bytes, content_type="application/pdf")
    url = generate_presigned_get_url(object_key)

    return {"report": url, "object_key": object_key}

class SessionQueues:
    def __init__(self, allow_multi: bool = False) -> None:
        self.event_queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.control_queue: "asyncio.Queue[SessionControl]" = asyncio.Queue()
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.final_state: Optional[Dict[str, Any]] = None
        self.report_urls: Optional[Dict[str, str]] = None
        self.final_states: Dict[str, Dict[str, Any]] = {}
        self.report_urls_map: Dict[str, Dict[str, str]] = {}
        self.expected_files: List[str] = []
        self.allow_multi = allow_multi
        self.sub_control_queues: Dict[str, "asyncio.Queue[SessionControl]"] = {}
        self.sub_cancel_events: Dict[str, asyncio.Event] = {}
        self.sub_session_ids: Dict[str, str] = {}

async def get_graph() -> ProposalAgentGraph:
    return ProposalAgentGraph(config=TONGYI_CONFIG)

def format_sse(event: str, data: Any) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def build_interrupt_prompt(
    final_state: Optional[Dict[str, Any]],
    interrupt_payload: Optional[Any] = None,
) -> str:
    analysis_summary = final_state.get("final_analysis_summary", "") if final_state else ""
    completeness = final_state.get("completeness_check_result", {}) if final_state else {}

    task_desc = ""
    extra_instructions = ""
    if isinstance(interrupt_payload, dict):
        task_desc = interrupt_payload.get("task", "")
        extra_instructions = interrupt_payload.get("instructions", "")
    elif interrupt_payload is not None:
        task_desc = str(interrupt_payload)

    context_lines: List[str] = [
        "请审查项目评估分析并提供反馈意见",
        f"分析摘要：{analysis_summary}",
        f"完备性问题：{json.dumps(completeness, ensure_ascii=False)}",
        "",
        "请提供您的反馈意见。如果分析满足要求，请输入'approved'。如果需要改进，请详细说明需要改进的方面。",
    ]

    if task_desc:
        context_lines.insert(0, f"任务说明：{task_desc}")
    if extra_instructions and extra_instructions not in context_lines:
        context_lines.append(f"补充指引：{extra_instructions}")

    return "\n".join(context_lines)


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, BaseMessage):
        return {
            "type": obj.__class__.__name__,
            "content": make_serializable(getattr(obj, "content", "")),
            "additional_kwargs": make_serializable(getattr(obj, "additional_kwargs", {})),
        }

    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(item) for item in obj]

    return str(obj)


def _flatten_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return make_serializable(value)
    if isinstance(value, (list, tuple, set)):
        return [_flatten_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _flatten_value(v) for k, v in value.items()}
    return value


def flatten_state(state: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    stack = [state]
    visited: set[int] = set()

    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue

        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        for key, value in current.items():
            if isinstance(value, dict):
                stack.append(value)
            else:
                flat[key] = _flatten_value(value)

    return flat

@app.post("/upload")
async def upload_file(
    files: List[UploadFile] = File(...),
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for file in files:
        suffix = Path(file.filename or "uploaded").suffix
        dest_path = UPLOAD_ROOT / f"{uuid.uuid4().hex}{suffix}"
        await file.seek(0)
        total_written = 0
        with dest_path.open("wb") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                total_written += len(chunk)
        await file.close()

        file_id = str(uuid.uuid4())
        results.append(
            {
                "file_id": file_id,
                "filename": file.filename,
                "path": str(dest_path),
                "size": total_written,
            }
        )

    return results


async def run_evaluation(
    graph: ProposalAgentGraph,
    request: ChatRequest,
    session_thread_id: str,
    eval_thread_config: Dict[str, Any],
    queues: SessionQueues,
    file_id: Optional[str] = None,
    eval_thread_id: Optional[str] = None,
) -> None:
    final_state: Optional[Dict[str, Any]] = None
    log_thread_id = (
        eval_thread_id
        or eval_thread_config.get("configurable", {}).get("thread_id")
        or session_thread_id
    )
    logger.info(
        "run_evaluation: session_thread_id=%s, eval_thread_id=%s",
        session_thread_id,
        log_thread_id,
    )

    sub_control_queue: Optional["asyncio.Queue[SessionControl]"] = None
    sub_cancel_event: Optional[asyncio.Event] = None
    if file_id and queues.allow_multi:
        sub_control_queue = queues.sub_control_queues.get(file_id)
        sub_cancel_event = queues.sub_cancel_events.get(file_id)

    async def feedback_handler(interrupt_payload: Any) -> Optional[str]:
        payload_value = getattr(interrupt_payload, "value", interrupt_payload)
        prompt = build_interrupt_prompt(final_state, payload_value)
        interrupt_event: Dict[str, Any] = {
            "thread_id": session_thread_id,
            "prompt": prompt,
        }

        if payload_value is not None:
            interrupt_event["payload"] = payload_value
        if file_id:
            interrupt_event["file_id"] = file_id

        await queues.event_queue.put(("interrupt", interrupt_event))
        logger.info("接收到中断事件: %s", interrupt_event)
        while True:
            queue_to_use: "asyncio.Queue[SessionControl]"
            if sub_control_queue is not None:
                queue_to_use = sub_control_queue
            else:
                queue_to_use = queues.control_queue

            control = await queue_to_use.get()

            if control.file_id and file_id and control.file_id != file_id:
                await queue_to_use.put(control)
                await asyncio.sleep(0)
                continue
            if control.action == "cancel":
                if sub_cancel_event is not None:
                    sub_cancel_event.set()
                else:
                    queues.cancel_event.set()
                return None

            if control.action == "resume":
                if sub_cancel_event is not None:
                    sub_cancel_event.clear()
                else:
                    queues.cancel_event.clear()
                return control.feedback or ""
    file_id_to_use = file_id or (request.file_ids[0] if request.file_ids else "")

    initial_state = graph.propagator.create_initial_state(
        user_prompt=request.query,
        user_interest=request.user_interest,
        filepath=file_id_to_use,
    )

    try:
        stream_thread_id = session_thread_id
        stream_closed_early = False
        try:
            async for chunk in graph.stream_project(
                initial_state,
                eval_thread_config,
                feedback_handler=feedback_handler,
                cancel_event=sub_cancel_event or queues.cancel_event,
            ):
                if queues.cancel_event.is_set():
                    await queues.event_queue.put(("cancelled", {"thread_id": stream_thread_id}))
                    return

                if "__interrupt__" in chunk:
                    final_state = chunk.get("state")
                    continue

                final_state = chunk.get("state")
        except GeneratorExit:
            stream_closed_early = True
            logger.info(
                "stream_project terminated early for session %s (file_id=%s)",
                stream_thread_id,
                file_id,
            )

        if stream_closed_early:
            await queues.event_queue.put(("cancelled", {"thread_id": stream_thread_id}))
            return

        final_state = final_state or graph.curr_state or {}

        serializable_state = make_serializable(final_state)

        with open("final_state.json", "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, ensure_ascii=False)
            print(f"最终状态已保存到 final_state.json，长度: {len(serializable_state)}")

        SESSION_QUEUES[session_thread_id].final_state = final_state

        try:
            report_urls = await asyncio.to_thread(
                create_and_upload_reports, final_state, log_thread_id
            )
        except Exception as exc_upload:  # noqa: BLE001
            logger.exception("生成或上传报告文件失败: %s", exc_upload)
            report_urls = None

        if file_id:
            serializable_state_with_id = {**serializable_state, "__file_id__": file_id}
            queues.final_states[file_id] = serializable_state_with_id
            queues.final_state = serializable_state
            if report_urls:
                queues.report_urls_map[file_id] = report_urls
        if queues.expected_files:
            if all(fid in queues.final_states for fid in queues.expected_files):
                merged_payload = {
                    fid: queues.final_states.get(fid, {}) for fid in queues.expected_files
                }
                await queues.event_queue.put(
                    (
                        "stage_complete",
                        {
                            "thread_id": stream_thread_id,
                            "states": merged_payload,
                        },
                    )
                )
                if queues.report_urls_map:
                    await queues.event_queue.put(
                        (
                            "report_ready",
                            {
                                "thread_id": stream_thread_id,
                                "downloads": queues.report_urls_map,
                            },
                        )
                    )
                await queues.event_queue.put(("done", {"thread_id": stream_thread_id}))
        else:
            queues.final_state = final_state
            queues.report_urls = report_urls
            await queues.event_queue.put(
                (
                    "stage_complete",
                    {"thread_id": stream_thread_id, "state": serializable_state},
                )
            )
            if report_urls:
                await queues.event_queue.put(
                    (
                        "report_ready",
                        {
                            "thread_id": stream_thread_id,
                            "downloads": report_urls,
                        },
                    )
                )
            await queues.event_queue.put(("done", {"thread_id": stream_thread_id}))
    except Exception as exc:  # noqa: BLE001
        await queues.event_queue.put(
            ("error", {"thread_id": stream_thread_id, "message": str(exc)})
        )



@app.post("/chat/normal/sse")
async def chat_progress_sse(
    request: ChatMessageRequest,
) -> StreamingResponse:
    thread_id = request.thread_id or f"chat_{uuid.uuid4().hex}"
    history = CHAT_SESSIONS[thread_id]

    if not history or history[0].role != "system":
        history.insert(0, ChatMessage(role="system", content="你是由张子豪开发的成果评价助手"))

    if request.message:
        history.append(ChatMessage(role="user", content=request.message))

    conversation = [
        HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content)
        for msg in history
    ]

    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-9ce983386aa74c8f8131eb8ecbf90f58",
        streaming=True
    )
        
    async def event_generator() -> AsyncGenerator[bytes, None]:
        reply_chunks: List[str] = []
        yield format_sse("session", {"thread_id": thread_id})
        try:
            async for chunk in llm.astream(conversation):
                if isinstance(chunk, AIMessageChunk):
                    delta = chunk.content
                elif isinstance(chunk, AIMessage):
                    delta = chunk.content
                else:
                    delta = str(chunk)

                if delta:
                    reply_chunks.append(str(delta))
                    yield format_sse("delta", {"thread_id": thread_id, "delta": str(delta)})

            reply_text = "".join(reply_chunks)
            if reply_text:
                history.append(ChatMessage(role="assistant", content=reply_text))
                if request.history_limit and request.history_limit > 0:
                    history.insert(0, ChatMessage(role="system", content="你是由张子豪开发的成果评价助手"))
                    CHAT_SESSIONS[thread_id] = history[-request.history_limit :]
                else:
                    CHAT_SESSIONS[thread_id] = history

                yield format_sse(
                    "complete",
                    {
                        "thread_id": thread_id,
                        "reply": reply_text,
                    #    "history": [msg.dict() for msg in CHAT_SESSIONS[thread_id]],
                    },
                )
            yield format_sse("done", {"thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            yield format_sse("error", {"thread_id": thread_id, "message": str(exc)})
            yield format_sse("done", {"thread_id": thread_id})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/evaluate")
async def proposal_evaluation(
    request: ChatRequest,
    graph: ProposalAgentGraph = Depends(get_graph),
) -> StreamingResponse:

    thread_id, thread_config = graph.create_session(
        request.thread_id,
        recursion_limit=150,
    )
    queues = SessionQueues(allow_multi=len(request.file_ids) > 1)
    SESSION_QUEUES[thread_id] = queues

    # 将评估请求添加到对话历史
    history = CHAT_SESSIONS[thread_id]
    if not history or history[0].role != "system":
        history.insert(0, ChatMessage(role="system", content="你是由张子豪开发的成果评价助手"))
    
    # 添加用户的评估请求到历史
    user_message = f"请评估项目：{request.query}"
    if request.user_interest:
        user_message += f"\n关注点：{', '.join(request.user_interest)}"
    history.append(ChatMessage(role="user", content=user_message))

    multiple_files = len(request.file_ids) > 1
    if multiple_files:
        queues.expected_files = list(request.file_ids)

    await queues.event_queue.put(("session", {"thread_id": thread_id}))

    # 运行coro任务，向queue中添加chunk
    if multiple_files:
        for fid in request.file_ids:
            per_file_request = ChatRequest(
                file_ids=[fid],
                query=request.query,
                user_interest=request.user_interest,
                thread_id=request.thread_id,
            )
            recursion_limit = thread_config.get("recursion_limit") or thread_config.get(
                "config", {}
            ).get("recursion_limit")
            per_thread_id, per_thread_config = graph.create_session(
                recursion_limit=recursion_limit,
            )

            sub_control_queue: Optional["asyncio.Queue[SessionControl]"] = None
            sub_cancel_event: Optional[asyncio.Event] = None
            if queues.allow_multi:
                sub_control_queue = asyncio.Queue()
                sub_cancel_event = asyncio.Event()
                queues.sub_control_queues[fid] = sub_control_queue
                queues.sub_cancel_events[fid] = sub_cancel_event
                queues.sub_session_ids[fid] = per_thread_id

            asyncio.create_task(
                run_evaluation(
                    graph,
                    per_file_request,
                    thread_id,
                    per_thread_config,
                    queues,
                    file_id=fid,
                    eval_thread_id=per_thread_id,
                )
            )
    else:
        asyncio.create_task(
            run_evaluation(
                graph,
                request,
                thread_id,
                thread_config,
                queues,
            )
        )

    async def event_generator() -> AsyncGenerator[bytes, None]:
        try:
            while True:
                event, payload = await queues.event_queue.get()
                if event == "stage_complete":
                    if payload.get("states"):
                        raw_states = {}
                        flattened_states = {}
                        for fid, per_state in payload["states"].items():
                            raw_states[fid] = per_state
                            flattened_states[fid] = flatten_state(per_state)

                        # todo 可以引入外部记忆
                        # todo 权重引入配置化
                        # 添加额外的控制模版
                        # 来源锚点 vs 原文锚点
                        # 添加记忆模块，来优化各个agent的回复行为，根据最后生成报告的内容来惩罚对各个agent的输出 **** 重要
                        formatted_lines: List[str] = []
                        for fid, state_flat in flattened_states.items():
                            formatted_lines.append(f"文件ID: {fid}")
                            formatted_lines.append(
                                f"  ### 学术分析：{state_flat.get('academic_analysis_report', '未生成') }"
                            )
                            formatted_lines.append(
                                f"  ### 未来影响分析：{state_flat.get('future_influence_report', '未生成')}"
                            )
                            formatted_lines.append(
                                f"  ### 辩论结果：{_format_debate_results(state_flat.get('debate_results', {}))}"
                            )
                            formatted_lines.append(
                                f"  ### 原始文章结构化信息(原始文本+正文摘要信息): {str(state_flat.get('research_structure', '无'))}"
                            )
                            
                            formatted_lines.append("")

                        formatted_states = "\n".join(formatted_lines)

                        prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    """
                                        你是国家自然科学基金委员会项目评审专家。请仅依据【本轮提供的材料】进行横向对比评审并形成最终意见。

                                        【硬性规则——务必全部满足】
                                        1) 来源锚定：每一条关键判断与结论，句末必须标注来源，格式统一为：〔来源：文件id：第X页/图Y/表Z/式(K)/URL/DOI〕；"。
                                        2) 信息边界：禁止使用、臆测或补充任何外部信息（含常识、既往经验、网络资料）。若输入材料本身包含外部资料，请以"〔外部资料：……，不计入评分依据〕"单独标注，并与正式结论分段隔离。
                                        3) 可执行性：每条"问题/建议"均需落地为可执行条目（含：目标/动作/指标/验收/负责人或资源来源/时间）。
                                        4) 科学性：必须给出对照实验设计与验证闭环（含：基线、数据切分、统计检验、外部验证与复现要素）。缺项时明确写明并给出"最低可行补充清单（MVP）"。
                                        5) 风险透明：对关键结论标注"来源强度(高/中/低)"与"风险等级(高/中/低)"，二者分别独立判断。
                                        6) NSFC口径：语气客观克制、就事论事；不使用宣传化、市场化措辞；篇章结构符合NSFC常见评审格式。

                                        【写作与版式要求】
                                        - 全文中文；结构化小标题；重要信息用 Markdown 表格呈现。
                                        - 所有页码/图表编号必须出自本申请材料；不得输出网址或外部参考链接。
                                        - 若材料存在缺项，请明确指出并给出MVP补充清单，但不得捏造信息。
                                        """,
                                ),
                                (
                                    "human",
                                    """
                                        请基于以下多份申请材料的分析结果，生成"国家自然科学基金项目横向对比评审意见"。内容要求翔实、可追溯、可执行，并严格遵守 system 中的硬性规则。
                                        文件数目：{file_count}
                                        【可用材料】
                                        {formatted_states}

                                        【输出结构与要求】
                                    
                                        一、项目基本信息对比（请原样列出）
                                        - 申请人
                                        - 依托单位
                                        - 申请代码
                                        - 项目题目
                                        - 项目类型（青年/面上/重大项目/重点支持项目）
                                        注意：
                                                青年：以个人成长为主 → 为后续申报面上打基础。
                                                面上：以稳定方向的连续探索为主 → 成果积累到一定程度后，可凝练为重点。
                                                重点：在学科内具有关键意义的问题的加强版攻关 → 若问题上升到国家战略或重大前沿交叉层面，且需系统组织与多课题协同，则进一步形成重大项目。
                                                专项：不直接对应科学问题攻关，而是支撑 NSFC 与学科生态（交流、战略研究、科普、平台），与上述科研项目不在一条赛道

                                        A. 项目概述对比
                                        - 准确概括各项目研究主题、技术主线与验证场景；给出与申请材料页/图的锚定。句句有据。〔来源：页/图〕

                                        A.1. 来源锚定清单对比（要点式）
                                        - 按"判断 → 来源来源（页/图/表/式） → 来源强度(高/中/低)"逐条列示，覆盖：研究基础、代表性成果、数据与算力条件、预研结果、应用场景。

                                        B. 学术能力对比
                                        - 列举学术能力，包括h index等内容
                                        - 列举发表论文，包括论文数量、论文被引次数、论文影响因子等内容
                                        
                                        C. 科学问题与创新性对比
                                        - 逐条结构：问题 → 申请书来源 → 本评审判断（与国内外的差异/预期增量） → 来源强度。
                                        - 明确核心创新点不超过3条，每条都需有"式/图/流程"的来源锚定。

                                        D. 技术路线与可行性对比,生成多个表格，一个表格对应一个文件
                                        - 按模块/数据/人力/合规展开；每条均需来源锚定，并给出"可行性结论 + 风险等级"。

                                        E. 对照实验与验证闭环对比,生成多个表格，一个表格对应一个文件
                                        - 基线集
                                        - 数据切分策略（
                                        - 指标
                                        - 统计检验
                                        - 外部验证
                                        - 复现要素

                                        F. 风险与对策矩阵对比,生成多个表格，一个表格对应一个文件
                                        - 列：风险 | 触发信号 | 缓解动作 | 备用方案 | 负责人/资源来源 | 时间
                                        - 每行必须可执行（目标/动作/指标/验收/时间均完整），并标注来源强度与风险等级。

                                        G. 里程碑与KPI对比,生成多个表格，一个表格对应一个文件
                                        - 列：阶段 | 交付物 | 量化指标 | 验收口径 | 阻断条件
                                        - 每条需页/图/表/式来源；若未见，写"〔来源：未见于材料〕"，并给出最低可行门槛。

                                        H. 横向对比与取舍,生成多个表格，一个表格对应一个文件
                                        - 关键差异矩阵（至少3个维度的正反对照）
                                        - 取舍性判断（给出清晰理由）
                                        - 主要问题与建议（3–6条，逐条"可执行"）
                                        - 逐条格式：问题 → 目标 → 动作 → 指标 → 验收 → 负责人/资源 → 时间；每条标注风险等级与来源强度。

                                        I. 评分与结论，生成多个，一个表格对应一个文件
                                        - 维度（0.0–5.0）：创新性、科学意义、可行性、研究基础、团队与条件、经费匹配、风险控制。
                                        - 给出加总/平均说明，并产出综合等级（优/良/中/差）。
                                        - 资助建议：可资助 / 可资助并附条件 / 不建议资助。
                                        - 若"附条件"，列2–3条"可量化里程碑"（阈值+时间点）。
                                        - 每个分值后附一句依据+来源锚定；若依据不足，明确"〔来源：未见于材料〕"。

                                        【额外要求】
                                        - 若发现材料缺项导致无法判断，直接写明"因来源不足无法下结论"，并提供"最低可行补充清单（MVP）"，但不降低来源标准。
                                        """
                                        ,
                                ),
                                
                            ]
                        )

                        input_data = {
                            "file_count": len(raw_states),
                            "formatted_states": formatted_states,
                        }

                        chain = prompt | graph.deep_thinking_llm
                        chunks = []
                        async for chunk in chain.astream(input_data):
                            delta = getattr(chunk, "content", str(chunk))
                            if not delta:
                                continue
                            chunks.append(delta)
                            yield format_sse("report_delta", {"thread_id": thread_id, "delta": delta})

                        final_report = "".join(chunks)
                        history.append(ChatMessage(role="assistant", content=final_report))
                        CHAT_SESSIONS[thread_id] = history
                        await queues.event_queue.put(
                            (
                                "multi_report_complete",
                                {
                                    "thread_id": thread_id,
                                    "reports": raw_states,
                                    "final_report": final_report,
                                },
                            )
                        )
                        yield format_sse(
                            "report_complete",
                            {
                                "thread_id": thread_id,
                                "report": final_report,
                            },
                        )
                        continue

                    raw_state_nested = getattr(queues, "final_state", {})
                    raw_state = (
                        flatten_state(raw_state_nested)
                        if isinstance(raw_state_nested, dict)
                        else {}
                    )

                    prompt = ChatPromptTemplate.from_messages(
                                [
                                    (
                                        "system",
                                        """
                                            你是国家自然科学基金委员会项目评审专家。请仅依据【本轮提供的材料】进行评审并形成最终意见。

                                            【硬性规则——务必全部满足】
                                            1) 来源锚定：每一条关键判断与结论，句末必须标注来源，格式统一为：〔来源：第X页/图Y/表Z/式(K)/URL/DOI〕；”。
                                            2) 信息边界：禁止使用、臆测或补充任何外部信息（含常识、既往经验、网络资料）。若输入材料本身包含外部资料，请以“〔外部资料：……，不计入评分依据〕”单独标注，并与正式结论分段隔离。
                                            3) 可执行性：每条“问题/建议”均需落地为可执行条目（含：目标/动作/指标/验收/负责人或资源来源/时间）。
                                            4) 科学性：必须给出对照实验设计与验证闭环（含：基线、数据切分、统计检验、外部验证与复现要素）。缺项时明确写明并给出“最低可行补充清单（MVP）”。
                                            5) 风险透明：对关键结论标注“来源强度(高/中/低)”与“风险等级(高/中/低)”，二者分别独立判断。
                                            6) NSFC口径：语气客观克制、就事论事；不使用宣传化、市场化措辞；篇章结构符合NSFC常见评审格式。

                                            【写作与版式要求】
                                            - 全文中文；结构化小标题；重要信息用 Markdown 表格呈现。
                                            - 所有页码/图表编号必须出自本申请材料；不得输出网址或外部参考链接。
                                            - 若材料存在缺项，请明确指出并给出MVP补充清单，但不得捏造信息。
                                            """
                                    ),
                                    (
                                        "human",
                                        """
                            请基于以下全部输入，生成“国家自然科学基金项目评审意见（青年/面上/重大项目/）”。内容要求翔实、可追溯、可执行，并严格遵守 system 中的硬性规则。

                            【可用材料】
                            - 学术分析：{academic_analysis_report}
                            - 未来影响分析：{future_influence_report}
                            - 辩论结果（多智能体交叉评议）：{debate_results}
                            - 原始pdf结构化信息: {research_structure}

                           

                            【输出结构与要求】
                            
                            项目基本信息（请原样列出）
                            - 申请人
                            - 依托单位
                            - 申请代码
                            - 项目题目
                            - 项目类型（青年/面上/重大项目/重点支持项目）
                            注意：
                                    青年：以个人成长为主 → 为后续申报面上打基础。
                                    面上：以稳定方向的连续探索为主 → 成果积累到一定程度后，可凝练为重点。
                                    重点：在学科内具有关键意义的问题的加强版攻关 → 若问题上升到国家战略或重大前沿交叉层面，且需系统组织与多课题协同，则进一步形成重大项目。
                                    专项：不直接对应科学问题攻关，而是支撑 NSFC 与学科生态（交流、战略研究、科普、平台），与上述科研项目不在一条赛道
                            A. 项目概述（
                            - 准确概括研究主题、技术主线与验证场景；给出与申请材料页/图的锚定。句句有据。〔来源：页/图〕

                            B. 来源锚定清单（要点式）
                            - 按“判断 → 来源来源（页/图/表/式） → 来源强度(高/中/低)”逐条列示，覆盖：研究基础、代表性成果、数据与算力条件、预研结果、应用场景。

                            C. 科学问题与创新性
                            - 逐条结构：问题 → 申请书来源 → 本评审判断（与国内外的差异/预期增量） → 来源强度。
                            - 明确核心创新点不超过3条，每条都需有“式/图/流程”的来源锚定。

                            D. 技术路线与可行性
                            - 按模块/数据/人力/合规展开；每条均需来源锚定，并给出“可行性结论 + 风险等级”。

                            E. 对照实验与验证闭环（强制）
                            - 基线集（≥3类可比方法：经典ML/可解释方法/现有AutoFE或SOTA）；
                            - 数据切分策略（患者/实体级、时间切分或K折；避免泄露）；
                            - 指标（分类/回归/聚类分别列出）；
                            - 统计检验（配对t或Wilcoxon，含多重校正）；
                            - 外部验证（至少1个独立公开队列）；
                            - 复现要素（环境/Docker或requirements、随机种子、日志与追溯）。
                            - 每项后给出“验收口径”；缺项写“〔来源：未见于材料〕”，并附MVP补充清单。

                            F. 风险与对策矩阵（表格）
                            - 列：风险 | 触发信号 | 缓解动作 | 备用方案 | 负责人/资源来源 | 时间
                            - 每行必须可执行（目标/动作/指标/验收/时间均完整），并标注来源强度与风险等级。

                            G. 里程碑与KPI（表格，M0–M6 / M6–M18 / M18–M36 / 结题）
                            - 列：阶段 | 交付物 | 量化指标 | 验收口径 | 阻断条件
                            - 每条需页/图/表/式来源；若未见，写“〔来源：未见于材料〕”，并给出最低可行门槛。

                            H. 主要问题与建议（3–6条，逐条“可执行”）
                            - 逐条格式：问题 → 目标 → 动作 → 指标 → 验收 → 负责人/资源 → 时间；每条标注风险等级与来源强度。

                            I. 评分与结论
                            - 维度（0.0–5.0）：创新性、科学意义、可行性、研究基础、团队与条件、经费匹配、风险控制。
                            - 给出加总/平均说明，并产出综合等级（优/良/中/差）。
                            - 资助建议：可资助 / 可资助并附条件 / 不建议资助。
                            - 若“附条件”，列2–3条“可量化里程碑”（阈值+时间点）。
                            - 每个分值后附一句依据+来源锚定；若依据不足，明确“〔来源：未见于材料〕”。

                            【额外要求】
                            - 若发现材料缺项导致无法判断，直接写明“因来源不足无法下结论”，并提供“最低可行补充清单（MVP）”，但不降低来源标准。
                            """
                                    ),
                                ]
                            )


                    input_data = {
                        "academic_analysis_report": raw_state.get("academic_analysis_report", "未进行学术分析"),
                        "future_influence_report": raw_state.get("future_influence_report", "未进行未来影响分析"),
                     #   "interdisciplinary_results": raw_state.get("interdisciplinary_results", []),
                        "research_structure": raw_state.get("research_structure", "无"),
                        "debate_results": _format_debate_results(raw_state.get("debate_results", {})),
                        # "final_analysis_summary": raw_state.get("final_analysis_summary", "未完成最终分析"),
                        # "completeness_check_result": _format_completeness_result(
                        #     raw_state.get("completeness_check_result", {})
                        # ),
                        # "human_feedback": raw_state.get("human_feedback", "无人类反馈"),
                    }

                    chain = prompt | graph.deep_thinking_llm
                    chunks: List[str] = []
                    async for chunk in chain.astream(input_data):
                        delta = getattr(chunk, "content", str(chunk))
                        if not delta:
                            continue
                        chunks.append(delta)
                        yield format_sse("report_delta", {"thread_id": thread_id, "delta": delta})

                    final_report = "".join(chunks)
                    raw_state["final_report"] = final_report
                    history.append(ChatMessage(role="assistant", content=final_report))
                    CHAT_SESSIONS[thread_id] = history
                    yield format_sse("report_complete", {"thread_id": thread_id, "report": final_report})
                    continue

                elif event == "report_ready":
                    yield format_sse(
                        "report_ready",
                        {
                            "thread_id": payload.get("thread_id"),
                            "downloads": payload.get("downloads", {}),
                        },
                    )
                else:
                    yield format_sse(event, payload)

                if event in {"done", "error", "cancelled"}:
                    break
        finally:
            SESSION_QUEUES.pop(thread_id, None)
            yield format_sse("done", {"thread_id": thread_id})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/control")
async def chat_control(control: SessionControl) -> Dict[str, str]:
    queues = SESSION_QUEUES.get(control.thread_id)
    if not queues:
        raise HTTPException(status_code=404, detail="session not found")

    target_queue: Optional["asyncio.Queue[SessionControl]"] = None
    target_cancel: Optional[asyncio.Event] = None

    if queues.allow_multi and control.file_id:
        target_queue = queues.sub_control_queues.get(control.file_id)
        target_cancel = queues.sub_cancel_events.get(control.file_id)
        if target_queue is None:
            raise HTTPException(status_code=404, detail="file session not found")

    target_queue = target_queue or queues.control_queue
    target_cancel = target_cancel or queues.cancel_event

    if control.action == "cancel":
        target_cancel.set()
        await target_queue.put(control)
    elif control.action == "resume":
        await target_queue.put(control)
    else:
        await target_queue.put(control)

    return {"status": "received", "action": control.action, "thread_id": control.thread_id}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "ProposalAgent API is running"}


