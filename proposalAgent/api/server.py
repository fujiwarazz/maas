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
from typing import Any, AsyncGenerator, Dict, List, Optional

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

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    flat_state: Dict[str, Any] = (
        flatten_state(final_state) if isinstance(final_state, dict) else {}
    )

    academic_report = (
        flat_state.get("academic_analysis_report")
        or "尚未生成学术分析报告"
    )
    future_report = (
        flat_state.get("future_influence_report")
        or "尚未生成未来影响力分析报告"
    )
    debate_results = flat_state.get("debate_results", {}) or final_state.get("debate_results", {})
    judge_summaries = extract_judge_summaries(debate_results)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def draw_text_block(title: str, content: str, start_y: float) -> float:
        pdf.setFont("STSong-Light", 14)
        pdf.drawString(40, start_y, title)
        pdf.setFont("STSong-Light", 11)

        text_obj = pdf.beginText(40, start_y - 24)
        text_obj.setFont("STSong-Light", 11)
        wrapped_lines: List[str] = []
        for line in content.splitlines() or [""]:
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(textwrap.wrap(line, width=60))

        for line in wrapped_lines:
            if not line:
                text_obj.textLine(" ")
            else:
                text_obj.textLine(line)
        pdf.drawText(text_obj)
        return text_obj.getY() - 16

    pdf.setTitle("ProposalAgent Analysis Report")

    y_position = height - 60
    pdf.setFont("STSong-Light", 16)
    pdf.drawString(40, y_position, "ProposalAgent 评估输出摘要")
    y_position -= 40

    y_position = draw_text_block("线程 ID", str(thread_id), y_position)
    y_position = draw_text_block("学术分析报告", academic_report, y_position - 16)
    y_position = draw_text_block("未来影响力分析", future_report, y_position - 16)

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
        judge_lines.append("尚未生成辩论裁判总结")

    y_position = draw_text_block("辩论裁判最终结论", "\n".join(judge_lines), y_position - 16)

    pdf.setFont("STSong-Light", 10)
    pdf.drawRightString(width - 40, 30, "Powered by ProposalAgent")

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    pdf_bytes = buffer.read()

    object_key = f"reports/{thread_id}/{uuid.uuid4().hex}.pdf"
    upload_bytes_to_oss(object_key, pdf_bytes, content_type="application/pdf")
    url = generate_presigned_get_url(object_key)

    return {"report": url, "object_key": object_key}

class SessionQueues:
    def __init__(self) -> None:
        self.event_queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.control_queue: "asyncio.Queue[SessionControl]" = asyncio.Queue()
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.final_state: Optional[Dict[str, Any]] = None
        self.report_urls: Optional[Dict[str, str]] = None





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
                flat[key] = value

    return flat






@app.post("/upload")
async def upload_file(
    files: List[UploadFile] = File(...),
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for file in files:
        suffix = Path(file.filename or "uploaded").suffix
        dest_path = UPLOAD_ROOT / f"{uuid.uuid4().hex}{suffix}"
        # 重新定位到流起始位置，分块读取并写入，避免一次性读取为空或耗尽内存
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
    thread_id: str,
    thread_config: Dict[str, Any],
    queues: SessionQueues,
) -> None:
    final_state: Optional[Dict[str, Any]] = None
    print("🔥 run_evaluation: thread_id=%s, thread_config=%s", thread_id, thread_config)
    logger.info("run_evaluation: thread_id=%s, thread_config=%s", thread_id, thread_config)

    async def feedback_handler(interrupt_payload: Any) -> Optional[str]:
        payload_value = getattr(interrupt_payload, "value", interrupt_payload)
        prompt = build_interrupt_prompt(final_state, payload_value)

        interrupt_event: Dict[str, Any] = {
            "thread_id": thread_id,
            "prompt": prompt,
        }

        if payload_value is not None:
            interrupt_event["payload"] = payload_value

        await queues.event_queue.put(("interrupt", interrupt_event))
        logger.info("接收到中断事件: %s", interrupt_event)
        while True:
            control = await queues.control_queue.get()
            if control.action == "cancel":
                queues.cancel_event.set()
                return None

            if control.action == "resume":
                queues.cancel_event.clear()
                return control.feedback or ""


    initial_state = graph.propagator.create_initial_state(
        user_prompt=request.query,
        user_interest=request.user_interest,
        filepath=request.file_ids[0],
    )

    try:
        async for chunk in graph.stream_project(
            initial_state,
            thread_config,
            feedback_handler=feedback_handler,
            cancel_event=queues.cancel_event,
        ):
            if queues.cancel_event.is_set():
                await queues.event_queue.put(("cancelled", {"thread_id": thread_id}))
                return

            if "__interrupt__" in chunk:
                final_state = chunk.get("state")
                continue

            final_state = chunk.get("state")

        final_state = final_state or graph.curr_state or {}

        serializable_state = make_serializable(final_state)

        with open("final_state.json", "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, ensure_ascii=False)
            print(f"最终状态已保存到 final_state.json，长度: {len(serializable_state)}")

        SESSION_QUEUES[thread_id].final_state = final_state

        try:
            report_urls = await asyncio.to_thread(
                create_and_upload_reports, final_state, thread_id
            )
            queues.report_urls = report_urls
        except Exception as exc_upload:  # noqa: BLE001
            logger.exception("生成或上传报告文件失败: %s", exc_upload)
            queues.report_urls = None

        await queues.event_queue.put(
            ("stage_complete", {"thread_id": thread_id, "state": serializable_state})
        )
        if queues.report_urls:
            await queues.event_queue.put(
                (
                    "report_ready",
                    {
                        "thread_id": thread_id,
                        "downloads": queues.report_urls,
                    },
                )
            )
        await queues.event_queue.put(("done", {"thread_id": thread_id}))
    except Exception as exc:  # noqa: BLE001
        await queues.event_queue.put(("error", {"thread_id": thread_id, "message": str(exc)}))
    finally:
        SESSION_QUEUES.pop(thread_id, None)



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
    queues = SessionQueues()
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

    await queues.event_queue.put(("session", {"thread_id": thread_id}))

    # 运行携程任务，向queue中添加chunk
    asyncio.create_task(run_evaluation(graph, request, thread_id, thread_config, queues))

    async def event_generator() -> AsyncGenerator[bytes, None]:
        try:
            while True:
                event, payload = await queues.event_queue.get()
                if event == "stage_complete":
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
                                """你是一个专业的项目评估报告生成智能体。你的任务是基于所有收集到的分析信息，生成一份全面、专业、结构化的项目评价报表。""",
                            ),
                            (
                                "human",
                                """请基于以下全部分析信息，生成最终尽可能详尽的项目评估报告：\n学术分析：{academic_analysis_report}\n社会分析：{social_analysis_report}\n未来影响分析：{future_influence_report}\n跨学科分析结果：{interdisciplinary_results}\n辩论结果：{debate_results}\n最终分析摘要：{final_analysis_summary}\n完备性检查结果：{completeness_check_result}\n人类反馈：{human_feedback}""",
                            ),
                        ]
                    )

                    input_data = {
                        "academic_analysis_report": raw_state.get("academic_analysis_report", "未进行学术分析"),
                        "social_analysis_report": raw_state.get("social_analysis_report", "未进行社会分析"),
                        "future_influence_report": raw_state.get("future_influence_report", "未进行未来影响分析"),
                        "interdisciplinary_results": raw_state.get("interdisciplinary_results", []),
                        "debate_results": _format_debate_results(raw_state.get("debate_results", {})),
                        "final_analysis_summary": raw_state.get("final_analysis_summary", "未完成最终分析"),
                        "completeness_check_result": _format_completeness_result(
                            raw_state.get("completeness_check_result", {})
                        ),
                        "human_feedback": raw_state.get("human_feedback", "无人类反馈"),
                    }

                    chain = prompt | graph.quick_thinking_llm
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

    if control.action == "cancel":
        queues.cancel_event.set()
        await queues.control_queue.put(control)
    elif control.action == "resume":
        await queues.control_queue.put(control)
    else:
        await queues.control_queue.put(control)

    return {"status": "received", "action": control.action, "thread_id": control.thread_id}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "ProposalAgent API is running"}


