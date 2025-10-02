from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import AIMessageChunk

from proposalAgent.graphs.proposal_graph import ProposalAgentGraph
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.models import (
    ChatMessage,
    ChatMessageRequest,
    ChatRequest,
    SessionControl,
)

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

class SessionQueues:
    def __init__(self) -> None:
        self.event_queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.control_queue: "asyncio.Queue[SessionControl]" = asyncio.Queue()
        self.cancel_event: asyncio.Event = asyncio.Event()





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
        
        with open("final_state.json", "w") as f:
            json.dump(final_state, f)
            print(f"最终状态已保存到 final_state.json，长度: {len(final_state)}")
            
        if final_state and final_state.get("final_report"):
            await queues.event_queue.put(
                ("report", {"thread_id": thread_id, "report": final_state["final_report"]})
            )
        else:
            await queues.event_queue.put(("done", {"thread_id": thread_id}))
    except Exception as exc:  # noqa: BLE001
        await queues.event_queue.put(("error", {"thread_id": thread_id, "message": str(exc)}))
    finally:
        await queues.event_queue.put(("done", {"thread_id": thread_id}))



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

    thread_id, thread_config = graph.create_session(request.thread_id)
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
                yield format_sse(event, payload)
                
                # 当评估完成时，将结果添加到对话历史
                if event == "report":
                    final_report = payload.get("report", "")
                    if final_report:
                        history.append(ChatMessage(role="assistant", content=final_report))
                        CHAT_SESSIONS[thread_id] = history
                
                if event in {"done", "error", "cancelled"}:
                    break
        finally:
            SESSION_QUEUES.pop(thread_id, None)

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


