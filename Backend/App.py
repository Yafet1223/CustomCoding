"""
Coding Agent — FastAPI backend.

Wraps coding_agent.py's app_graph. The key difference from the Sage
backend: this agent can PAUSE mid-turn (via interrupt()) waiting for
approval. Two endpoints handle that:

  POST /api/chat    - send a new message. May come back either finished
                        ("status": "done") or paused ("status": "paused",
                        with the pending approval details).
  POST /api/resume   - approve or reject a paused action. May itself come
                        back paused again (if the agent wants to do
                        another approval-gated thing next) or done.

Requires:
    pip install fastapi uvicorn langgraph langchain-ollama --break-system-packages

Run:
    uvicorn app:app --reload --port 5001
Then open:
    http://localhost:5001
"""

import uuid
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.types import Command

from agent import app_graph


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]

class ToolTraceItem(BaseModel):
    tool: str
    args: dict
    result: str

class PendingApproval(BaseModel):
    action: str
    details: dict

class ChatResponse(BaseModel):
    status: Literal["done", "paused"]
    response: Optional[str] = None
    trace: list[ToolTraceItem] = []
    pending: Optional[PendingApproval] = None
    thread_id: str


app = FastAPI(title="Coding Agent")


def _turn_start_index(messages) -> int:
    """Find the index of the most recent HumanMessage — the start of the
    current turn. Using this (instead of 'message count right before this
    specific API call') means the trace stays correct even across a
    pause/resume cycle, since the AIMessage with the tool_call was added
    BEFORE the pause, but its ToolMessage result only arrives AFTER resume."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def _build_trace(messages, before_count: int) -> list[ToolTraceItem]:
    """Same pairing logic as the Sage backend: match tool_calls to their
    ToolMessage results, but only for messages added this turn."""
    new_messages = messages[before_count:]
    trace = []
    pending_calls = {}
    for m in new_messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                pending_calls[tc["id"]] = {"name": tc["name"], "args": tc["args"]}
        if isinstance(m, ToolMessage):
            call = pending_calls.get(m.tool_call_id, {"name": "unknown", "args": {}})
            trace.append(ToolTraceItem(tool=call["name"], args=call["args"], result=m.content))
    return trace


def _check_paused(config, thread_id, before_count, messages) -> ChatResponse:
    """After any invoke/resume, check whether the graph stopped because it
    finished, or because a tool called interrupt() and it's waiting."""
    state = app_graph.get_state(config)

    if state.next:
        # Graph is paused mid-execution — a tool called interrupt().
        payload = state.tasks[0].interrupts[0].value
        return ChatResponse(
            status="paused",
            trace=_build_trace(messages, before_count),
            pending=PendingApproval(action=payload.get("action", "unknown"), details=payload),
            thread_id=thread_id,
        )

    return ChatResponse(
        status="done",
        response=messages[-1].content,
        trace=_build_trace(messages, before_count),
        thread_id=thread_id,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = app_graph.invoke({"messages": [("user", message)]}, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    before_count = _turn_start_index(result["messages"])
    return _check_paused(config, thread_id, before_count, result["messages"])


@app.post("/api/resume", response_model=ChatResponse)
def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    existing = app_graph.get_state(config)
    if not existing or not existing.next:
        raise HTTPException(status_code=400, detail="No paused action for this thread_id")

    try:
        result = app_graph.invoke(Command(resume=req.decision), config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    before_count = _turn_start_index(result["messages"])
    return _check_paused(config, req.thread_id, before_count, result["messages"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
