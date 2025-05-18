# backend.py
#
# How to run:
#   uvicorn backend:app --reload
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import Relationship, Session, SQLModel, create_engine, select

from .cfo import thread as run_agent

# ────────────────────────────
# Configuration
# ────────────────────────────

DATA_ROOT = Path("data")  # every workspace lives in DATA_ROOT / <workspace_id>
DB_URL = "sqlite:///operate_ai.db"

engine = create_engine(DB_URL, echo=False)


# ────────────────────────────
# ORM Models
# ────────────────────────────


class Workspace(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    name: str = SQLField(index=True)

    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC), nullable=False)
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC), nullable=False)

    workspace_dir: str = SQLField(nullable=False)

    threads: list["Thread"] = Relationship(back_populates="workspace")


class Thread(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    workspace_id: int = SQLField(foreign_key="workspace.id", index=True)

    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC), nullable=False)
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC), nullable=False)

    thread_dir: str = SQLField(nullable=False)

    workspace: Workspace | None = Relationship(back_populates="threads")


# ────────────────────────────
# API Schemas
# ────────────────────────────


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1)


class WorkspaceRead(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime


class ThreadCreate(BaseModel):
    initial_prompt: str | None = None  # optional – user may create then prompt later


class ThreadRead(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime


class MessageSend(BaseModel):
    content: str | None = None  # empty string or None == “continue”


class AgentOutput(BaseModel):
    thread_id: int
    output_type: str
    payload: str


# ────────────────────────────
# Helper utilities
# ────────────────────────────


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def workspace_path(workspace_id: int) -> Path:
    return DATA_ROOT / str(workspace_id)


def thread_path(workspace_id: int, thread_id: int) -> Path:
    return workspace_path(workspace_id) / "threads" / str(thread_id)


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ────────────────────────────
# CFO Agent integration
# ────────────────────────────

# Your agent's async thread() function lives in cfo.py


async def _agent_runner(
    workspace_id: int,
    thread_id: int,
    prompt: str | None,
) -> None:
    """
    Runs the agent for a single prompt in a specific thread.

    Creates / updates the thread directory layout that cfo.thread() expects:
        • <thread_dir>/analysis
        • <thread_dir>/results
        • <thread_dir>/message_history.json

    The cfo.thread() function internally handles message_history persistence
    once the paths exist.
    """
    t_path = thread_path(workspace_id, thread_id)
    analysis_dir = t_path / "analysis"
    results_dir = t_path / "results"
    ensure_dirs(analysis_dir)
    ensure_dirs(results_dir)

    # cfo.thread() writes to message_history.json by itself.
    # We just call it with the user prompt (empty string == “continue”)
    await run_agent(prompt or "")


# ────────────────────────────
# FastAPI application
# ────────────────────────────

app = FastAPI(title="OperateAI CFO Agent Backend")
init_db()


# ── Workspace endpoints ────────────────────────────────────────────────


@app.post("/workspaces", response_model=WorkspaceRead, status_code=status.HTTP_201_CREATED)
def create_workspace(payload: WorkspaceCreate) -> WorkspaceRead:
    with Session(engine) as session:
        ws = Workspace(
            name=payload.name,
            workspace_dir=str(workspace_path(-1)),  # temp, update after flush
        )
        session.add(ws)
        session.flush()  # 🗝 get auto-generated id before commit

        # Now that we have the id we can build the real dir and update
        ws.workspace_dir = str(workspace_path(ws.id))
        ensure_dirs(Path(ws.workspace_dir) / "data")
        ensure_dirs(Path(ws.workspace_dir) / "threads")

        session.add(ws)
        session.commit()
        session.refresh(ws)
        return WorkspaceRead.model_validate(ws)


@app.get("/workspaces", response_model=list[WorkspaceRead])
def list_workspaces() -> list[WorkspaceRead]:
    with Session(engine) as session:
        workspaces = session.exec(select(Workspace)).all()
        return [WorkspaceRead.model_validate(w) for w in workspaces]


# ── Thread endpoints ──────────────────────────────────────────────────


@app.post(
    "/workspaces/{workspace_id:int}/threads",
    response_model=ThreadRead,
    status_code=status.HTTP_201_CREATED,
)
def create_thread(workspace_id: int, payload: ThreadCreate, background_tasks: BackgroundTasks) -> ThreadRead:
    with Session(engine) as session:
        ws = session.get(Workspace, workspace_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")

        th = Thread(
            workspace_id=workspace_id,
            thread_dir=str(thread_path(workspace_id, -1)),  # temp
        )
        session.add(th)
        session.flush()

        # update thread_dir now that id is known
        th.thread_dir = str(thread_path(workspace_id, th.id))
        ensure_dirs(Path(th.thread_dir))

        session.add(th)
        session.commit()
        session.refresh(th)

        # Kick the agent if an initial prompt is supplied
        if payload.initial_prompt is not None:
            background_tasks.add_task(_agent_runner, workspace_id, th.id, payload.initial_prompt)

        return ThreadRead.model_validate(th)


@app.get(
    "/workspaces/{workspace_id:int}/threads",
    response_model=list[ThreadRead],
)
def list_threads(workspace_id: int) -> list[ThreadRead]:
    with Session(engine) as session:
        threads = session.exec(select(Thread).where(Thread.workspace_id == workspace_id)).all()
        return [ThreadRead.model_validate(t) for t in threads]


# ── Messaging endpoint ────────────────────────────────────────────────


@app.post(
    "/workspaces/{workspace_id:int}/threads/{thread_id:int}/messages",
    status_code=status.HTTP_202_ACCEPTED,
)
def send_message(
    workspace_id: int,
    thread_id: int,
    payload: MessageSend,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    • `content = str | None`
        • None or ""   → continue (empty prompt)
        • otherwise    → normal user message

    We immediately return 202; the agent work happens in a background task.
    Front-end (Streamlit) should poll another endpoint (or watch filesystem /
    message_history.json) to display results.
    """
    with Session(engine) as session:
        thread = session.exec(
            select(Thread).where(Thread.id == thread_id).where(Thread.workspace_id == workspace_id)
        ).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

    # enqueue agent run
    background_tasks.add_task(_agent_runner, workspace_id, thread_id, payload.content)

    return {"detail": "Message accepted – agent is processing."}


# ────────────────────────────
# (Optional) Health check
# ────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
