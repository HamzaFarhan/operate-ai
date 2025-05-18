from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select

from operate_ai.cfo import thread as run_agent

# ────────────────────────────
# Configuration
# ────────────────────────────

DATA_ROOT = Path("data")  # every workspace lives in DATA_ROOT / <workspace_id>
DB_URL = "sqlite:///operate_ai.db"
AGENT_EXECUTION_TIMEOUT_SECONDS = 20

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


class Thread(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    workspace_id: int = SQLField(foreign_key="workspace.id", index=True)

    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC), nullable=False)
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC), nullable=False)

    thread_dir: str = SQLField(nullable=False)
    agent_status: str = SQLField(
        default="IDLE", nullable=False
    )  # e.g., IDLE, QUEUED, PROCESSING, AWAITING_INPUT, ERROR
    last_agent_message: str | None = SQLField(default=None, nullable=True)


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
    agent_status: str
    last_agent_message: str | None


class MessageSend(BaseModel):
    content: str | None = None  # empty string or None == "continue"


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
    # Ensure DATA_ROOT is Path object if it's not already
    return Path(DATA_ROOT) / str(workspace_id)


def thread_path(workspace_id: int, thread_id: int) -> Path:
    return workspace_path(workspace_id=workspace_id) / "threads" / str(thread_id)


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ────────────────────────────
# CFO Agent integration
# ────────────────────────────

# Your agent's async thread() function lives in cfo.py


async def _agent_runner(db_url: str, workspace_id: int, thread_id: int, prompt: str | None) -> None:
    """
    Runs the agent for a single prompt in a specific thread.
    Updates thread status and stores agent's final message.
    Includes a timeout for the agent execution.
    """
    # Create a new engine and session for this background task
    # Note: echo=False is important for background tasks to avoid excessive logging from engine
    task_engine = create_engine(db_url, echo=False)

    def update_thread_db(status: str, message: str | None = None, set_message: bool = False) -> None:
        with Session(task_engine) as session:
            db_thread = session.get(Thread, thread_id)
            if db_thread:
                db_thread.agent_status = status
                if set_message:  # Allows explicitly setting or clearing the message
                    db_thread.last_agent_message = message
                db_thread.updated_at = datetime.now(UTC)
                session.add(db_thread)
                session.commit()
            else:
                logger.warning(f"Thread {thread_id} not found during agent run status update.")

    try:
        update_thread_db(status="PROCESSING", message="Agent is processing...", set_message=True)

        t_path = thread_path(workspace_id=workspace_id, thread_id=thread_id)
        # Define paths for the agent based on current workspace/thread
        agent_data_dir = workspace_path(workspace_id=workspace_id) / "data"
        agent_analysis_dir = t_path / "analysis"
        agent_results_dir = t_path / "results"  # For message_history.json and potential agent outputs

        ensure_dirs(agent_data_dir)
        ensure_dirs(agent_analysis_dir)
        ensure_dirs(agent_results_dir)

        agent_task = run_agent(
            user_prompt=prompt or "",
            data_dir=agent_data_dir,
            analysis_dir=agent_analysis_dir,
            results_dir=agent_results_dir,
        )

        agent_response = await asyncio.wait_for(agent_task, timeout=AGENT_EXECUTION_TIMEOUT_SECONDS)

        update_thread_db(status="AWAITING_INPUT", message=agent_response, set_message=True)
        logger.info(f"Agent for thread {thread_id} finished, awaiting input. Message: {agent_response}")

    except asyncio.TimeoutError:
        logger.warning(
            f"Agent execution timed out for thread {thread_id} after {AGENT_EXECUTION_TIMEOUT_SECONDS} seconds."
        )
        timeout_message = f"Agent processing timed out after {AGENT_EXECUTION_TIMEOUT_SECONDS} seconds."
        update_thread_db(status="ERROR", message=timeout_message, set_message=True)
    except Exception as e:
        logger.error(f"Agent failed for thread {thread_id}: {e}", exc_info=True)
        error_message = f"Agent error: {str(e)}"
        update_thread_db(status="ERROR", message=error_message, set_message=True)


# ────────────────────────────
# FastAPI application
# ────────────────────────────

app = FastAPI(title="OperateAI CFO Agent Backend")
init_db()


# ── Workspace endpoints ────────────────────────────────────────────────


@app.post("/workspaces", response_model=WorkspaceRead, status_code=status.HTTP_201_CREATED)
def create_workspace(payload: WorkspaceCreate) -> WorkspaceRead:
    with Session(engine) as session:
        ws = Workspace(name=payload.name, workspace_dir=str(workspace_path(-1)))
        session.add(ws)
        session.flush()  # 🗝 get auto-generated id before commit

        # Now that we have the id we can build the real dir and update
        ws.workspace_dir = str(workspace_path(ws.id))  # type: ignore
        ensure_dirs(Path(ws.workspace_dir) / "data")
        ensure_dirs(Path(ws.workspace_dir) / "threads")

        session.add(ws)
        session.commit()
        session.refresh(ws)
        return WorkspaceRead.model_validate(ws.model_dump())


@app.get("/workspaces", response_model=list[WorkspaceRead])
def list_workspaces() -> list[WorkspaceRead]:
    with Session(engine) as session:
        workspaces = session.exec(select(Workspace)).all()
        return [WorkspaceRead.model_validate(w.model_dump()) for w in workspaces]


# ── Thread endpoints ──────────────────────────────────────────────────


@app.post("/workspaces/{workspace_id:int}/threads", response_model=ThreadRead, status_code=status.HTTP_201_CREATED)
def create_thread(workspace_id: int, payload: ThreadCreate, background_tasks: BackgroundTasks) -> ThreadRead:
    with Session(engine) as session:
        ws = session.get(Workspace, workspace_id)
        if not ws:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

        th = Thread(
            workspace_id=workspace_id, thread_dir=str(thread_path(workspace_id=workspace_id, thread_id=-1))
        )
        session.add(th)
        session.flush()

        th.thread_dir = str(thread_path(workspace_id=workspace_id, thread_id=th.id))  # type: ignore
        ensure_dirs(Path(th.thread_dir))

        if payload.initial_prompt is not None:
            th.agent_status = "QUEUED"
            th.last_agent_message = "Agent is preparing to process initial prompt..."
            background_tasks.add_task(_agent_runner, DB_URL, workspace_id, th.id, payload.initial_prompt)  # type: ignore
        else:
            th.agent_status = "IDLE"  # No prompt, so idle
            th.last_agent_message = None

        session.add(th)
        session.commit()
        session.refresh(th)
        return ThreadRead.model_validate(th.model_dump())


@app.get("/workspaces/{workspace_id:int}/threads", response_model=list[ThreadRead])
def list_threads(workspace_id: int) -> list[ThreadRead]:
    with Session(engine) as session:
        threads = session.exec(select(Thread).where(Thread.workspace_id == workspace_id)).all()
        return [ThreadRead.model_validate(t.model_dump()) for t in threads]


@app.get("/workspaces/{workspace_id:int}/threads/{thread_id:int}", response_model=ThreadRead)
def get_thread_details(workspace_id: int, thread_id: int) -> ThreadRead:
    with Session(engine) as session:
        db_thread = session.exec(
            select(Thread).where(Thread.id == thread_id).where(Thread.workspace_id == workspace_id)
        ).first()
        if not db_thread:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        return ThreadRead.model_validate(db_thread.model_dump())


# ── Messaging endpoint ────────────────────────────────────────────────


@app.post("/workspaces/{workspace_id:int}/threads/{thread_id:int}/messages", status_code=status.HTTP_202_ACCEPTED)
def send_message(
    workspace_id: int, thread_id: int, payload: MessageSend, background_tasks: BackgroundTasks
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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

        # Update status before starting task
        thread.agent_status = "QUEUED"
        thread.last_agent_message = "Agent is preparing to process message..."
        thread.updated_at = datetime.now(UTC)
        session.add(thread)
        session.commit()

    # enqueue agent run
    background_tasks.add_task(_agent_runner, DB_URL, workspace_id, thread_id, payload.content)

    return {"detail": "Message accepted – agent is processing."}


# ────────────────────────────
# (Optional) Health check
# ────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
