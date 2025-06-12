import asyncio
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from operate_ai.cfo_graph import (
    RunSQLResult,
    TaskResult,
    WriteDataToExcelResult,
    get_prev_state_path,
    load_prev_state,
)
from operate_ai.cfo_graph import thread as run_thread

load_dotenv()

app = FastAPI(title="Operate AI API")


# Base directory for workspaces
WORKSPACES_DIR = Path(os.getenv("WORKSPACES_DIR", "workspaces"))
WORKSPACES_DIR.mkdir(exist_ok=True)


# Models
class WorkspaceCreate(BaseModel):
    name: str = Field(..., description="Name of the workspace")


class WorkspaceInfo(BaseModel):
    id: str
    name: str
    created_at: str
    thread_count: int


class ThreadCreate(BaseModel):
    name: str = Field(..., description="Name of the thread")
    prev_state_path: str | None = Field(None, description="Path to the previous state")


class ThreadInfo(BaseModel):
    id: str
    name: str
    workspace_id: str
    created_at: str


class MessageCreate(BaseModel):
    content: str = Field(..., description="Content of the message")
    prev_state_path: str | None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: RunSQLResult | WriteDataToExcelResult | str
    state_path: str


# Workspace endpoints
@app.post("/workspaces/", response_model=WorkspaceInfo)
async def create_workspace(workspace: WorkspaceCreate):
    workspace_id = str(uuid.uuid4())
    workspace_dir = WORKSPACES_DIR / workspace_id

    if workspace_dir.exists():
        raise HTTPException(status_code=400, detail="Workspace ID already exists")

    # Create workspace directory structure
    workspace_dir.mkdir(parents=True)
    logger.info(f"Created workspace directory at {workspace_dir}")
    (workspace_dir / "data").mkdir()
    (workspace_dir / "threads").mkdir()

    # Save workspace metadata
    metadata = {"id": workspace_id, "name": workspace.name, "created_at": str(asyncio.get_event_loop().time())}
    (workspace_dir / "metadata.json").write_text(str(metadata))

    return WorkspaceInfo(id=workspace_id, name=workspace.name, created_at=metadata["created_at"], thread_count=0)


@app.get("/workspaces/", response_model=list[WorkspaceInfo])
async def list_workspaces() -> list[WorkspaceInfo]:
    workspaces: list[WorkspaceInfo] = []

    for workspace_dir in WORKSPACES_DIR.iterdir():
        if workspace_dir.is_dir() and (workspace_dir / "metadata.json").exists():
            try:
                metadata = eval((workspace_dir / "metadata.json").read_text())
                thread_count = sum(1 for _ in (workspace_dir / "threads").iterdir() if _.is_dir())

                workspaces.append(
                    WorkspaceInfo(
                        id=metadata["id"],
                        name=metadata["name"],
                        created_at=metadata["created_at"],
                        thread_count=thread_count,
                    )
                )
            except Exception:
                logger.exception(f"Error reading workspace {workspace_dir}")

    return workspaces


# Thread endpoints
@app.post("/workspaces/{workspace_id}/threads/", response_model=ThreadInfo)
async def create_thread(workspace_id: str, thread_data: ThreadCreate) -> ThreadInfo:
    workspace_dir = WORKSPACES_DIR / workspace_id

    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    thread_id = str(uuid.uuid4())
    thread_dir = workspace_dir / "threads" / thread_id

    # Create thread directory structure
    thread_dir.mkdir(parents=True)
    (thread_dir / "analysis").mkdir()
    (thread_dir / "results").mkdir()
    (thread_dir / "states").mkdir()
    logger.info(
        f"Creating thread {thread_data.name} with prev_state_path {thread_data.prev_state_path} and id {thread_id}"
    )
    if thread_data.prev_state_path:
        prev_thread_id = Path(thread_data.prev_state_path).parent.parent.name
        new_state_path = (thread_dir / "states") / Path(thread_data.prev_state_path).name
        try:
            prev_state = Path(thread_data.prev_state_path).read_text()
            state_content = prev_state.replace(prev_thread_id, thread_id)
            new_state_path.write_text(state_content)
            prev_analysis_dir = Path(thread_data.prev_state_path).parent.parent / "analysis"
            new_analysis_dir = thread_dir / "analysis"
            for file in prev_analysis_dir.iterdir():
                shutil.copy2(file, new_analysis_dir / file.name)
            prev_results_dir = Path(thread_data.prev_state_path).parent.parent / "results"
            new_results_dir = thread_dir / "results"
            for file in prev_results_dir.iterdir():
                shutil.copy2(file, new_results_dir / file.name)
        except Exception:
            logger.exception("Error copying previous state")

    thread_info = ThreadInfo(
        id=thread_id,
        name=thread_data.name,
        workspace_id=workspace_id,
        created_at=str(asyncio.get_event_loop().time()),
    )

    (thread_dir / "metadata.json").write_text(thread_info.model_dump_json())
    return thread_info


@app.get("/workspaces/{workspace_id}/threads/", response_model=list[ThreadInfo])
async def list_threads(workspace_id: str) -> list[ThreadInfo]:
    workspace_dir = WORKSPACES_DIR / workspace_id

    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    threads: list[ThreadInfo] = []
    threads_dir = workspace_dir / "threads"

    for thread_dir in threads_dir.iterdir():
        states_dir = thread_dir / "states"
        if thread_dir.is_dir() and states_dir.exists():
            try:
                threads.append(ThreadInfo.model_validate_json((thread_dir / "metadata.json").read_text()))
            except Exception:
                logger.exception(f"Error reading thread {thread_dir}")

    return threads


# Message endpoints
@app.post("/workspaces/{workspace_id}/threads/{thread_id}/messages/", response_model=ChatMessage)
async def create_message(workspace_id: str, thread_id: str, message: MessageCreate):
    workspace_dir = WORKSPACES_DIR / workspace_id
    thread_dir = workspace_dir / "threads" / thread_id

    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    if not thread_dir.exists():
        raise HTTPException(status_code=404, detail="Thread not found")
    try:
        response = await run_thread(
            thread_dir=thread_dir, user_prompt=message.content, prev_state_path=message.prev_state_path
        )
        if isinstance(response, TaskResult):
            response = response.message
        return ChatMessage(role="assistant", content=response, state_path=str(get_prev_state_path(thread_dir)))
    except Exception as e:
        logger.exception("Error processing message")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/workspaces/{workspace_id}/threads/{thread_id}/messages/", response_model=list[ChatMessage])
async def list_messages(workspace_id: str, thread_id: str) -> list[ChatMessage]:
    workspace_dir = WORKSPACES_DIR / workspace_id
    thread_dir = workspace_dir / "threads" / thread_id

    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    if not thread_dir.exists():
        raise HTTPException(status_code=404, detail="Thread not found")
    try:
        prev_state = await load_prev_state(thread_dir=thread_dir)
        return [
            ChatMessage(role=message["role"], content=message["content"], state_path=message["state_path"])
            for message in prev_state.chat_messages
        ]
    except Exception as e:
        logger.exception("Error reading messages")
        raise HTTPException(status_code=500, detail=f"Error reading messages: {str(e)}")


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}
