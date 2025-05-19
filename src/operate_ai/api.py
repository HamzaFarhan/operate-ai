import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from operate_ai.cfo_graph import RunSQLResult, WriteSheetResult
from operate_ai.cfo_graph import thread as run_thread

app = FastAPI(title="Operate AI API")


# Base directory for workspaces
WORKSPACES_DIR = Path("workspaces")
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


class ThreadInfo(BaseModel):
    id: str
    name: str
    workspace_id: str
    created_at: str
    message_count: int


class MessageCreate(BaseModel):
    content: str = Field(..., description="Content of the message")


class MessageResponse(BaseModel):
    id: str
    content: str
    response: RunSQLResult | WriteSheetResult | str
    created_at: str


# Workspace endpoints
@app.post("/workspaces/", response_model=WorkspaceInfo)
async def create_workspace(workspace: WorkspaceCreate):
    workspace_id = str(uuid.uuid4())
    workspace_dir = WORKSPACES_DIR / workspace_id

    if workspace_dir.exists():
        raise HTTPException(status_code=400, detail="Workspace ID already exists")

    # Create workspace directory structure
    workspace_dir.mkdir(parents=True)
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
            except Exception as e:
                logger.error(f"Error reading workspace {workspace_dir}: {e}")

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

    # Save thread metadata
    metadata: dict[str, Any] = {
        "id": thread_id,
        "name": thread_data.name,
        "workspace_id": workspace_id,
        "created_at": str(asyncio.get_event_loop().time()),
        "messages": [],
    }
    (thread_dir / "metadata.json").write_text(json.dumps(metadata))

    return ThreadInfo(
        id=thread_id,
        name=thread_data.name,
        workspace_id=workspace_id,
        created_at=metadata["created_at"],
        message_count=0,
    )


@app.get("/workspaces/{workspace_id}/threads/", response_model=list[ThreadInfo])
async def list_threads(workspace_id: str) -> list[ThreadInfo]:
    workspace_dir = WORKSPACES_DIR / workspace_id

    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    threads: list[ThreadInfo] = []
    threads_dir = workspace_dir / "threads"

    for thread_dir in threads_dir.iterdir():
        if thread_dir.is_dir() and (thread_dir / "metadata.json").exists():
            try:
                metadata = eval((thread_dir / "metadata.json").read_text())
                message_count = len(metadata.get("messages", []))

                threads.append(
                    ThreadInfo(
                        id=metadata["id"],
                        name=metadata["name"],
                        workspace_id=workspace_id,
                        created_at=metadata["created_at"],
                        message_count=message_count,
                    )
                )
            except Exception as e:
                logger.error(f"Error reading thread {thread_dir}: {e}")

    return threads


# Message endpoints
@app.post("/workspaces/{workspace_id}/threads/{thread_id}/messages/", response_model=MessageResponse)
async def create_message(workspace_id: str, thread_id: str, message: MessageCreate):
    workspace_dir = WORKSPACES_DIR / workspace_id
    thread_dir = workspace_dir / "threads" / thread_id

    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    if not thread_dir.exists():
        raise HTTPException(status_code=404, detail="Thread not found")

    # Run the thread with the message content
    try:
        response = await run_thread(thread_dir=thread_dir, user_prompt=message.content)

        # Update thread metadata with new message
        metadata_file = thread_dir / "metadata.json"
        if metadata_file.exists():
            metadata = eval(metadata_file.read_text())
        else:
            metadata: dict[str, Any] = {
                "id": thread_id,
                "name": "Unknown",
                "workspace_id": workspace_id,
                "created_at": str(asyncio.get_event_loop().time()),
                "messages": [],
            }

        message_id = str(uuid.uuid4())
        created_at = str(asyncio.get_event_loop().time())
        message_data = {
            "id": message_id,
            "content": message.content,
            "response": response if isinstance(response, str) else "Please Review",
            "created_at": created_at,
        }

        metadata["messages"].append(message_data)
        metadata_file.write_text(json.dumps(metadata))

        return MessageResponse(id=message_id, content=message.content, response=response, created_at=created_at)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/workspaces/{workspace_id}/threads/{thread_id}/messages/", response_model=list[MessageResponse])
async def list_messages(workspace_id: str, thread_id: str) -> list[MessageResponse]:
    workspace_dir = WORKSPACES_DIR / workspace_id
    thread_dir = workspace_dir / "threads" / thread_id

    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    if not thread_dir.exists():
        raise HTTPException(status_code=404, detail="Thread not found")

    metadata_file = thread_dir / "metadata.json"
    if not metadata_file.exists():
        return []

    try:
        metadata = eval(metadata_file.read_text())
        messages = metadata.get("messages", [])

        return [
            MessageResponse(
                id=msg["id"], content=msg["content"], response=msg["response"], created_at=msg["created_at"]
            )
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error reading messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading messages: {str(e)}")
