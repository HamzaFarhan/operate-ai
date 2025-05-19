from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from .cfo_graph import thread as run_thread

app = FastAPI(title="Operate AI API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for workspaces
WORKSPACES_DIR = Path("workspaces")
WORKSPACES_DIR.mkdir(exist_ok=True)


class WorkspaceCreate(BaseModel):
    name: str = Field(..., description="Name of the workspace")


class Workspace(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Name of the workspace")


class ThreadCreate(BaseModel):
    name: str = Field(..., description="Name of the thread")


class Thread(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Name of the thread")
    workspace_id: UUID = Field(..., description="ID of the workspace this thread belongs to")


class Message(BaseModel):
    content: str = Field(..., description="Content of the message")


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[UUID, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, thread_id: UUID):
        await websocket.accept()
        self.active_connections.setdefault(thread_id, []).append(websocket)
        logger.info(
            f"WebSocket connected for thread {thread_id}. Total connections for thread: {len(self.active_connections[thread_id])}"
        )

    def disconnect(self, websocket: WebSocket, thread_id: UUID):
        if thread_id in self.active_connections:
            try:
                self.active_connections[thread_id].remove(websocket)
                logger.info(
                    f"WebSocket disconnected for thread {thread_id}. Remaining connections: {len(self.active_connections[thread_id])}"
                )
                if not self.active_connections[thread_id]:
                    del self.active_connections[thread_id]  # Clean up if no connections left
            except ValueError:
                logger.warning(f"WebSocket to remove not found in thread {thread_id} list.")
        else:
            logger.warning(f"Thread {thread_id} not found in active_connections for disconnect.")

    async def send_message(self, message: str, thread_id: UUID):
        connections_for_thread = self.active_connections.get(thread_id, [])
        logger.info(
            f"Attempting to send message to {len(connections_for_thread)} connections for thread {thread_id}: {message[:100]}..."
        )
        if not connections_for_thread:
            logger.warning(f"No active WebSocket connections found for thread {thread_id} to send message.")
            return

        for connection in list(connections_for_thread):  # Iterate over a copy for safe removal
            try:
                logger.debug(f"Sending to connection: {connection} for thread {thread_id}")
                await connection.send_text(message)
                logger.debug(f"Successfully sent to {connection} for thread {thread_id}")
            except Exception as e:
                logger.error(
                    f"Error sending message to websocket {connection} for thread {thread_id}: {e}", exc_info=True
                )
                # Remove failed connection
                self.disconnect(connection, thread_id)


manager = ConnectionManager()


@app.post("/workspaces/", response_model=Workspace)
async def create_workspace(workspace: WorkspaceCreate) -> Workspace:
    """Create a new workspace."""
    workspace_id = uuid4()
    workspace_dir = WORKSPACES_DIR / str(workspace_id)
    workspace_dir.mkdir(exist_ok=True)

    # Create data directory
    (workspace_dir / "data").mkdir(exist_ok=True)

    # Save workspace metadata
    workspace_obj = Workspace(id=workspace_id, name=workspace.name)
    (workspace_dir / "metadata.json").write_text(workspace_obj.model_dump_json())

    return workspace_obj


@app.get("/workspaces/", response_model=list[Workspace])
async def list_workspaces() -> list[Workspace]:
    """List all workspaces."""
    workspaces: list[Workspace] = []
    for workspace_dir in WORKSPACES_DIR.iterdir():
        if workspace_dir.is_dir():
            metadata_file = workspace_dir / "metadata.json"
            if metadata_file.exists():
                workspace = Workspace.model_validate_json(metadata_file.read_text())
                workspaces.append(workspace)
    return workspaces


@app.get("/workspaces/{workspace_id}", response_model=Workspace)
async def get_workspace(workspace_id: UUID) -> Workspace:
    """Get a workspace by ID."""
    workspace_dir = WORKSPACES_DIR / str(workspace_id)
    metadata_file = workspace_dir / "metadata.json"

    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    return Workspace.model_validate_json(metadata_file.read_text())


@app.post("/workspaces/{workspace_id}/threads/", response_model=Thread)
async def create_thread(workspace_id: UUID, thread_create: ThreadCreate) -> Thread:
    """Create a new thread in a workspace."""
    workspace_dir = WORKSPACES_DIR / str(workspace_id)
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    thread_id = uuid4()
    thread_dir = workspace_dir / "threads" / str(thread_id)
    thread_dir.mkdir(parents=True, exist_ok=True)

    # Create analysis and results directories
    (thread_dir / "analysis").mkdir(exist_ok=True)
    (thread_dir / "results").mkdir(exist_ok=True)

    # Save thread metadata
    thread_obj = Thread(id=thread_id, name=thread_create.name, workspace_id=workspace_id)
    (thread_dir / "metadata.json").write_text(thread_obj.model_dump_json())

    return thread_obj


@app.get("/workspaces/{workspace_id}/threads/", response_model=list[Thread])
async def list_threads(workspace_id: UUID) -> list[Thread]:
    """List all threads in a workspace."""
    workspace_dir = WORKSPACES_DIR / str(workspace_id)
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    threads_dir = workspace_dir / "threads"
    if not threads_dir.exists():
        return []

    threads: list[Thread] = []
    for thread_dir in threads_dir.iterdir():
        if thread_dir.is_dir():
            metadata_file = thread_dir / "metadata.json"
            if metadata_file.exists():
                thread = Thread.model_validate_json(metadata_file.read_text())
                threads.append(thread)

    return threads


@app.get("/workspaces/{workspace_id}/threads/{thread_id}", response_model=Thread)
async def get_thread(workspace_id: UUID, thread_id: UUID) -> Thread:
    """Get a thread by ID."""
    thread_dir = WORKSPACES_DIR / str(workspace_id) / "threads" / str(thread_id)
    metadata_file = thread_dir / "metadata.json"

    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Thread not found")

    return Thread.model_validate_json(metadata_file.read_text())


@app.post("/workspaces/{workspace_id}/threads/{thread_id}/messages/")
async def send_message(workspace_id: UUID, thread_id: UUID, message: Message) -> dict[str, Any]:
    """Send a message to a thread."""
    thread_dir = WORKSPACES_DIR / str(workspace_id) / "threads" / str(thread_id)
    if not thread_dir.exists():
        raise HTTPException(status_code=404, detail="Thread not found")

    # Create a background task for processing the message
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_message, str(thread_dir), message.content, thread_id)

    return {"status": "processing", "message": "Message received and processing started"}


async def process_message(thread_dir: str, message_content: str, thread_id: UUID):
    """Process a message and send the result through WebSocket."""
    try:
        logger.info(f"BACKGROUND TASK: Processing message for thread {thread_id}: {message_content[:100]}...")
        result = await run_thread(thread_dir=thread_dir, user_prompt=message_content)
        logger.info(f"BACKGROUND TASK: Thread {thread_id} processed. Result: {result[:100]}...")
        await manager.send_message(result, thread_id)
        logger.info(f"BACKGROUND TASK: Sent message to WebSocket for thread {thread_id}")
    except Exception as e:
        logger.error(f"BACKGROUND TASK: Error processing message for thread {thread_id}: {e}", exc_info=True)
        try:
            await manager.send_message(f"Error processing your request: {str(e)}", thread_id)
        except Exception as send_e:
            logger.error(
                f"BACKGROUND TASK: Failed to send error message to client for thread {thread_id}: {send_e}",
                exc_info=True,
            )


@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: UUID):
    """WebSocket endpoint for receiving messages from a thread."""
    await manager.connect(websocket, thread_id)
    try:
        while True:
            # Wait for any message from client to keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, thread_id)
