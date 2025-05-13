from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="AI CFO API", version="1.0.0")

# ─── In-memory fake DB ──────────────────────────────────────────────────────────
DB: dict[str, dict[str, dict[str, Any]]] = {
    "workspaces": {},
    "threads": {},
    "messages": {},
    "tasks": {},
    "results": {},
}


# ─── Models ─────────────────────────────────────────────────────────────────────
class WorkspaceIn(BaseModel):
    name: str = Field(..., min_length=1)


class WorkspaceOut(WorkspaceIn):
    id: UUID
    created_at: datetime


class ThreadIn(BaseModel):
    title: str


class ThreadOut(ThreadIn):
    id: UUID
    workspace_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime


class MessageIn(BaseModel):
    role: str
    content: str


class MessageOut(MessageIn):
    id: UUID
    thread_id: UUID
    created_at: datetime


class TaskIn(BaseModel):
    name: str
    payload: dict[str, Any] | None = None


class TaskOut(TaskIn):
    id: UUID
    thread_id: UUID
    state: str
    created_at: datetime


class ResultIn(BaseModel):
    filename: str
    path: str
    mime_type: str


class ResultOut(ResultIn):
    id: UUID
    task_id: UUID
    created_at: datetime


OutModel = WorkspaceOut | ThreadOut | MessageOut | TaskOut | ResultOut


# ─── Helpers ────────────────────────────────────────────────────────────────────
def _store(kind: str, obj: OutModel) -> None:
    DB[kind][str(obj.id)] = obj.model_dump()


def _get(kind: str, obj_id: UUID) -> dict[str, Any]:
    try:
        return DB[kind][str(obj_id)]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{kind[:-1].title()} not found")


def _delete(kind: str, obj_id: UUID) -> None:
    try:
        del DB[kind][str(obj_id)]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{kind[:-1].title()} not found")


# ─── Workspace endpoints ────────────────────────────────────────────────────────
@app.post("/workspaces", response_model=WorkspaceOut, status_code=201)
def create_workspace(ws: WorkspaceIn):
    obj = WorkspaceOut(id=uuid4(), created_at=datetime.now(), **ws.model_dump())
    _store("workspaces", obj)
    return obj


@app.get("/workspaces", response_model=list[WorkspaceOut])
def list_workspaces():
    return list(DB["workspaces"].values())


@app.get("/workspaces/{ws_id}", response_model=WorkspaceOut)
def read_workspace(ws_id: UUID):
    return _get("workspaces", ws_id)


@app.delete("/workspaces/{ws_id}", status_code=204)
def delete_workspace(ws_id: UUID):
    _delete("workspaces", ws_id)


# ─── Thread endpoints ───────────────────────────────────────────────────────────
@app.post("/workspaces/{ws_id}/threads", response_model=ThreadOut, status_code=201)
def create_thread(ws_id: UUID, th: ThreadIn):
    _get("workspaces", ws_id)
    now = datetime.now()
    obj = ThreadOut(
        id=uuid4(), workspace_id=ws_id, status="open", created_at=now, updated_at=now, **th.model_dump()
    )
    _store("threads", obj)
    return obj


@app.get("/workspaces/{ws_id}/threads", response_model=list[ThreadOut])
def list_threads(ws_id: UUID):
    _get("workspaces", ws_id)
    return [t for t in DB["threads"].values() if t["workspace_id"] == ws_id]


@app.get("/workspaces/{ws_id}/threads/{th_id}", response_model=ThreadOut)
def read_thread(ws_id: UUID, th_id: UUID):
    thread = _get("threads", th_id)
    if thread["workspace_id"] != ws_id:
        raise HTTPException(404, "Thread not in this workspace")
    return thread


@app.patch("/workspaces/{ws_id}/threads/{th_id}", response_model=ThreadOut)
def update_thread(ws_id: UUID, th_id: UUID, th_update: ThreadIn):
    thread = _get("threads", th_id)
    if thread["workspace_id"] != ws_id:
        raise HTTPException(404, "Thread not in this workspace")
    thread.update(th_update.model_dump())
    thread["updated_at"] = datetime.now()
    DB["threads"][str(th_id)] = thread
    return thread


@app.delete("/workspaces/{ws_id}/threads/{th_id}", status_code=204)
def delete_thread(ws_id: UUID, th_id: UUID):
    thread = _get("threads", th_id)
    if thread["workspace_id"] != ws_id:
        raise HTTPException(404, "Thread not in this workspace")
    _delete("threads", th_id)


# ─── Message endpoints ──────────────────────────────────────────────────────────
@app.post("/threads/{th_id}/messages", response_model=MessageOut, status_code=201)
def add_message(th_id: UUID, msg: MessageIn):
    _get("threads", th_id)
    obj = MessageOut(id=uuid4(), thread_id=th_id, created_at=datetime.now(), **msg.model_dump())
    _store("messages", obj)
    return obj


@app.get("/threads/{th_id}/messages", response_model=list[MessageOut])
def list_messages(th_id: UUID):
    _get("threads", th_id)
    return [m for m in DB["messages"].values() if m["thread_id"] == th_id]


# ─── Task endpoints ─────────────────────────────────────────────────────────────
@app.post("/threads/{th_id}/tasks", response_model=TaskOut, status_code=201)
def create_task(th_id: UUID, task: TaskIn):
    _get("threads", th_id)
    now = datetime.now()
    obj = TaskOut(id=uuid4(), thread_id=th_id, state="queued", created_at=now, **task.model_dump())
    _store("tasks", obj)
    return obj


@app.get("/threads/{th_id}/tasks", response_model=list[TaskOut])
def list_tasks(th_id: UUID):
    _get("threads", th_id)
    return [t for t in DB["tasks"].values() if t["thread_id"] == th_id]


@app.get("/tasks/{task_id}", response_model=TaskOut)
def read_task(task_id: UUID):
    return _get("tasks", task_id)


@app.patch("/tasks/{task_id}", response_model=TaskOut)
def update_task(task_id: UUID, task_update: TaskIn):
    task = _get("tasks", task_id)
    task.update(task_update.model_dump())
    DB["tasks"][str(task_id)] = task
    return task


@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: UUID):
    _delete("tasks", task_id)


# ─── Result endpoints ───────────────────────────────────────────────────────────
@app.post("/tasks/{task_id}/results", response_model=ResultOut, status_code=201)
def create_result(task_id: UUID, res: ResultIn):
    _get("tasks", task_id)
    now = datetime.now()
    obj = ResultOut(id=uuid4(), task_id=task_id, created_at=now, **res.model_dump())
    _store("results", obj)
    return obj


@app.get("/tasks/{task_id}/results", response_model=list[ResultOut])
def list_results(task_id: UUID):
    _get("tasks", task_id)
    return [r for r in DB["results"].values() if r["task_id"] == task_id]


@app.get("/results/{res_id}", response_model=ResultOut)
def read_result(res_id: UUID):
    return _get("results", res_id)


@app.delete("/results/{res_id}", status_code=204)
def delete_result(res_id: UUID):
    _delete("results", res_id)
