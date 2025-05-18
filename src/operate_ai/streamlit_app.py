"""
streamlit_app.py

Streamlit chat UI for OperateAI CFO Agent
Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh

# ─────────────────────────
# Config
# ─────────────────────────
BACKEND = "http://localhost:8000"


# ─────────────────────────
# Helpers
# ─────────────────────────
class Workspace(TypedDict):
    id: int
    name: str
    created_at: str
    updated_at: str


class Thread(TypedDict):
    id: int
    created_at: str
    updated_at: str


class LatestMsg(TypedDict, total=False):
    index: int
    role: str
    content: str | None
    output_type: str | None
    payload: Any
    created_at: str


def parse_payload(p: str | dict | None) -> dict[str, Any]:
    if p is None:
        return {}
    if isinstance(p, dict):
        return p
    try:
        return cast(dict[str, Any], json.loads(p))
    except Exception:
        return {}


def api(method: str, path: str, **kwargs) -> Any:
    url = f"{BACKEND}{path}"
    r = requests.request(method, url, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


# ─────────────────────────
# Session state
# ─────────────────────────
if "workspace" not in st.session_state:
    st.session_state.workspace: Workspace | None = None  # type: ignore[attr-defined]
if "thread" not in st.session_state:
    st.session_state.thread: Thread | None = None  # type: ignore[attr-defined]
if "chat" not in st.session_state:
    st.session_state.chat: list[LatestMsg] = []  # type: ignore[attr-defined]
if "latest_index" not in st.session_state:
    st.session_state.latest_index: int | None = None  # type: ignore[attr-defined]

# ─────────────────────────
# Sidebar ▸ workspaces
# ─────────────────────────
st.sidebar.header("OperateAI Workspaces")

# fetch & pick
workspaces: list[Workspace] = cast(list[Workspace], api("get", "/workspaces"))
names = {w["name"]: w for w in workspaces}
workspace_name = st.sidebar.selectbox(
    "Select workspace",
    ["—"] + list(names.keys()),
    index=0 if st.session_state.workspace is None else 1 + list(names).index(st.session_state.workspace["name"]),  # type: ignore[arg-type]
)

if workspace_name != "—":
    st.session_state.workspace = names[workspace_name]

# create new
with st.sidebar.expander("➕  Create new workspace"):
    new_name = st.text_input("Name", key="new_ws_name")
    if st.button("Create", key="create_ws_btn") and new_name:
        st.session_state.workspace = api("post", "/workspaces", json={"name": new_name})
        st.session_state.thread = None
        st.session_state.chat = []
        st.session_state.latest_index = None

# ─────────────────────────
# Sidebar ▸ threads
# ─────────────────────────
if st.session_state.workspace:
    ws_id = st.session_state.workspace["id"]
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Threads in **{st.session_state.workspace['name']}**")

    threads: list[Thread] = cast(list[Thread], api("get", f"/workspaces/{ws_id}/threads"))
    if not threads:
        st.sidebar.write("*(none yet)*")

    title_map = {f"#{t['id']} – {t['created_at'][:19]}": t for t in threads}
    thread_title = st.sidebar.selectbox(
        "Select thread",
        ["—"] + list(title_map.keys()),
        index=0
        if st.session_state.thread is None
        else 1
        + list(title_map).index(
            f"#{st.session_state.thread['id']} – {st.session_state.thread['created_at'][:19]}"
        ),
    )
    if thread_title != "—":
        st.session_state.thread = title_map[thread_title]

    # create thread
    with st.sidebar.expander("🆕  New thread"):
        init_prompt = st.text_area("Initial prompt (optional)", key="init_prompt")
        if st.button("Create thread", key="create_th_btn") and init_prompt is not None:
            thread = api(
                "post",
                f"/workspaces/{ws_id}/threads",
                json={"initial_prompt": init_prompt or None},
            )
            st.session_state.thread = thread
            st.session_state.chat = []
            st.session_state.latest_index = None
            st.rerun()

# ─────────────────────────
# Main chat area
# ─────────────────────────
st.title("OperateAI CFO Agent")

if not st.session_state.workspace or not st.session_state.thread:
    st.info("Choose or create a workspace ⟶ thread to start chatting.")
    st.stop()

ws_id = st.session_state.workspace["id"]
th_id = st.session_state.thread["id"]

# ── Poll latest message every 2 s ──────────────────────────────────────
autorefresh = st_autorefresh(interval=2000, key="poll")  # page reruns ⇒ chat updates


def fetch_latest() -> None:
    latest: LatestMsg | None = cast(
        LatestMsg | None,
        api("get", f"/workspaces/{ws_id}/threads/{th_id}/latest"),
    )
    if latest is None:
        return
    if st.session_state.latest_index != latest["index"]:
        st.session_state.chat.append(latest)
        st.session_state.latest_index = latest["index"]


fetch_latest()

# ── Show chat with actionable previews ───────────────────────────────────
for msg in st.session_state.chat:
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        ot = msg.get("output_type")
        pp = parse_payload(msg.get("payload"))

        # A. RunSQL → preview + CSV/SQL download
        if ot == "RunSQL":
            file_path: str | None = cast(str | None, pp.get("file_path"))
            if file_path and Path(file_path).exists():
                df = pd.read_csv(file_path)
                st.dataframe(df.head(25))
                st.download_button(
                    "⬇️  Full CSV",
                    data=df.to_csv(index=False).encode(),
                    file_name=Path(file_path).name,
                )
                sql_path = Path(file_path).with_suffix(".sql")
                if sql_path.exists():
                    st.code(sql_path.read_text(), language="sql")
            else:
                # fallback if files missing
                st.write(pp.get("preview") or msg.get("content", ""))

        # B. WriteSheetFromFile → workbook download + recap
        elif ot == "WriteSheetFromFile":
            wb_path: str | None = cast(str | None, pp.get("file_path"))
            sheet_name: str | None = cast(str | None, pp.get("sheet_name"))
            if wb_path and Path(wb_path).exists():
                with open(wb_path, "rb") as fh:
                    st.download_button("⬇️  Download workbook", fh, file_name=Path(wb_path).name)
                st.write(f"Appended **{sheet_name or 'sheet'}** to the workbook above.")
            else:
                # fallback
                st.write(msg.get("content", ""))

        # C. Plain chat
        else:
            st.write(msg.get("content", ""))

# ── Input area ────────────────────────────────────────────────────────
last_ot = st.session_state.chat[-1]["output_type"] if st.session_state.chat else None
is_actionable = last_ot in {"RunSQL", "WriteSheetFromFile"}


def send(content: str | None) -> None:
    api(
        "post",
        f"/workspaces/{ws_id}/threads/{th_id}/messages",
        json={"content": content},
    )


if is_actionable:
    col1, col2, col3 = st.columns([1, 1, 3])
    if col1.button("Continue ⏩"):
        send("")  # empty = continue
    if col2.button("Stop ✋"):
        st.stop()
    with col3:
        user_reply = st.text_input("Optional feedback", key="reply_box")
        if st.button("Send"):
            send(user_reply)
            st.session_state.chat.append(
                {
                    "index": (st.session_state.latest_index or 0) + 1,
                    "role": "user",
                    "content": user_reply,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
else:
    user_msg = st.chat_input("Write your message…")
    if user_msg is not None:
        send(user_msg)
        st.session_state.chat.append(
            {
                "index": (st.session_state.latest_index or -1) + 1,
                "role": "user",
                "content": user_msg,
                "created_at": datetime.utcnow().isoformat(),
            }
        )
