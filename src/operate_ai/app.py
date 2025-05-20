import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd
import streamlit as st
from loguru import logger
from pydantic import ValidationError

from operate_ai.api import ChatMessage, ThreadInfo, WorkspaceInfo
from operate_ai.cfo_graph import RunSQLResult, WriteSheetResult

# API configuration
API_URL = "http://localhost:8000"
TIMEOUT = 60
COUNT_DOWN_SECONDS = 30
CONTINUE_MESSAGE = "Looks good, please continue."

st.set_page_config(layout="wide")


# Helper functions for API calls
async def get_workspaces() -> list[WorkspaceInfo]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/workspaces/")
        if response.status_code == 200:
            return [WorkspaceInfo(**workspace) for workspace in response.json()]
        return []


async def create_workspace(name: str) -> WorkspaceInfo | None:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/workspaces/", json={"name": name})
        if response.status_code == 200:
            return WorkspaceInfo(**response.json())
        st.error(f"Failed to create workspace: {response.text}")
        return None


async def get_threads(workspace_id: str) -> list[ThreadInfo]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/workspaces/{workspace_id}/threads/")
        if response.status_code == 200:
            return [ThreadInfo(**thread) for thread in response.json()]
        return []


async def create_thread(workspace_id: str, name: str, prev_state_path: str | None = None) -> ThreadInfo | None:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/workspaces/{workspace_id}/threads/",
            json={"name": name, "prev_state_path": prev_state_path},
        )
        if response.status_code == 200:
            return ThreadInfo(**response.json())
        st.error(f"Failed to create thread: {response.text}")
        return None


async def get_messages(workspace_id: str, thread_id: str) -> list[ChatMessage]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/workspaces/{workspace_id}/threads/{thread_id}/messages/")
        if response.status_code == 200:
            return [ChatMessage(**message) for message in response.json()]
        return []


async def send_message(
    workspace_id: str, thread_id: str, content: str, prev_state_path: str | None = None
) -> ChatMessage | None:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{API_URL}/workspaces/{workspace_id}/threads/{thread_id}/messages/",
            json={"content": content, "prev_state_path": prev_state_path},
        )
        if response.status_code == 200:
            return ChatMessage(**response.json())
        st.error(f"Failed to send message: {response.text}")
        return None


async def upload_csv_to_workspace(workspace_id: str, file: io.BytesIO):
    # Get the workspace directory
    workspace_dir = Path("workspaces") / workspace_id / "data"
    workspace_dir.mkdir(exist_ok=True, parents=True)

    # Save the uploaded file to the workspace data directory
    file_path = workspace_dir / file.name
    file_path.write_bytes(file.getbuffer())


def parse_response(resp: Any) -> RunSQLResult | WriteSheetResult | str:
    # If already correct type, return as is
    if isinstance(resp, (RunSQLResult, WriteSheetResult)):
        return resp
    # If dict, try to parse
    if isinstance(resp, dict):
        for cls in (RunSQLResult, WriteSheetResult):
            try:
                return cls.model_validate(resp)
            except ValidationError:
                continue
    # If string, try to parse as JSON then as above
    if isinstance(resp, str):
        try:
            data = json.loads(resp)
            for cls in (RunSQLResult, WriteSheetResult):
                try:
                    return cls.model_validate(data)
                except ValidationError:
                    continue
        except Exception:
            pass
    return str(resp)  # type: ignore


async def main():
    # Initialize session state
    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "thread_name" not in st.session_state:
        st.session_state.thread_name = None
    if "show_countdown" not in st.session_state:
        st.session_state.show_countdown = False
    if "cancel_countdown" not in st.session_state:
        st.session_state.cancel_countdown = False
    if "countdown" not in st.session_state:
        st.session_state.countdown = COUNT_DOWN_SECONDS

    st.title("Operate AI")

    # --- Sidebar for workspace/thread/data ---
    with st.sidebar:
        st.header("Workspaces")

        # Create a new workspace
        with st.expander("Create New Workspace"):
            workspace_name = st.text_input("Workspace Name", key="new_workspace_name")
            if st.button("Create Workspace"):
                if workspace_name:
                    new_workspace = await create_workspace(workspace_name)
                    if new_workspace:
                        st.session_state.workspace_id = new_workspace.id
                        st.success(f"Created workspace: {workspace_name}")
                        st.rerun()
                else:
                    st.error("Please enter a workspace name")

        # List and select workspaces
        workspaces = await get_workspaces()
        workspace_options = {w.name: w.id for w in workspaces}
        selected_workspace_name = st.selectbox(
            "Select Workspace", options=list(workspace_options.keys()), index=0 if workspace_options else None
        )

        if selected_workspace_name:
            selected_workspace_id = workspace_options[selected_workspace_name]
            st.session_state.workspace_id = selected_workspace_id

            # Upload CSV to workspace
            with st.expander("Upload CSV Data"):
                uploaded_files = st.file_uploader("Choose CSV file(s)", type="csv", accept_multiple_files=True)
                if uploaded_files:
                    if st.button("Upload to Workspace"):
                        for uploaded_file in uploaded_files:
                            await upload_csv_to_workspace(selected_workspace_id, uploaded_file)
                        st.success("Files uploaded!")

            # Check if workspace has data files
            workspace_data_dir = Path("workspaces") / selected_workspace_id / "data"
            has_data = any(workspace_data_dir.glob("*.csv"))

            st.header("Threads")

            # Create a new thread
            with st.expander("Create New Thread"):
                if not has_data:
                    st.warning("Upload at least one CSV file before creating a thread.")
                thread_name = st.text_input("Thread Name", key="new_thread_name", disabled=not has_data)
                if st.button("Create Thread", disabled=not has_data):
                    if thread_name:
                        new_thread = await create_thread(selected_workspace_id, thread_name)
                        if new_thread:
                            st.session_state.thread_id = new_thread.id
                            st.session_state.thread_name = thread_name
                            st.success(f"Created thread: {thread_name}")
                            st.rerun()
                    else:
                        st.error("Please enter a thread name")

            # List and select threads
            threads = await get_threads(selected_workspace_id)
            thread_options = {t.name: t.id for t in threads}

            if thread_options:
                selected_thread_name = st.selectbox(
                    "Select Thread", options=list(thread_options.keys()), index=len(thread_options) - 1
                )

                if selected_thread_name:
                    selected_thread_id = thread_options[selected_thread_name]
                    st.session_state.thread_id = selected_thread_id
                    st.session_state.thread_name = selected_thread_name

    # --- Main chat area ---
    if st.session_state.workspace_id and st.session_state.thread_id:
        workspace_id = st.session_state.workspace_id
        thread_id = st.session_state.thread_id
        thread_name = st.session_state.thread_name
        messages = await get_messages(workspace_id, thread_id)

        for i, msg in enumerate(messages):
            if msg.role == "user":
                with st.chat_message("user"):
                    st.markdown(msg.content)
            else:
                with st.chat_message("assistant"):
                    resp = parse_response(msg.content)
                    if isinstance(resp, RunSQLResult):
                        st.session_state.show_countdown = True
                        st.markdown("Ran an SQL query, please review")
                        with st.expander(resp.purpose or "Show Progress"):
                            tabs = st.tabs(["CSV Preview", "SQL Command"])
                            with tabs[0]:
                                csv_path = resp.csv_path
                                try:
                                    df = pd.read_csv(csv_path)  # type: ignore
                                    st.dataframe(df)  # type: ignore
                                except Exception:
                                    st.error("Could not read CSV file.")
                            with tabs[1]:
                                sql_path = resp.sql_path
                                try:
                                    sql_text = Path(sql_path).read_text()
                                except Exception:
                                    sql_text = "(Could not read SQL file)"
                                st.code(sql_text, language="sql")

                    elif isinstance(resp, WriteSheetResult):
                        st.session_state.show_countdown = True
                        st.markdown("Wrote the results to a Workbook, please review")
                        with st.expander("Show Results"):
                            excel_path = resp.file_path
                            try:
                                excel_data: dict[str, pd.DataFrame] = pd.read_excel(excel_path, sheet_name=None)  # type: ignore
                                sheet_names = list(excel_data.keys())
                                sheet_tabs = st.tabs(sheet_names)
                                for tab, sheet_name in zip(sheet_tabs, sheet_names):
                                    with tab:
                                        st.dataframe(excel_data[sheet_name])  # type: ignore
                                file_bytes = Path(excel_path).read_bytes()
                                st.download_button(
                                    label="Download Excel file",
                                    data=file_bytes,
                                    file_name=Path(excel_path).name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"download_excel_{thread_id}_{i}",
                                )
                            except Exception:
                                st.error("Could not read Excel file.")
                    else:
                        st.session_state.show_countdown = False
                        st.markdown(resp)
                    if st.button("New Thread From Here", key=f"new_thread_from_here_{thread_id}_{i}"):
                        logger.info(f"Creating new thread from here: {msg.state_path}")
                        new_thread_name = f"{thread_name}_{uuid4()}"
                        new_thread = await create_thread(
                            workspace_id, new_thread_name, prev_state_path=msg.state_path
                        )
                        if new_thread:
                            st.session_state.thread_id = new_thread.id
                            st.session_state.thread_name = new_thread_name
                            st.success(f"Created thread: {new_thread_name}")
                            st.rerun()
                        else:
                            st.error("Failed to create thread")

        if st.session_state.show_countdown and not st.session_state.cancel_countdown:
            if "countdown" not in st.session_state:
                st.session_state.countdown = COUNT_DOWN_SECONDS
            if st.session_state.countdown > 0:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(st.session_state.countdown / COUNT_DOWN_SECONDS)
                    st.text(f"Continuing in {st.session_state.countdown} seconds...")
                with col2:
                    if st.button("Cancel Auto-Continue"):
                        st.session_state.show_countdown = False
                        st.session_state.countdown = COUNT_DOWN_SECONDS
                        st.session_state.cancel_countdown = True
                        st.rerun()

            # Decrease countdown
            st.session_state.countdown -= 1

            # If countdown reaches zero, send message
            if st.session_state.countdown < 0:
                st.session_state.show_countdown = False
                st.session_state.countdown = COUNT_DOWN_SECONDS
                response = await send_message(workspace_id, thread_id, CONTINUE_MESSAGE)
                if response:
                    st.rerun()
            else:
                # Use st.rerun to update the UI with new countdown value
                time.sleep(1)  # Small delay to prevent too rapid refreshes
                st.rerun()

        if user_message := st.chat_input("Your message..."):
            with st.chat_message("user"):
                st.markdown(user_message)
            with st.spinner("Processing..."):
                response = await send_message(workspace_id, thread_id, user_message)
                if response:
                    st.session_state.cancel_countdown = False
                    st.rerun()
    else:
        st.info("Please select or create a workspace and thread from the sidebar.")


if __name__ == "__main__":
    asyncio.run(main())
