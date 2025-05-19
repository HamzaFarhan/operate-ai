import asyncio
import io
from pathlib import Path

import httpx
import streamlit as st

from operate_ai.api import MessageResponse, ThreadInfo, WorkspaceInfo

# API configuration
API_URL = "http://localhost:8000"
TIMEOUT = 60


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


async def create_thread(workspace_id: str, name: str) -> ThreadInfo | None:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/workspaces/{workspace_id}/threads/", json={"name": name})
        if response.status_code == 200:
            return ThreadInfo(**response.json())
        st.error(f"Failed to create thread: {response.text}")
        return None


async def get_messages(workspace_id: str, thread_id: str) -> list[MessageResponse]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/workspaces/{workspace_id}/threads/{thread_id}/messages/")
        if response.status_code == 200:
            return [MessageResponse(**message) for message in response.json()]
        return []


async def send_message(workspace_id: str, thread_id: str, content: str) -> MessageResponse | None:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{API_URL}/workspaces/{workspace_id}/threads/{thread_id}/messages/", json={"content": content}
        )
        if response.status_code == 200:
            return MessageResponse(**response.json())
        st.error(f"Failed to send message: {response.text}")
        return None


async def upload_csv_to_workspace(workspace_id: str, file: io.BytesIO):
    # Get the workspace directory
    workspace_dir = Path("workspaces") / workspace_id / "data"
    workspace_dir.mkdir(exist_ok=True, parents=True)

    # Save the uploaded file to the workspace data directory
    file_path = workspace_dir / file.name
    file_path.write_bytes(file.getbuffer())


async def main():
    # Initialize session state
    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    # App title
    st.title("Operate AI")

    # Sidebar for workspace and thread selection
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
                            st.success(f"Created thread: {thread_name}")
                            st.rerun()
                    else:
                        st.error("Please enter a thread name")

            # List and select threads
            threads = await get_threads(selected_workspace_id)
            thread_options = {t.name: t.id for t in threads}

            if thread_options:
                selected_thread_name = st.selectbox("Select Thread", options=list(thread_options.keys()), index=0)

                if selected_thread_name:
                    selected_thread_id = thread_options[selected_thread_name]
                    st.session_state.thread_id = selected_thread_id

    # Main content area
    if st.session_state.workspace_id and st.session_state.thread_id:
        workspace_id = st.session_state.workspace_id
        thread_id = st.session_state.thread_id

        # Display conversation history using chat_message
        messages = await get_messages(workspace_id, thread_id)

        for msg in messages:
            with st.chat_message("user"):
                st.markdown(msg.content)
            with st.chat_message("assistant"):
                st.markdown(msg.response)

        # Input for new message using chat_input
        if user_message := st.chat_input("Your message..."):
            # Display the user's message immediately
            with st.chat_message("user"):
                st.markdown(user_message)
            with st.spinner("Processing..."):
                response = await send_message(workspace_id, thread_id, user_message)
                if response:
                    st.rerun()
    else:
        st.info("Please select or create a workspace and thread from the sidebar.")


if __name__ == "__main__":
    asyncio.run(main())
