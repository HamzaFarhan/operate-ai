import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
import streamlit as st
import websocket
from loguru import logger

from operate_ai.api import Thread, Workspace

# API endpoint
API_URL = "http://localhost:8000"

# Setup page config
st.set_page_config(
    page_title="Operate AI CFO",
    page_icon="💼",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ws" not in st.session_state:
    st.session_state.ws = None
if "current_workspace" not in st.session_state:
    st.session_state.current_workspace = None
if "current_thread" not in st.session_state:
    st.session_state.current_thread = None

# Custom CSS
st.markdown(
    """
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .message-container {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .ai-message {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
</style>
""",
    unsafe_allow_html=True,
)


def on_ws_message(ws: websocket.WebSocket, message: str):
    """Handle incoming WebSocket messages"""
    logger.info(f"STREAMLIT WS: Received message: {message[:100]}...")
    st.session_state.messages.append({"role": "assistant", "content": message, "time": datetime.now().isoformat()})
    logger.info("STREAMLIT WS: Appended AI message to session_state.messages")
    try:
        st.rerun()
        logger.info("STREAMLIT WS: Called st.rerun() after AI message.")
    except Exception as e:  # Add try-except for st.rerun() if it's called from a non-main thread issue (though websocket-client runs callbacks in their own threads)
        logger.error(f"STREAMLIT WS: Error during st.rerun(): {e}", exc_info=True)


def connect_websocket(thread_id: UUID):
    """Connect to the WebSocket for the current thread"""
    if st.session_state.ws:
        logger.info("STREAMLIT WS: Closing existing WebSocket connection.")
        try:
            st.session_state.ws.close()
        except Exception as e:
            logger.error(f"STREAMLIT WS: Error closing existing WebSocket: {e}", exc_info=True)

    ws_url = f"ws://localhost:8000/ws/{thread_id}"
    logger.info(f"STREAMLIT WS: Connecting to WebSocket: {ws_url}")

    def on_open(ws_app):
        logger.info(f"STREAMLIT WS: WebSocket connection opened for thread {thread_id}")

    def on_error(ws_app, error):
        logger.error(f"STREAMLIT WS: WebSocket error for thread {thread_id}: {error}", exc_info=True)

    def on_close(ws_app, close_status_code, close_msg):
        logger.info(
            f"STREAMLIT WS: WebSocket connection closed for thread {thread_id}. Status: {close_status_code}, Msg: {close_msg}"
        )

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_ws_message,
        on_error=on_error,
        on_close=on_close,
    )

    # Start WebSocket connection in a separate thread
    import threading

    wst = threading.Thread(target=ws.run_forever)  # type: ignore
    wst.daemon = True
    wst.start()

    st.session_state.ws = ws
    return ws


async def get_workspaces() -> list[Workspace]:
    """Get list of workspaces from API"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/workspaces/")
        return (
            [Workspace.model_validate(workspace) for workspace in response.json()]
            if response.status_code == 200
            else []
        )


async def create_workspace(name: str) -> Workspace | None:
    """Create a new workspace"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/workspaces/", json={"name": name})
        return Workspace.model_validate(response.json()) if response.status_code == 200 else None


async def get_threads(workspace_id: UUID) -> list[Thread]:
    """Get list of threads for a workspace"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/workspaces/{workspace_id}/threads/")
        return [Thread.model_validate(thread) for thread in response.json()] if response.status_code == 200 else []


async def create_thread(workspace_id: UUID, name: str) -> Thread | None:
    """Create a new thread in a workspace"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/workspaces/{workspace_id}/threads/", json={"name": name})
        return Thread.model_validate(response.json()) if response.status_code == 200 else None


async def send_message(workspace_id: UUID, thread_id: UUID, content: str) -> dict[str, Any] | None:
    """Send a message to a thread"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/workspaces/{workspace_id}/threads/{thread_id}/messages/", json={"content": content}
        )
        return response.json() if response.status_code == 200 else None


def workspace_selector() -> Workspace | None:
    """Workspace selection and creation UI"""
    st.sidebar.title("Workspaces")

    # Get workspaces
    workspaces = asyncio.run(get_workspaces())

    # Create new workspace
    with st.sidebar.expander("Create New Workspace"):
        workspace_name = st.text_input("Workspace Name", key="new_workspace_name")
        if st.button("Create Workspace") and workspace_name:
            new_workspace = asyncio.run(create_workspace(workspace_name))
            if new_workspace:
                st.success(f"Workspace '{workspace_name}' created!")
                st.rerun()

    # Select workspace
    if workspaces:
        workspace_options = {w.name: w for w in workspaces}
        selected_workspace_name = st.sidebar.selectbox(
            "Select Workspace",
            options=list(workspace_options.keys()),
            index=0
            if st.session_state.current_workspace is None
            else list(workspace_options.keys()).index(
                next(
                    (w.name for w in workspaces if w.id == st.session_state.current_workspace.id),
                    list(workspace_options.keys())[0],
                )
            ),
        )

        if selected_workspace_name:
            st.session_state.current_workspace = workspace_options[selected_workspace_name]
            return st.session_state.current_workspace

    return None


def thread_selector(workspace_id: UUID) -> Thread | None:
    """Thread selection and creation UI"""
    st.sidebar.title("Threads")

    # Get threads
    threads = asyncio.run(get_threads(workspace_id))

    # Create new thread
    with st.sidebar.expander("Create New Thread"):
        thread_name = st.text_input("Thread Name", key="new_thread_name")
        if st.button("Create Thread") and thread_name:
            new_thread = asyncio.run(create_thread(workspace_id, thread_name))
            if new_thread:
                st.success(f"Thread '{thread_name}' created!")
                st.session_state.messages = []
                st.session_state.current_thread = new_thread
                connect_websocket(new_thread.id)
                st.rerun()

    # Select thread
    if threads:
        thread_options = {t.name: t for t in threads}
        selected_thread_name = st.sidebar.selectbox(
            "Select Thread",
            options=list(thread_options.keys()),
            index=0
            if st.session_state.current_thread is None
            else list(thread_options.keys()).index(
                next(
                    (t.name for t in threads if t.id == st.session_state.current_thread.id),
                    list(thread_options.keys())[0],
                )
            ),
        )

        if selected_thread_name:
            selected_thread = thread_options[selected_thread_name]
            if st.session_state.current_thread is None or selected_thread.id != st.session_state.current_thread.id:
                st.session_state.messages = []
                st.session_state.current_thread = selected_thread
                connect_websocket(selected_thread.id)
            return selected_thread

    return None


def display_chat():
    """Display chat messages"""
    for message in st.session_state.messages:
        with st.container():
            role_class = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(
                f"""
            <div class="message-container {role_class}">
                <p><strong>{"You" if message["role"] == "user" else "AI"}</strong> ({message["time"].split("T")[1][:8]})</p>
                <p>{message["content"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def main():
    """Main Streamlit app"""
    st.title("Operate AI CFO Assistant")

    # Sidebar for workspace/thread selection
    workspace = workspace_selector()

    if workspace:
        thread = thread_selector(workspace.id)

        if thread:
            # Main chat area
            st.header(f"Thread: {thread.name}")

            # Display chat messages
            for message_item in st.session_state.messages:  # Renamed to message_item to avoid conflict
                with st.chat_message(message_item["role"]):
                    st.markdown(message_item["content"])

            # Message input
            if user_prompt := st.chat_input("Ask something..."):  # Renamed to user_prompt
                # Add user message to history
                st.session_state.messages.append(
                    {"role": "user", "content": user_prompt, "time": datetime.now().isoformat()}
                )
                # Display the user's message immediately by rerunning the script
                # The API call will happen in this current run after the rerun displays the user message

                # Send message to API (this happens in the current script run)
                logger.info(f"STREAMLIT APP: Sending message to API: {user_prompt[:100]}...")
                api_response = asyncio.run(send_message(workspace.id, thread.id, user_prompt))
                if api_response:
                    logger.info(f"STREAMLIT APP: API response from send_message: {api_response}")
                else:
                    logger.error("STREAMLIT APP: No response or error from send_message API call.")

                # Rerun to display the user's message.
                # The AI response will come via WebSocket and trigger its own rerun.
                st.rerun()
        else:
            st.info("Create or select a thread to start chatting.")
    else:
        st.info("Create or select a workspace to get started.")


if __name__ == "__main__":
    main()
