import asyncio
import io
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

from operate_ai.api import ChatMessage, ThreadInfo, WorkspaceInfo
from operate_ai.cfo_graph import RunSQLResult, WriteDataToExcelResult
from operate_ai.memory_tools import KnowledgeGraph

load_dotenv()

# API configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"
WORKSPACES_DIR = Path(os.getenv("WORKSPACES_DIR", "workspaces"))
WORKSPACES_DIR.mkdir(exist_ok=True)
TIMEOUT = 900
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
    workspace_dir = (WORKSPACES_DIR / workspace_id) / "data"
    workspace_dir.mkdir(exist_ok=True, parents=True)
    # logger.info(f"Created workspace directory at {workspace_dir}")

    # Save the uploaded file to the workspace data directory
    file_path = workspace_dir / file.name
    file_path.write_bytes(file.getbuffer())


def parse_response(resp: Any) -> RunSQLResult | WriteDataToExcelResult | str:
    # If already correct type, return as is
    if isinstance(resp, (RunSQLResult, WriteDataToExcelResult)):
        return resp
    # If dict, try to parse
    if isinstance(resp, dict):
        for cls in (RunSQLResult, WriteDataToExcelResult):
            try:
                return cls.model_validate(resp)
            except ValidationError:
                continue
    # If string, try to parse as JSON then as above
    if isinstance(resp, str):
        try:
            data = json.loads(resp)
            for cls in (RunSQLResult, WriteDataToExcelResult):
                try:
                    return cls.model_validate(data)
                except ValidationError:
                    continue
        except Exception:
            pass
    return str(resp)  # type: ignore


def create_graph_visualization(kg: KnowledgeGraph):
    """Create a plotly visualization of the knowledge graph."""
    if not kg.entities:
        return None

    try:
        # Convert to NetworkX graph
        G = kg.to_networkx()

        # Get node positions using spring layout
        import networkx as nx

        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

        # Separate entities and observations for different styling
        entity_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "entity"]
        observation_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "observation"]

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Add edge info for hover
            edge_type = edge[2].get("edge_type", "unknown")
            if edge_type == "relation":
                relation_type = edge[2].get("relation_type", "unknown")
                edge_info.append(f"{edge[0]} → {edge[1]} ({relation_type})")
            else:
                edge_info.append(f"{edge[0]} → {edge[1]} ({edge_type})")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo="none", mode="lines"
        )

        # Create entity node trace
        entity_x = [pos[node][0] for node in entity_nodes]
        entity_y = [pos[node][1] for node in entity_nodes]
        entity_text = []
        entity_hover = []

        for node in entity_nodes:
            node_data = G.nodes[node]
            entity_text.append(node_data["name"])

            # Create hover text with entity info
            hover_text = f"<b>{node_data['name']}</b><br>"
            hover_text += f"Type: {node_data['entity_type']}<br>"

            # Add observations from the original knowledge graph
            if node in kg.entities:
                entity = kg.entities[node]
                if entity.observations:
                    hover_text += f"Observations: {len(entity.observations)}<br>"
                    # Show first few observations
                    for i, obs in enumerate(entity.observations[:3]):
                        hover_text += f"• {obs}<br>"
                    if len(entity.observations) > 3:
                        hover_text += f"... and {len(entity.observations) - 3} more"

            entity_hover.append(hover_text)

        entity_trace = go.Scatter(
            x=entity_x,
            y=entity_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=entity_hover,
            text=entity_text,
            textposition="middle center",
            marker=dict(size=20, color="lightblue", line=dict(width=2, color="darkblue")),
        )

        # Create observation node trace
        obs_x = [pos[node][0] for node in observation_nodes]
        obs_y = [pos[node][1] for node in observation_nodes]
        obs_text = []
        obs_hover = []

        for node in observation_nodes:
            node_data = G.nodes[node]
            obs_text.append("📝")  # Small icon for observations
            obs_hover.append(f"<b>Observation</b><br>{node_data['observation']}")

        obs_trace = go.Scatter(
            x=obs_x,
            y=obs_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=obs_hover,
            text=obs_text,
            textposition="middle center",
            marker=dict(size=10, color="lightgreen", line=dict(width=1, color="darkgreen")),
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, entity_trace, obs_trace])

        fig.update_layout(
            title=dict(text="Knowledge Graph Visualization", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Entities are shown in blue, observations in green. Hover for details.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="gray", size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating graph visualization: {e}")
        return None


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
    if "force_continue" not in st.session_state:
        st.session_state.force_continue = False
    if "countdown" not in st.session_state:
        st.session_state.countdown = COUNT_DOWN_SECONDS

    st.title("AI CFO")

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
            workspace_data_dir = (WORKSPACES_DIR / selected_workspace_id) / "data"
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
        # Persistent Memory tab across all threads, if memory.json exists
        memory_path = WORKSPACES_DIR / workspace_id / "memory.json"
        if memory_path.exists():
            chat_tab, memory_tab = st.tabs(["Chat", "Memory"])
            with memory_tab:
                st.subheader("Workspace Memory")
                try:
                    mem = json.loads(memory_path.read_text())

                    # Try to load as KnowledgeGraph for visualization
                    try:
                        kg = KnowledgeGraph.model_validate(mem)

                        # Create tabs for different views
                        graph_tab, json_tab = st.tabs(["Graph View", "JSON View"])

                        with graph_tab:
                            if kg.entities:
                                fig = create_graph_visualization(kg)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("Could not create graph visualization")

                                # Show some stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Entities", len(kg.entities))
                                with col2:
                                    st.metric("Relations", len(kg.relations))
                                with col3:
                                    total_observations = sum(
                                        len(entity.observations) for entity in kg.entities.values()
                                    )
                                    st.metric("Observations", total_observations)
                            else:
                                st.info("No entities in memory graph yet")

                        with json_tab:
                            # Initialize edit mode state
                            edit_key = f"edit_memory_{workspace_id}"
                            if edit_key not in st.session_state:
                                st.session_state[edit_key] = False

                            if not st.session_state[edit_key]:
                                # Display mode - show formatted JSON
                                st.json(mem)

                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    if st.button("Edit Memory", type="secondary"):
                                        st.session_state[edit_key] = True
                                        st.rerun()
                            else:
                                # Edit mode - show text area
                                edited_json = st.text_area(
                                    "Edit Memory (JSON format)",
                                    value=json.dumps(mem, indent=2),
                                    height=800,
                                    key=f"memory_editor_{workspace_id}",
                                )

                                col1, col2, col3 = st.columns([0.05, 0.05, 0.9])
                                with col1:
                                    if st.button("Save", type="primary"):
                                        try:
                                            # Validate JSON format
                                            parsed_json = json.loads(edited_json)
                                            # Write back to file
                                            memory_path.write_text(json.dumps(parsed_json, indent=2))
                                            st.session_state[edit_key] = False
                                            st.success("Memory saved successfully!")
                                            st.rerun()
                                        except json.JSONDecodeError as e:
                                            st.error(f"Invalid JSON format: {e}")
                                        except Exception as e:
                                            st.error(f"Error saving memory: {e}")

                                with col2:
                                    if st.button("Cancel", type="secondary"):
                                        st.session_state[edit_key] = False
                                        st.rerun()

                                with col3:
                                    st.caption("Make sure to use valid JSON format before saving")

                    except Exception:
                        # Fallback to JSON view if not a valid KnowledgeGraph
                        st.warning("Memory data is not in KnowledgeGraph format, showing JSON view")
                        st.json(mem)

                except Exception as e:
                    st.error(f"Could not load memory.json: {e}")
            container = chat_tab
        else:
            container = st.container()

        # All of the existing message‐rendering & input logic goes under this block:
        with container:
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
                            sql_header = "Ran an SQL query, please review"
                            if resp.is_task_result:
                                sql_header = "Task completed"
                                st.session_state.show_countdown = False
                                st.session_state.countdown = COUNT_DOWN_SECONDS
                                st.session_state.cancel_countdown = True
                            st.markdown(sql_header)
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

                        elif isinstance(resp, WriteDataToExcelResult):
                            excel_path = resp.file_path
                            try:
                                excel_data: dict[str, pd.DataFrame] = pd.read_excel(  # type: ignore
                                    excel_path, sheet_name=None
                                )
                                file_bytes = Path(excel_path).read_bytes()
                            except Exception as e:
                                logger.error(f"Error reading Excel file: {e}")
                                continue
                            st.session_state.show_countdown = True
                            st.markdown("Wrote the results to a Workbook, please review")
                            with st.expander("Show Results"):
                                sheet_names = list(excel_data.keys())
                                sheet_tabs = st.tabs(sheet_names)
                                for tab, sheet_name in zip(sheet_tabs, sheet_names):
                                    with tab:
                                        try:
                                            st.dataframe(excel_data[sheet_name])  # type: ignore
                                        except Exception:
                                            st.error(f"Could not read sheet: {sheet_name}")
                            st.download_button(
                                label="Download Excel file",
                                data=file_bytes,
                                file_name=Path(excel_path).name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"download_excel_{thread_id}_{i}",
                            )
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
                if st.session_state.countdown > 0 and not st.session_state.force_continue:
                    col1, col2, col3 = st.columns([0.8, 0.07, 0.13])
                    with col1:
                        st.progress(min(st.session_state.countdown / COUNT_DOWN_SECONDS, 1.0))
                        st.text(f"Continuing in {st.session_state.countdown} seconds...")
                    with col2:
                        if st.button("Cancel"):
                            st.session_state.show_countdown = False
                            st.session_state.countdown = COUNT_DOWN_SECONDS
                            st.session_state.cancel_countdown = True
                            st.rerun()
                    with col3:
                        if st.button("Continue"):
                            st.session_state.show_countdown = False
                            st.session_state.countdown = COUNT_DOWN_SECONDS
                            st.session_state.force_continue = True
                            st.rerun()

                # Decrease countdown
                st.session_state.countdown -= 1

                # If countdown reaches zero, send message
                if st.session_state.countdown < 0 or st.session_state.force_continue:
                    st.session_state.show_countdown = False
                    st.session_state.countdown = COUNT_DOWN_SECONDS
                    st.session_state.force_continue = False
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
