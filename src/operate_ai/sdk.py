from __future__ import annotations

import shutil
from typing import Any

import httpx
from loguru import logger

from operate_ai.backend import DATA_ROOT, MessageSend, ThreadCreate, ThreadRead, WorkspaceCreate, WorkspaceRead


class OperateAIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client()

    def __enter__(self) -> OperateAIClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.client.close()

    def create_workspace(self, name: str) -> WorkspaceRead:
        """Create a new workspace."""
        response = self.client.post(
            f"{self.base_url}/workspaces",
            json=WorkspaceCreate(name=name).model_dump(),
        )
        response.raise_for_status()
        return WorkspaceRead.model_validate(response.json())

    def list_workspaces(self) -> list[WorkspaceRead]:
        """List all workspaces."""
        response = self.client.get(f"{self.base_url}/workspaces")
        response.raise_for_status()
        return [WorkspaceRead.model_validate(w) for w in response.json()]

    def create_thread(self, workspace_id: int, initial_prompt: str | None = None) -> ThreadRead:
        """Create a new thread in a workspace."""
        response = self.client.post(
            f"{self.base_url}/workspaces/{workspace_id}/threads",
            json=ThreadCreate(initial_prompt=initial_prompt).model_dump(),
        )
        response.raise_for_status()
        return ThreadRead.model_validate(response.json())

    def list_threads(self, workspace_id: int) -> list[ThreadRead]:
        """List all threads in a workspace."""
        response = self.client.get(f"{self.base_url}/workspaces/{workspace_id}/threads")
        response.raise_for_status()
        return [ThreadRead.model_validate(t) for t in response.json()]

    def get_thread(self, workspace_id: int, thread_id: int) -> ThreadRead:
        """Get details of a specific thread, including status and last agent message."""
        response = self.client.get(f"{self.base_url}/workspaces/{workspace_id}/threads/{thread_id}")
        response.raise_for_status()
        return ThreadRead.model_validate(response.json())

    def send_message(self, workspace_id: int, thread_id: int, content: str | None = None) -> dict[str, str]:
        """Send a message to a thread."""
        response = self.client.post(
            f"{self.base_url}/workspaces/{workspace_id}/threads/{thread_id}/messages",
            json=MessageSend(content=content).model_dump(),
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict[str, str]:
        """Check if the API is healthy."""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# Example usage:
if __name__ == "__main__":
    with OperateAIClient() as client:
        # Check health
        logger.info("Health check: {}", client.health_check())

        # Create workspace
        workspace = client.create_workspace(name="test-sdk-workspace")
        logger.info("Created workspace: {}", workspace.model_dump_json(indent=2))
        data_dir = DATA_ROOT / f"{workspace.id}/data"
        shutil.copytree("/Users/hamza/dev/operate-ai/operateai_scenario1_data", data_dir, dirs_exist_ok=True)

        # Create thread with initial prompt
        thread_info = client.create_thread(
            workspace_id=workspace.id,
            initial_prompt="How many customers in 2023 had a monthly plan? doesn't matter when they joined",
        )
        logger.info("Created thread: {}", thread_info.model_dump_json(indent=2))

        # Poll for thread status (simplified polling loop)
        logger.info("Polling for thread {} updates...", thread_info.id)
        import time

        for _ in range(10):  # Poll for a short period
            time.sleep(2)
            updated_thread = client.get_thread(workspace_id=workspace.id, thread_id=thread_info.id)
            logger.info(
                "Thread status: {}, Message: {}", updated_thread.agent_status, updated_thread.last_agent_message
            )
            if updated_thread.agent_status == "AWAITING_INPUT":
                logger.info("Agent is awaiting input. Responding...")
                # Send follow-up message
                response = client.send_message(
                    workspace_id=workspace.id, thread_id=thread_info.id, content=input()
                )
                logger.info("Sent message response: {}", response)
            elif updated_thread.agent_status == "ERROR":
                logger.error("Agent encountered an error.")
                break
            if updated_thread.agent_status not in [
                "QUEUED",
                "PROCESSING",
            ]:  # Stop if it's awaiting input or errored
                if updated_thread.agent_status == "AWAITING_INPUT" and "product category" in (
                    updated_thread.last_agent_message or ""
                ):
                    # If it's asking about the follow up, we can stop for this example
                    logger.info("Agent responded to follow-up, exiting poll.")
                    break
        else:
            logger.info("Polling finished.")

        final_thread_state = client.get_thread(workspace_id=workspace.id, thread_id=thread_info.id)
        logger.info("Final thread state: {}", final_thread_state.model_dump_json(indent=2))
