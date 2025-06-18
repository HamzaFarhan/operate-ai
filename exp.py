from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_json

load_dotenv()


agent = Agent(model="google-gla:gemini-2.0-flash")
message_history = (
    ModelMessagesTypeAdapter.validate_json(Path("message_history.json").read_bytes())
    if Path("message_history.json").exists()
    else []
)
res = agent.run_sync(user_prompt="What is the capital of France?", message_history=message_history)
Path("message_history.json").write_bytes(to_json(res.all_messages()))

message_history = ModelMessagesTypeAdapter.validate_json(Path("message_history.json").read_bytes())
res2 = agent.run_sync(user_prompt="and of germany?", message_history=message_history)

print(res2.all_messages())
