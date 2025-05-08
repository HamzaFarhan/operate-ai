from pathlib import Path

import nest_asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_json

nest_asyncio.apply()


MODEL = "google-gla:gemini-2.0-flash"

agent = Agent(model=MODEL)
res = agent.run_sync(user_prompt="Hello, how are you?")
Path("messages.json").write_bytes(to_json(res.all_messages()))
res2 = agent.run_sync(
    user_prompt="fine", message_history=ModelMessagesTypeAdapter.validate_json(Path("messages.json").read_bytes())
)
print(res2.output)
