from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart

load_dotenv()


def user_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


message_history = None
agent = Agent(model="google-gla:gemini-2.0-flash", instructions="Be jolly", message_history=message_history)
run = await agent.run("What is the capital of France?")
message_history = run.all_messages()
message_history.append(user_message("nice"))

run2 = await agent.run(message_history=message_history)
run2.all_messages()
