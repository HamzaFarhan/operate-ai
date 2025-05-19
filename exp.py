from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()


agent = Agent(model="google-gla:gemini-2.0-flash")

await agent.run(user_prompt="")
