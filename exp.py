from datetime import datetime

from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

MODEL = "google-gla:gemini-2.0-flash"

agent = Agent(model=MODEL)


@agent.instructions
def add_today_date() -> str:
    return f"\n<today_date>{datetime.now().strftime('%Y-%m-%d')}</today_date>\n\n"


user_prompt = "Tell me an interesting fact about today's date throughout history."

res = agent.run_sync(user_prompt=user_prompt)

print(res.output)
