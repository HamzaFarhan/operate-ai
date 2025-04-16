from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

agent1 = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt="You are a joke teller",
    instructions="make your jokes super funny/silly and rhyme",
)
agent2 = Agent(model="google-gla:gemini-2.0-flash", system_prompt="You are super serious", instructions="no jokes")

user_prompt = "Tell me a about cats"

res = agent1.run_sync(user_prompt)

res2 = agent2.run_sync("what are your instructions?", message_history=res.all_messages())

print(f"MY INSTRUCTIONS:\n{res2.output}")

[print(msg, "\n") for msg in res2.all_messages()]
