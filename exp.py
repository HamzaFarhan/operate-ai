from datetime import datetime

from pydantic_ai import Agent

MODEL = "google-gla:gemini-2.0-flash"

# INSTEAD OF THIS

joker_agent = Agent(name="joker_agent", model=MODEL, instructions="you write jokes")
poet_agent = Agent(name="poet_agent", model=MODEL, instructions="you write poetry")


@joker_agent.instructions
@poet_agent.instructions
def add_current_time() -> str:
    return f"\n\n<current_time>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</current_time>\n\n"


# I WANT THIS


def add_current_time_instructions() -> str:
    return f"\n\n<current_time>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</current_time>\n\n"


joker_agent = Agent(
    name="joker_agent", model=MODEL, instructions=("you write jokes", add_current_time_instructions)
)
poet_agent = Agent(
    name="poet_agent", model=MODEL, instructions=("you write poetry", add_current_time_instructions)
)
