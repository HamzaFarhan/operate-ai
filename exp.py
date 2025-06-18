from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from operate_ai.finn import add_current_time

load_dotenv()


async def get_weather(city: str) -> str:
    """
    Get the weather for a given city.
    """
    weather_map = {
        "New York": "sunny",
        "Los Angeles": "sunny",
        "Chicago": "cloudy",
        "Houston": "rainy",
        "Miami": "sunny",
    }
    return weather_map.get(city, "unknown")


async def get_item_price(item: str) -> float:
    """
    Get the price of a given item.
    """
    price_map = {
        "apple": 1.00,
        "banana": 0.50,
        "orange": 0.75,
        "eggs": 2.00,
        "milk": 3.00,
    }
    return price_map.get(item, 0.00)


async def add_available_tools() -> str:
    return (
        "<tool_list>\n"
        "get_weather: Get the weather for a given city.\n"
        "get_item_price: Get the price of a given item.\n"
        "</tool_list>"
    )


class ProposedPlan(BaseModel):
    plan: str = Field(description="The plan that to be approved by the user.")


class ApprovedPlan(BaseModel):
    plan: str = Field(description="The plan that the user has approved.")


planner_agent = Agent(
    model="google-gla:gemini-2.5-flash",
    name="planner",
    instructions=[
        add_current_time,
        add_available_tools,
        "Make a detailed plan based on the user's task.\n"
        "Start off by returning a `ProposedPlan` object.\n"
        "Keep interacting with the user until the plan is finalized.\n"
        "Once finalized, return an `ApprovedPlan` object.",
    ],
    output_type=[ProposedPlan, ApprovedPlan, str],
)

steps_agent = Agent(
    model="google-gla:gemini-2.5-flash",
    name="steps",
    instructions=[
        add_current_time,
        add_available_tools,
        "Given the user's task and the approved plan, "
        "return a list of actionable steps to complete the task.\n"
        "Each step will be ticked off as it is completed.\n"
        "You or the user may edit a step or add more steps as needed.",
    ],
    output_type=list[str],
)

task = "I am going to buy eggs in houston. should I bring an umbrella? and how much money should I bring?"


def plan_task(user_prompt: str, message_history: list[ModelMessage] | None = None) -> str:
    while True:
        res = planner_agent.run_sync(user_prompt=user_prompt, message_history=message_history)
        message_history = res.all_messages()
        if isinstance(res.output, ApprovedPlan):
            break
        user_prompt = input(f"{res.output} > ")
    return res.output.plan


plan = plan_task(task)
print("\n", plan.strip(), "\n\n")
steps = steps_agent.run_sync(user_prompt=f"<task>\n{task}\n</task>\n\n<approved_plan>\n{plan}\n</approved_plan>")
print(steps.output)
