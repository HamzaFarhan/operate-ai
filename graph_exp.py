from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

import nest_asyncio
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

nest_asyncio.apply()
load_dotenv()

MODEL = "google-gla:gemini-2.0-flash"

OutputT = TypeVar("OutputT")


class Success(BaseModel):
    message: str


AgentOutput = OutputT | Success


class AgentPrompt(BaseModel):
    prompt: str
    tool: str


@dataclass
class GraphDeps:
    name: str
    location: str

    def __str__(self) -> str:
        return f"<user_name>\n{self.name}\n</user_name>\n<user_location>\n{self.location}\n</user_location>"


user_info_agent = Agent(
    name="user_info_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions="Use the tools at your disposal to fetch user info based on the prompt.",
)


@user_info_agent.tool_plain
def get_user_age(name: str) -> int:
    """
    Returns the age of the user specified by name.
    """
    logger.info(f"Getting age for {name=}")
    age_dict = {"hamza": 29, "rafay": 26}
    return age_dict.get(name, 0)


@user_info_agent.tool_plain
def get_user_location(name: str) -> str:
    """
    Returns the location of the user specified by name.
    """
    logger.info(f"Getting location for {name=}")
    location_dict = {"hamza": "london", "rafay": "paris"}
    return location_dict.get(name, "Unknown")


async def get_user_info(ctx: RunContext[GraphDeps], prompt: str) -> str:
    """
    Processes the given prompt to retrieve user information (age or location).
    Use this tool if the required agent prompt targets user information.

    Args:
        ctx: The graph context.
        prompt: The prompt describing what user info is needed and for whom.

    Returns:
        str: The requested user information or an error message.
    """
    logger.info(f"User Info Agent received prompt: {prompt}")
    res = await user_info_agent.run(user_prompt=prompt, deps=ctx.deps)
    return res.output


weather_agent = Agent(
    name="weather_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions="Use the tools at your disposal to fetch weather info based on the prompt.",
)


@weather_agent.tool_plain
def get_current_weather(location: str) -> str:
    """
    Returns the current weather condition for the given location.
    """
    logger.info(f"Getting weather for {location=}")
    weather_dict = {"london": "sunny", "paris": "rainy"}
    return weather_dict.get(location.lower(), "Unknown weather")


@weather_agent.tool_plain
def get_temperature(location: str) -> float:
    """
    Returns the temperature in Celsius for the given location.
    """
    logger.info(f"Getting temperature for {location=}")
    temp_dict = {"london": 18.5, "paris": 22.0, "new york": 25.5}
    return temp_dict.get(location.lower(), 0.0)


async def get_weather(ctx: RunContext[GraphDeps], prompt: str) -> str:
    """
    Processes the given prompt to retrieve weather information (condition or temperature).
    Use this tool if the required agent prompt targets weather information.

    Args:
        ctx: The graph context.
        prompt: The prompt describing what weather info is needed and for where.

    Returns:
        str: The requested weather information or an error message.
    """
    logger.info(f"Weather Agent received prompt: {prompt}")
    # Pass the prompt directly to the weather_agent
    res = await weather_agent.run(user_prompt=prompt, deps=ctx.deps)
    return res.output


planner_agent = Agent(
    name="planner_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=(
        "Based on the user's overall goal, create a list of prompts for specialized agents.\n"
        "Each item in the list should be an 'AgentPrompt' containing:\n"
        "- prompt: A specific instruction or question for a specialized agent.\n"
        "- tool: The name of the single tool (which represents an agent) that can best handle that prompt.\n"
        "Your plan will be given to an executor agent that routes the prompts to the correct tool/agent.\n"
        "Refer to the <available_tools> list to know which tools/agents the executor has access to.\n"
        "Do not try to execute the prompts yourself. Just create the list of AgentPrompt objects."
    ),
    output_type=list[AgentPrompt],
)


executor_agent = Agent(
    name="executor_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=(
        "You will receive an overall goal and a list of specific prompts targeted at different tools/agents.\n"
        "Execute the prompts sequentially using the specified tool for each.\n"
        "You may add relevant data/context from completed tasks to subsequent prompts as needed.\n"
        "Pass the 'prompt' from the AgentPrompt object to the corresponding tool, "
        "updating it with new context when appropriate.\n"
        "Gather the results from each tool execution.\n"
        "Once all prompts are executed, synthesize the results and return a final answer in a `Success` object.\n"
        "Use the results from previous steps if needed for subsequent prompts."
    ),
    output_type=AgentOutput[str],
    tools=[get_user_info, get_weather],
)


@user_info_agent.instructions
@weather_agent.instructions
@executor_agent.instructions
@planner_agent.instructions
def user_data_context(ctx: RunContext[GraphDeps]) -> str:
    return f"\n\n{ctx.deps}\n\n<current_time>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</current_time>\n\n"


@dataclass
class Planner(BaseNode[None, GraphDeps]):
    user_goal: str
    available_tools: dict[str, Tool[GraphDeps]]

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Executor:
        tools_str = "<available_tools>\n"
        for tool in self.available_tools.values():
            tools_str += f"- {tool.name}: {tool.description}\n\n"
        tools_str += "</available_tools>"
        user_prompt = f"<user_goal>\n{self.user_goal}\n</user_goal>\n\n{tools_str}\n\n"
        logger.info(f"Planner received prompt: {user_prompt}")
        plan_res = await planner_agent.run(user_prompt=user_prompt, deps=ctx.deps)
        return Executor(user_goal=self.user_goal, prompts=plan_res.output)


@dataclass
class Executor(BaseNode[None, GraphDeps, str]):
    user_goal: str
    prompts: list[AgentPrompt]

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> End[str]:
        prompts_str = "<prompts>\n"
        for i, agent_prompt in enumerate(self.prompts):
            prompts_str += f"{i + 1}. Tool: {agent_prompt.tool}, Prompt: {agent_prompt.prompt}\n"
        prompts_str += "</prompts>"
        user_prompt = f"<user_goal>\n{self.user_goal}\n</user_goal>\n\n{prompts_str}\n\n"
        logger.info(f"Executor received prompt: {user_prompt}")
        message_history = None
        while True:
            run = await executor_agent.run(user_prompt=user_prompt, deps=ctx.deps, message_history=message_history)
            if isinstance(run.output, Success):
                return End(run.output.message)
            message_history = run.all_messages()
            user_goal = input(f"{run.output}:  ")
            if user_goal == "q":
                return End(run.output)


async def main():
    graph = Graph(nodes=[Planner, Executor])
    graph_deps = GraphDeps(name="hamza", location="london")

    user_goal = "Tell me the weather and my age."

    logger.info(f"User Goal: {user_goal}")

    res = await graph.run(
        start_node=Planner(user_goal=user_goal, available_tools=executor_agent._function_tools), deps=graph_deps
    )
    return res.output


if __name__ == "__main__":
    res = asyncio.run(main())
    logger.info(f"Final Result: {res}")
