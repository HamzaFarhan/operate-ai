from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import Tool, ToolDefinition

load_dotenv()


@dataclass
class AgentDeps:
    plan_presented: bool = False


async def present_plan(ctx: RunContext[AgentDeps], plan: str) -> str:
    """
    Present the plan to the user.
    """
    ctx.deps.plan_presented = True
    return plan


async def run_sql(ctx: RunContext[AgentDeps], purpose: str, query: str) -> str:
    """
    Run an SQL query.
    """
    ...


async def only_if_plan_presented(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    if ctx.deps.plan_presented:
        return tool_def


agent = Agent(
    model="google-gla:gemini-2.5-flash",
    deps_type=AgentDeps,
    output_type=[present_plan, Tool(run_sql, prepare=only_if_plan_presented)],
)


async def run_agent(
    user_prompt: str, agent_deps: AgentDeps, message_history: list[ModelMessage] | None = None
) -> str:
    while True:
        res = await agent.run(user_prompt=user_prompt, deps=agent_deps, message_history=message_history)
        message_history = res.all_messages()
        user_prompt = input(f"{res.output} > ")
        if user_prompt.lower() in ["q", "quit", "exit"]:
            break
    return res.output
