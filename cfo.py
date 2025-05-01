from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()


DATA_DIR = "./operateai_scenario1_data"
PROMPTS_DIR = "./src/operate_ai/prompts"
MODEL = "google-gla:gemini-2.0-flash"


SuccessT = TypeVar("SuccessT")
UntilSuccessT = TypeVar("UntilSuccessT")


class Success(BaseModel):
    message: str


AgentOutputT = UntilSuccessT | SuccessT


class AgentPrompt(BaseModel):
    prompt: str
    tool: str


@dataclass
class GraphDeps:
    data_dir: str
    available_tools: dict[str, Tool[GraphDeps]]


def add_current_time() -> str:
    return f"\n<current_time>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</current_time>"


def preview_df(df_path: str, num_rows: int = 3) -> list[dict[str, str]]:
    """
    Returns the first num_rows rows of the specified DataFrame.
    Should be used for previewing the data.
    """
    return pd.read_csv(df_path).head(num_rows).to_dict(orient="records")


def list_csv_files(ctx: RunContext[GraphDeps]) -> str:
    """
    Lists all CSV files in the specified directory.
    """
    csv_files = [str(file) for file in Path(ctx.deps.data_dir).glob("*.csv")]
    res = "\nAvailable CSV files:\n"
    for file in csv_files:
        res += file + "\n"
        res += "First row for preview:\n"
        res += json.dumps(preview_df(df_path=file, num_rows=1)) + "\n\n"
    return res.strip()


def list_available_tools(ctx: RunContext[GraphDeps]) -> str:
    tools_str = "\nAvailable tools:\n"
    for tool in ctx.deps.available_tools.values():
        tools_str += f"- {tool.name}: {tool.description}\n\n"
    return tools_str.strip()


class TaskSpec(BaseModel):
    goal: str = Field(..., description="One clear sentence describing the deliverable.")
    slots: list[str] = Field(
        ..., description="Every piece of information still required before analysis can start."
    )


class Slot(BaseModel):
    name: str
    value: str


class Task(BaseModel):
    goal: str
    slots: list[Slot]


task_spec_agent = Agent(
    name="task_spec_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=((Path(PROMPTS_DIR) / "task_spec.md").read_text(), add_current_time),
    output_type=TaskSpec,
)


slot_collector_agent = Agent(
    name="slot_collector_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=((Path(PROMPTS_DIR) / "collect_slots.md").read_text(), add_current_time, list_csv_files),
    output_type=Slot,
)

planner_agent = Agent(
    name="planner_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=(
        (Path(PROMPTS_DIR) / "planner.md").read_text(),
        add_current_time,
        list_csv_files,
        list_available_tools,
    ),
    output_type=list[AgentPrompt],
)


executor_agent = Agent(
    name="executor_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=((Path(PROMPTS_DIR) / "executor.md").read_text(), add_current_time, list_csv_files),
    output_type=AgentOutputT[str, Success],
)


@dataclass
class TaskSpecExtractor(BaseNode[None, GraphDeps]):
    user_prompt: str

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> SlotCollector:
        run = await task_spec_agent.run(user_prompt=self.user_prompt, deps=ctx.deps)
        return SlotCollector(task_spec=run.output)


@dataclass
class SlotCollector(BaseNode[None, GraphDeps]):
    task_spec: TaskSpec

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Planner:
        slots = []
        message_history = None
        for slot in self.task_spec.slots:
            run = await slot_collector_agent.run(
                user_prompt=f"<goal>\n{self.task_spec.goal}\n</goal>\n\n<slot>\n{slot}\n</slot>",
                deps=ctx.deps,
                message_history=message_history,
            )
            slots.append(run.output)
            message_history = run.all_messages()
        return Planner(task=Task(goal=self.task_spec.goal, slots=slots))


@dataclass
class Planner(BaseNode[None, GraphDeps]):
    task: Task

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Executor:
        plan_res = await planner_agent.run(
            user_prompt=f"<task>\n{self.task.model_dump_json()}\n</task>", deps=ctx.deps
        )
        return Executor(task=self.task, prompts=plan_res.output)


@dataclass
class Executor(BaseNode[None, GraphDeps, str]):
    task: Task
    prompts: list[AgentPrompt]

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> End[str]:
        prompts_str = "<prompts>\n"
        for i, agent_prompt in enumerate(self.prompts):
            prompts_str += f"{i + 1}. Tool: {agent_prompt.tool}, Prompt: {agent_prompt.prompt}\n"
        prompts_str += "</prompts>"
        user_prompt = (
            f"<task>\n{self.task.model_dump_json()}\n</task>\n\n{prompts_str}\n\n"
        )
        console.print(
            Panel(
                f"[info]Executor received prompt:\n[bold]{user_prompt}[/bold][/info]",
                title="Internal Log",
                border_style="dim blue",
            )
        )
        message_history = None
        while True:
            run = await executor_agent.run(user_prompt=user_prompt, deps=ctx.deps, message_history=message_history)
            if isinstance(run.output, Success):
                return End(run.output.message)
            message_history = run.all_messages()
            console.print(Panel(f"[ai]{run.output}[/ai]", title="AI Message", border_style="magenta"))
            user_goal = Prompt.ask("[user]Your response[/user]", console=console)
            if user_goal == "q":
                return End(run.output)


async def main():
    graph = Graph(nodes=[TaskInfoNode, Planner, Executor])
    graph_deps = GraphDeps(available_tools=executor_agent._function_tools)
    # first_prompt = "Get the company revenue and competitor analysis of 'Tech Solutions'"

    # Display welcome message
    console.print(
        Panel(
            "[ai]Welcome to OperateAI![/ai]",
            title="AI Assistant",
            border_style="magenta",
        )
    )

    # Get user input with rich prompt
    user_prompt = Prompt.ask("[user]What can I help you with?[/user]", console=console)

    # Log user input with rich panel
    console.print(Panel(f"[user]{user_prompt}[/user]", title="User Request", border_style="green"))

    res = await graph.run(start_node=TaskInfoNode(user_prompt=user_prompt), deps=graph_deps)
    return res.output


if __name__ == "__main__":
    res = asyncio.run(main())
    console.print(Panel(f"[ai]{res}[/ai]", title="Final Result", border_style="bold magenta", padding=(1, 2)))
