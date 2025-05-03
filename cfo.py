from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import duckdb
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName
from pydantic_ai.tools import Tool
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

load_dotenv()


DATA_DIR = "./operateai_scenario1_data"
PROMPTS_DIR = "./src/operate_ai/prompts"
MODEL: KnownModelName = "google-gla:gemini-2.0-flash"
PLANNER_MODEL: KnownModelName = "google-gla:gemini-2.5-pro-exp-03-25"


SuccessT = TypeVar("SuccessT")
UntilSuccessT = TypeVar("UntilSuccessT")


class Success(BaseModel):
    value: str


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


def run_sql(query: str) -> list[dict[str, str]]:
    """
    Runs a SQL query on csv file(s) using duckdb and returns the results as a list of dictionaries.
    In the query, include the path to the csv file in single quotes.
    It will return the result by doing a to_df() and then to_dict(orient="records")

    Example Query:
    SELECT * FROM 'full_path_to_csv_file.csv'
    """
    return duckdb.sql(query).to_df().to_dict(orient="records")


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
    instructions=((Path(PROMPTS_DIR) / "slot_collector.md").read_text(), add_current_time, list_csv_files),
    output_type=AgentOutputT[str, Slot],
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
    tools=[Tool(run_sql, max_retries=5)],
)


@dataclass
class TaskSpecExtractor(BaseNode[None, GraphDeps]):
    """Extracts task goal and identifies missing information slots from initial finance request"""

    docstring_notes = True
    user_prompt: str

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> SlotCollector:
        logger.info(f"Prompt for task spec agent: {self.user_prompt}")
        run = await task_spec_agent.run(user_prompt=self.user_prompt, deps=ctx.deps)
        return SlotCollector(task_spec=run.output)


@dataclass
class SlotCollector(BaseNode[None, GraphDeps]):
    """Collects values for all missing information slots through user conversation"""

    docstring_notes = True
    task_spec: TaskSpec

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Planner:
        slots = []
        for slot in self.task_spec.slots:
            user_prompt = f"<goal>\n{self.task_spec.goal}\n</goal>\n\n<slot>\n{slot}\n</slot>"
            if slots:
                user_prompt += f"\n\nAlready collected slots:\n{slots}"
            message_history = None
            while True:
                logger.info(f"Prompt for slot collector agent: {user_prompt}")
                run = await slot_collector_agent.run(
                    user_prompt=user_prompt, deps=ctx.deps, message_history=message_history
                )
                if isinstance(run.output, Slot):
                    slots.append(run.output)
                    break
                if run.output.strip().startswith("Slot("):
                    user_prompt = "Return the Slot object with the name and value. Not a string."
                else:
                    user_prompt = input(f"{run.output.strip()} > ")
                if user_prompt.lower() == "q":
                    user_prompt = (
                        "This is dragging on too long. "
                        "Just return a Slot object with the name and value "
                        "based on our conversation so far. Or, come up with a value on your own.\n"
                        "I trust you."
                    )
                message_history = run.all_messages()
        return Planner(task=Task(goal=self.task_spec.goal, slots=slots))


@dataclass
class Planner(BaseNode[None, GraphDeps]):
    """Converts fully specified task into ordered execution plan of tool steps"""

    docstring_notes = True
    task: Task

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Executor:
        user_prompt = f"<task>\n{self.task.model_dump_json()}\n</task>"
        logger.info(f"Prompt for planner agent: {user_prompt}")
        plan_res = await planner_agent.run(user_prompt=user_prompt, deps=ctx.deps)
        return Executor(task=self.task, prompts=plan_res.output)


@dataclass
class Executor(BaseNode[None, GraphDeps, str]):
    """Executes ordered plan steps and synthesizes results into final answer"""

    docstring_notes = True
    task: Task
    prompts: list[AgentPrompt]

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> End[str]:
        prompts_str = "<prompts>\n"
        for i, agent_prompt in enumerate(self.prompts):
            prompts_str += f"{i + 1}. Tool: {agent_prompt.tool}, Prompt: {agent_prompt.prompt}\n"
        prompts_str += "</prompts>"
        user_prompt = f"<task>\n{self.task.model_dump_json()}\n</task>\n\n{prompts_str}"
        message_history = None
        while True:
            logger.info(f"Prompt for executor agent: {user_prompt}")
            run = await executor_agent.run(user_prompt=user_prompt, deps=ctx.deps, message_history=message_history)
            if isinstance(run.output, Success):
                return End(run.output.value)
            message_history = run.all_messages()
            user_prompt = input(f"{run.output} > ")
            if user_prompt.lower() == "q":
                return End(run.output)


cfo_graph = Graph(nodes=[TaskSpecExtractor, SlotCollector, Planner, Executor])
cfo_graph.mermaid_save("cfo_graph.jpg", direction="LR")


async def main():
    graph_deps = GraphDeps(data_dir=DATA_DIR, available_tools=executor_agent._function_tools)
    user_prompt = input("Analytical Task: ")
    if Path(user_prompt).exists():
        user_prompt = Path(user_prompt).read_text()
    res = await cfo_graph.run(start_node=TaskSpecExtractor(user_prompt=user_prompt), deps=graph_deps)
    return res.output


if __name__ == "__main__":
    res = asyncio.run(main())
    logger.success(res)
