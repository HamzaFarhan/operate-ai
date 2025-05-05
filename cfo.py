from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import duckdb
import logfire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName
from pydantic_ai.tools import Tool
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

load_dotenv()

logfire.configure()

DATA_DIR = "./operateai_scenario1_data"
PROMPTS_DIR = "./src/operate_ai/prompts"
MODEL: KnownModelName = "google-gla:gemini-2.0-flash"
BETTER_MODEL: KnownModelName = "gpt-4.1-mini"


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
    results_dir: str
    available_tools: dict[str, Tool[GraphDeps]]


class ExcelValue(BaseModel):
    column_name: str
    value: Any


class ExcelRow(BaseModel):
    values: list[ExcelValue]

    def to_record(self) -> dict[str, Any]:
        return {value.column_name: value.value for value in self.values}


class TaskSpec(BaseModel):
    goal: str = Field(..., description="One clear sentence describing the deliverable.")
    slots: list[str] = Field(
        ..., description="Every piece of information still required before analysis can start."
    )


class Question(BaseModel):
    content: str


class Slot(BaseModel):
    name: str
    value: str


class Task(BaseModel):
    original_prompt: str
    goal: str
    slots: list[Slot]


class ProgressOrQuestion(BaseModel):
    """
    Use this to return progress or a question to the user.
    """

    content: str


def load_df(df_path: str) -> list[dict[str, str]]:
    """
    Returns the DataFrame from the specified CSV file.
    """
    return pd.read_csv(df_path).to_dict(orient="records")


def write_sheet(rows: list[ExcelRow], sheet_name: str, file_path: str | None = None) -> str:
    """
    Append one sheet (or create a new workbook) and return the file path.

    Parameters
    ----------
    rows : list[ExcelRow]
        Structured rows to write.
    sheet_name : str
        Desired tab name; trimmed to 31 chars to satisfy Excel.
    file_path : str | None, default None
        • Write your results in the 'results_dir'
        • First call: None or a new file path to create a fresh workbook in /tmp and get
          back the generated path.
        • Subsequent calls: reuse the same path to add more sheets.

    Returns
    -------
    str
        Absolute path to the .xlsx file on disk.
    """
    if file_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = str(Path(tempfile.gettempdir()) / f"ai_cfo_report_{ts}.xlsx")

    path = Path(file_path).expanduser().resolve().with_suffix(".xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_name = sheet_name[:31] or "Sheet1"
    records = [r.to_record() for r in rows]
    df = pd.DataFrame.from_records(records)
    mode = "w"
    if_sheet_exists = None
    if path.exists():
        mode = "a"
        if_sheet_exists = "replace"
    with pd.ExcelWriter(path, mode=mode, engine="openpyxl", if_sheet_exists=if_sheet_exists) as w:
        df.to_excel(w, sheet_name=safe_name, index=False)

    return str(path)


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


def add_results_dir(ctx: RunContext[GraphDeps]) -> str:
    return f"\n<results_dir>\n{ctx.deps.results_dir}\n</results_dir>"


def run_sql(query: str) -> list[dict[str, str]]:
    """
    Runs an SQL query on csv file(s) using duckdb and returns the results as a list of dictionaries.
    In the query, include the path to the csv file in single quotes.
    It will return the result by doing a to_df() and then to_dict(orient="records")

    Example Query:
    SELECT * FROM 'full_path_to_csv_file.csv'
    """
    return duckdb.sql(query).to_df().to_dict(orient="records")


task_spec_agent = Agent(
    name="task_spec_agent",
    model=BETTER_MODEL,
    deps_type=GraphDeps,
    instructions=((Path(PROMPTS_DIR) / "task_spec.md").read_text(), add_current_time),
    output_type=TaskSpec,
    instrument=True,
)


slot_collector_agent = Agent(
    name="slot_collector_agent",
    model=BETTER_MODEL,
    deps_type=GraphDeps,
    instructions=(
        (Path(PROMPTS_DIR) / "slot_collector.md").read_text(),
        add_current_time,
        list_csv_files,
        add_results_dir,
    ),
    output_type=AgentOutputT[Question, Slot],
    instrument=True,
)

planner_agent = Agent(
    name="planner_agent",
    model=BETTER_MODEL,
    deps_type=GraphDeps,
    instructions=(
        (Path(PROMPTS_DIR) / "planner.md").read_text(),
        add_current_time,
        list_csv_files,
        list_available_tools,
    ),
    output_type=list[AgentPrompt],
    instrument=True,
)


executor_agent = Agent(
    name="executor_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=((Path(PROMPTS_DIR) / "executor.md").read_text(), add_current_time, list_csv_files),
    output_type=AgentOutputT[ProgressOrQuestion, Success],
    tools=[Tool(run_sql, max_retries=5), Tool(write_sheet, max_retries=3)],
    instrument=True,
)


@dataclass
class TaskSpecExtractor(BaseNode[None, GraphDeps]):
    """Extracts task goal and identifies missing information slots from initial finance request"""

    docstring_notes = True
    user_prompt: str

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> SlotCollector:
        logger.info(f"Prompt for task spec agent: {self.user_prompt}")
        run = await task_spec_agent.run(user_prompt=self.user_prompt, deps=ctx.deps)
        return SlotCollector(original_prompt=self.user_prompt, task_spec=run.output)


@dataclass
class SlotCollector(BaseNode[None, GraphDeps]):
    """Collects values for all missing information slots through user conversation"""

    docstring_notes = True
    original_prompt: str
    task_spec: TaskSpec

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Planner:
        logger.info(f"Task spec:\n{self.task_spec.model_dump_json(indent=2)}\n\n")
        slots = []
        for slot in self.task_spec.slots:
            user_prompt = (
                f"<original_prompt>\n{self.original_prompt}\n</original_prompt>\n\n"
                f"<goal>\n{self.task_spec.goal}\n</goal>\n\n"
                f"<slot_to_collect>\n{slot}\n</slot_to_collect>"
            )
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

                user_prompt = input(f"{run.output.content.strip()} > ")
                if user_prompt.lower() == "q":
                    user_prompt = (
                        "This is dragging on too long. "
                        "Just return a Slot object with the name and value "
                        "based on our conversation so far. Or, come up with a value on your own.\n"
                        "I trust you."
                    )
                message_history = run.all_messages()
        return Planner(task=Task(original_prompt=self.task_spec.goal, goal=self.task_spec.goal, slots=slots))


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
            user_prompt = input(f"{run.output.content} > ")
            if user_prompt.lower() == "q":
                return End("User quit")


cfo_graph = Graph(nodes=[TaskSpecExtractor, SlotCollector, Planner, Executor])
cfo_graph.mermaid_save("cfo_graph.jpg", direction="LR")


async def main():
    graph_deps = GraphDeps(
        data_dir=DATA_DIR, results_dir="./analysis", available_tools=executor_agent._function_tools
    )
    user_prompt = input("Analytical Task: ")
    if Path(user_prompt).exists():
        user_prompt = Path(user_prompt).read_text()
    res = await cfo_graph.run(start_node=TaskSpecExtractor(user_prompt=user_prompt), deps=graph_deps)
    return res.output


if __name__ == "__main__":
    res = asyncio.run(main())
    logger.success(res)
