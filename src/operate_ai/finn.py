from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import logfire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_core import to_json

from operate_ai.csv_utils import csv_summary

load_dotenv()

logfire.configure()

MAX_ANALYSIS_FILE_TOKENS = 20_000
SQL_TIMEOUT_SECONDS = 5
MAX_RETRIES = 10
MEMORY_FILE_NAME = os.getenv("MEMORY_FILE_NAME", "memory.json")
MODULE_DIR = Path(__file__).parent


class AgentDeps:
    def __init__(self, thread_dir: Path, workspace_dir: Path | None = None):
        self.thread_dir = thread_dir.expanduser().resolve()
        self.workspace_dir = workspace_dir or self.thread_dir.parent.parent
        self.analysis_dir = self.thread_dir / "analysis"
        self.results_dir = self.thread_dir / "results"
        self.memory_file_path = self.workspace_dir / MEMORY_FILE_NAME
        self.message_history_path = self.thread_dir / "message_history.json"
        self.plan_path = self.thread_dir / "plan.md"
        self.data_dir = self.workspace_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def add_dirs(ctx: RunContext[AgentDeps]) -> str:
    return (
        f"<data_dir>\n{str(Path(ctx.deps.data_dir).expanduser().resolve())}\n</data_dir>\n"
        f"<analysis_dir>\n{str(Path(ctx.deps.analysis_dir).expanduser().resolve())}\n</analysis_dir>\n"
        f"<results_dir>\n{str(Path(ctx.deps.results_dir).expanduser().resolve())}\n</results_dir>\n\n"
        "`analysis_result_file_name` for `run_sql` will be saved in the `analysis_dir`.\n"
        "`workbook_name` for `write_sheet_from_file` will be saved in the `results_dir`."
    )


def add_current_time() -> str:
    return f"<current_time>\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n</current_time>"


def create_sequential_plan(ctx: RunContext[AgentDeps], initial_plan: str) -> str:
    """Creates a new sequential methodology plan file with user-approved systematic analysis steps.

    This creates a new plan.md file containing the SEQUENTIAL METHODOLOGY steps that were presented to the user during systematic planning and approved by them. These are the atomic, unambiguous, sequential steps that you will follow to complete the financial analysis. Users can view and edit this file.

    You can include more detailed specifications for each step than what was in the original
    SEQUENTIAL METHODOLOGY section - the user approved the overall approach, and you can
    expand with specific implementation details from anywhere in the approved plan.

    Use this after user has approved your systematic analysis plan from the planning phase.

    Args:
        initial_plan: The user-approved sequential methodology steps formatted as markdown.
            Can include more detailed step descriptions than the original methodology section.

    Returns:
        str: A formatted string containing the created plan content wrapped in XML tags.

    Example:
        create_sequential_plan("## SEQUENTIAL METHODOLOGY (User Approved)\n1. Data preparation: Filter active customers using subscription table\n2. Base calculation: Calculate monthly revenue per customer cohort\n3. Final output: Generate cohort analysis table with retention metrics")
    """
    plan_file = ctx.deps.plan_path
    plan_file.write_text(initial_plan)
    return f"<sequential_plan>\nCreated new plan file at: {plan_file}\n\n{initial_plan}\n</sequential_plan>"


def update_sequential_plan_step(ctx: RunContext[AgentDeps], old_text: str, new_text: str) -> str:
    """Updates existing content in the user-approved sequential methodology plan file.

    This replaces specific text in the existing plan file containing the SEQUENTIAL METHODOLOGY steps that were approved by the user. Commonly used to mark steps as completed during analysis execution, update step descriptions, or modify existing methodology steps. Only replaces the first occurrence for precision.

    You can add more detailed specifications when updating steps - the user approved the overall
    methodology, so you can expand with specific implementation details as needed during execution.

    Use this during analysis execution to track progress on the user-approved systematic plan.

    Args:
        old_text: The exact text to find and replace in the plan file.
        new_text: The replacement text. Can include more details than the original step.

    Returns:
        str: A formatted string containing the updated plan content wrapped in XML tags.

    Examples:
        # Mark a step as completed during execution
        update_sequential_plan_step("1. Data preparation: Filter active customers", "1. Data preparation: Filter active customers - ✓ COMPLETED")

        # Update step with more specific business logic
        update_sequential_plan_step("2. Base calculation: Calculate revenue metrics", "2. Base calculation: Calculate MRR using active customer count × ARPU")
    """
    plan_file = ctx.deps.plan_path

    if not plan_file.exists():
        raise ModelRetry(f"Plan file does not exist at: {plan_file}")

    current_content = plan_file.read_text()

    if old_text not in current_content:
        return f"<sequential_plan>\nText not found in plan: '{old_text}'\n\nCurrent plan:\n{current_content}\n</sequential_plan>"

    updated_content = current_content.replace(old_text, new_text, 1)  # Replace only first occurrence
    plan_file.write_text(updated_content)

    return f"<sequential_plan>\nUpdated plan step successfully.\n\n{updated_content}\n</sequential_plan>"


def add_sequential_plan_step(ctx: RunContext[AgentDeps], new_step: str) -> str:
    """Adds a new step to the user-approved sequential methodology plan file.

    This adds a new step to the end of the existing plan file containing the SEQUENTIAL METHODOLOGY steps that were approved by the user. Use this when you discover during analysis execution that additional methodology steps are needed that weren't in the original user-approved plan, or when expanding the analysis scope based on findings.

    Args:
        new_step: The new methodology step to append to the plan. Should be properly formatted
            and follow the atomic, sequential pattern (e.g., "6. Validate results against business logic").

    Returns:
        str: A formatted string containing the updated plan content wrapped in XML tags.

    Example:
        add_sequential_plan_step("4. Validation step: Cross-check MRR calculations against transaction totals")
    """
    plan_file = ctx.deps.plan_path

    if not plan_file.exists():
        raise ModelRetry(f"Plan file does not exist at: {plan_file}")

    current_content = plan_file.read_text()
    updated_content = current_content.rstrip() + "\n" + new_step + "\n"
    plan_file.write_text(updated_content)

    return f"<sequential_plan>\nAdded new step to plan.\n\n{updated_content}\n</sequential_plan>"


def _list_csv_files(dir: Path) -> str:
    """
    Lists all available csv files in the `dir` and their summaries.
    """
    dir_name = dir.name
    csv_files = [file.expanduser().resolve() for file in Path(dir).glob("*.csv")]
    res = f"\n<available_{dir_name}_files>\n"
    for file in csv_files:
        res += json.dumps(csv_summary(file_path=file)) + "\n\n"
    return res.strip() + f"\n</available_{dir_name}_files>"


def list_data_files(ctx: RunContext[AgentDeps]) -> str:
    """
    Lists all available csv files in the `data_dir` and their summaries.
    """
    return _list_csv_files(dir=Path(ctx.deps.data_dir))


def list_analysis_files(ctx: RunContext[AgentDeps]) -> str:
    """
    Lists all the analysis csv files created so far in the `analysis_dir` and their summaries.
    """
    return _list_csv_files(dir=Path(ctx.deps.analysis_dir))


class RunSQLResult(BaseModel):
    purpose: str
    sql_path: str
    analysis_result_file_path: str
    summary: dict[str, Any]


async def run_sql(
    ctx: RunContext[AgentDeps], purpose: str, query: str, analysis_result_file_name: str
) -> RunSQLResult:
    """Runs an SQL query on csv file(s) using DuckDB and writes the result to analysis_dir.

    Args:
        purpose: Describe what you want to achieve with the query and why. Relate it to the main task.
            This will be a user-facing message, so no need to be too technical or mention the internal errors faced if any.
        query: The SQL query to execute.
            When reading a csv, use the FULL path with the data_dir included.
            Example: 'select names from read_csv('workspaces/1/data/orders.csv')'
        analysis_result_file_name: Descriptive name of the file based on the query to save the result to in the `analysis_dir`.
            Example: 'customers_joined_in_2023.csv'
            '.csv' is optional. It will be added automatically if not provided.

    Returns:
        RunSQLResult: Contains the purpose, file paths, summary.
    """

    async def _run_with_timeout():
        query_replaced = query.replace('"', "'").strip()
        logger.info(f"Running SQL query: {query_replaced}")

        try:
            df: pd.DataFrame = duckdb.sql(query_replaced).df()  # type: ignore
        except Exception:
            # Handle transaction rollback for DuckDB transaction errors
            try:
                duckdb.sql("ROLLBACK")
                logger.info("Rolled back DuckDB transaction after error")
            except Exception:
                pass  # Rollback might fail if no transaction is active
            raise

        logger.success(f"Successfully ran SQL query: {query_replaced}")
        file_path = Path(ctx.deps.analysis_dir) / Path(analysis_result_file_name).name
        file_path = file_path.expanduser().resolve().with_suffix(".csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            file_path = file_path.with_stem(f"{file_path.stem}_{uuid4()}")

        df.to_csv(file_path, index=False)
        file_path.with_suffix(".sql").write_text(query_replaced)

        return RunSQLResult(
            purpose=purpose,
            sql_path=str(file_path.with_suffix(".sql")),
            analysis_result_file_path=str(file_path),
            summary=csv_summary(file_path=file_path),
        )

    try:
        return await asyncio.wait_for(_run_with_timeout(), timeout=SQL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise ModelRetry(f"SQL query execution timed out after {SQL_TIMEOUT_SECONDS} seconds")
    except FileNotFoundError as e:
        raise ModelRetry(str(e))
    except Exception as e:
        raise ModelRetry(str(e))


def load_analysis_file(ctx: RunContext[AgentDeps], file_name: str) -> list[dict[str, Any]]:
    """
    Loads an analysis csv file from the `analysis_dir` and returns all of the records.
    """
    file_path = Path(ctx.deps.analysis_dir) / Path(file_name).name
    records = pd.read_csv(file_path).to_dict(orient="records")  # type: ignore
    tokens = len(json.dumps(records)) / 4
    if tokens > MAX_ANALYSIS_FILE_TOKENS:
        raise ModelRetry(
            (
                f"The file {file_name} is too large. It has approx {tokens} tokens.\n "
                f"We can only load files with less than {MAX_ANALYSIS_FILE_TOKENS} tokens.\n"
                f"File Summary:\n{csv_summary(file_path=file_path)}\n"
                "Please try using your SQL expertise to get a smaller subset of the data. "
                "If you have already tried that, let the user know with a helpful message.\n"
                "The user won't know about tokens, but they will understand rows and columns."
            )
        )
    return records  # type: ignore


def calculate_sum(values: list[float]) -> float:
    """Calculate the sum of a list of values."""
    if len(values) < 2:
        raise ModelRetry("Need at least 2 values to calculate sum")
    return sum(values)


def calculate_difference(num1: float, num2: float) -> float:
    """Calculate the difference between two numbers."""
    return abs(num1 - num2)


def calculate_mean(values: list[float]) -> float:
    """Calculate the mean of a list of values."""
    if len(values) < 2:
        raise ModelRetry("Need at least 2 values to calculate mean")
    return sum(values) / len(values)


class WriteDataToExcelResult(BaseModel):
    excel_file_path: str


class UserInteraction(BaseModel):
    """
    Interacts with the user. Could be:
    - A question
    - A progress update
    - An assumption made that needs to be validated
    - A request for clarification
    - Anything else needed from the user to proceed
    """

    message: str = Field(description="The message to display to the user.")


class TaskResult(BaseModel):
    """
    A task result.
    """

    message: str = Field(description="The final response to the user.")


def create_agent(
    agent_deps: AgentDeps,
    model: Model | KnownModelName,
    use_excel_tools: bool = True,
    use_thinking: bool = True,
    use_memory: bool = True,
    temperature: float = 0.0,
) -> Agent[AgentDeps, RunSQLResult | WriteDataToExcelResult | UserInteraction | TaskResult]:
    thinking_server = MCPServerStdio(
        command="npx", args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
    )
    memory_server = MCPServerStdio(
        command="uv",
        args=["run", str(MODULE_DIR / "memory_mcp.py")],
        env={"MEMORY_FILE_PATH": str(agent_deps.memory_file_path)},
    )
    excel_server = MCPServerStdio(command="uvx", args=[str(MODULE_DIR / "../../../excel-mcp-server"), "stdio"])
    mcp_servers: list[MCPServerStdio] = []
    prompts = [Path(MODULE_DIR / "prompts/cfo_v2.md").read_text()]
    if use_thinking:
        mcp_servers.append(thinking_server)
    if use_memory:
        mcp_servers.append(memory_server)
        prompts.append(Path(MODULE_DIR / "prompts/memory.md").read_text())
    if use_excel_tools:
        mcp_servers.append(excel_server)
    output_types: list[type | Callable[..., Any]] = [run_sql, UserInteraction, TaskResult]
    if use_excel_tools:
        output_types.append(WriteDataToExcelResult)
    return Agent(
        model=model,
        instructions=[*prompts, add_current_time, add_dirs, list_data_files, list_analysis_files],
        deps_type=AgentDeps,
        retries=MAX_RETRIES,
        tools=[
            load_analysis_file,
            calculate_sum,
            calculate_difference,
            calculate_mean,
            create_sequential_plan,
            update_sequential_plan_step,
            add_sequential_plan_step,
        ],
        mcp_servers=mcp_servers,
        output_type=output_types,
        model_settings={"temperature": temperature},
        instrument=True,
    )


async def thread(
    thread_dir: Path,
    user_prompt: str,
    model: KnownModelName | FallbackModel | None = None,
    use_excel_tools: bool = True,
    use_thinking: bool = True,
    use_memory: bool = True,
    temperature: float = 0.0,
) -> RunSQLResult | WriteDataToExcelResult | UserInteraction | TaskResult:
    agent_deps = AgentDeps(thread_dir=thread_dir)
    model = model or FallbackModel(
        "anthropic:claude-4-sonnet-20250514",
        "openai:gpt-4.1",
        "google-gla:gemini-2.5-flash-preview-05-20",
        "openai:gpt-4.1-mini",
    )
    agent = create_agent(
        model=model,
        agent_deps=agent_deps,
        use_excel_tools=use_excel_tools,
        use_thinking=use_thinking,
        use_memory=use_memory,
        temperature=temperature,
    )
    res = await agent.run(
        user_prompt=user_prompt,
        deps=agent_deps,
        message_history=ModelMessagesTypeAdapter.validate_json(agent_deps.message_history_path.read_bytes())
        if agent_deps.message_history_path.exists()
        else None,
    )
    agent_deps.message_history_path.write_bytes(to_json(res.all_messages()))

    return res.output


output = None
step = 0
step_limit = 10
while not isinstance(output, TaskResult) and step < step_limit:
    output = await thread(
        thread_dir=thread_dir,
        user_prompt=user_prompt,
        model=model,
        use_excel_tools=use_excel_tools,
        use_thinking=use_thinking,
        use_memory=use_memory,
    )
    logger.info(f"Output: {output}")
    if do_user_interaction:
        user_prompt = input(f"{output} > ")
    else:
        user_prompt = (
            "The user is not available to provide input for this task. "
            "Please proceed independently by making reasonable assumptions where needed.\n"
            "Compile all your assumptions, any issues encountered, and your solutions in your final comprehensive response.\n"
            "Push through to completion even if you think you need user clarification - just document what you assumed and why."
        )
