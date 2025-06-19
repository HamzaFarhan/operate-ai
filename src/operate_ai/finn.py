from __future__ import annotations

import asyncio
import json
import os
import shutil
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


def create_plan_steps(ctx: RunContext[AgentDeps], initial_plan: str) -> str:
    """Creates a new sequential plan file with user-approved systematic analysis steps.

    This creates a new plan.md file containing the steps that were presented to the user during systematic planning and approved by them. These are the atomic, unambiguous, sequential steps that you will follow to complete the financial analysis. Users can view and edit this file.

    You can include more detailed specifications for each step than what was in the original
    section - the user approved the overall approach, and you can
    expand with specific implementation details from anywhere in the approved plan.

    Use this after user has approved your systematic analysis plan from the planning phase.

    Args:
        initial_plan: The user-approved sequential steps formatted as markdown. Can include more detailed step descriptions than the original section.

    Returns:
        str: A formatted string containing the created plan content wrapped in XML tags.

    Example:
        create_plan_steps("## SEQUENTIAL STEPS (User Approved)\n1. Data preparation: Filter active customers using subscription table\n2. Base calculation: Calculate monthly revenue per customer cohort\n3. Final output: Generate cohort analysis table with retention metrics")
    """
    plan_file = ctx.deps.plan_path
    plan_file.write_text(initial_plan)
    return f"<sequential_plan>\n{initial_plan}\n</sequential_plan>"


def update_plan(ctx: RunContext[AgentDeps], old_text: str, new_text: str) -> str:
    """Updates existing content in the user-approved sequential plan file.

    This replaces specific text in the existing plan file containing the steps that were approved by the user. Only replaces the first occurrence for precision.

    **TOKEN EFFICIENCY:** Use the absolute minimum substring needed. Don't use full steps or even full step titles - use just enough to uniquely identify what to replace:
    - **For completion:** Use just the unique ending part (e.g., "revenue analysis" instead of "1. Data preparation: Load and clean revenue analysis")
    - **For content updates:** Use just the unique middle part that needs changing
    - **For full step replacement:** Use the full step text when the entire step needs to be rewritten
    - **Only use more text if absolutely necessary** to differentiate from similar content

    **MULTIPLE UPDATES:** When one SQL query accomplishes multiple steps, prefer making separate calls to this tool for each step rather than one large call. This keeps each call token-efficient while still tracking all progress.

    Args:
        old_text: The absolute minimal substring to find and replace. Use only as much text as needed for uniqueness.
        new_text: The replacement text.

    Returns:
        str: A formatted string containing the updated plan content wrapped in XML tags.

    Examples:
        # Maximum efficiency: Use minimal unique substring
        # For "1. Data preparation: Load customer data" vs "2. Data preparation: Load revenue data"
        update_plan("customer data", "customer data - ✓ COMPLETED")
        update_plan("revenue data", "revenue data - ✓ COMPLETED")

        # Content updates: Use minimal middle part
        update_plan("Calculate revenue", "Calculate MRR using ARPU")
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


def add_plan_step(ctx: RunContext[AgentDeps], new_step: str) -> str:
    """Adds a new step to the user-approved sequential plan file.

    This adds a new step to the end of the existing plan file containing the steps that were approved by the user. Use this when you discover during analysis execution that additional steps are needed that weren't in the original user-approved plan, or when expanding the analysis scope based on findings.

    Args:
        new_step: The new step to add to the plan. Should be properly formatted and follow the atomic, sequential pattern (e.g., "6. Validate results against business logic").

    Returns:
        str: A formatted string containing the updated plan content wrapped in XML tags.

    Example:
        add_plan_step("4. Validation step: Cross-check MRR calculations against transaction totals")
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
            create_plan_steps,
            update_plan,
            add_plan_step,
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


def setup_workspace(data_dir: Path | str, workspace_dir: Path | str, delete_existing: bool = False):
    data_dir = Path(data_dir)
    workspace_dir = Path(workspace_dir)
    if workspace_dir.exists() and delete_existing:
        shutil.rmtree(workspace_dir)
        logger.info(f"Removed {workspace_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace_data_dir = workspace_dir / "data"
    if data_dir.exists():
        try:
            shutil.copytree(data_dir, workspace_data_dir, dirs_exist_ok=True)
            logger.success(f"Copied data from {data_dir} to {workspace_data_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to copy data from {data_dir} to {workspace_data_dir}: {e}") from e


async def run_task(task: str, workspace_id: str = "1", thread_id: str = "1") -> TaskResult:
    main_dir = MODULE_DIR.parent.parent
    workspace_dir = main_dir / f"workspaces/{workspace_id}"
    thread_dir = workspace_dir / f"threads/{thread_id}"
    setup_workspace(main_dir / "operateai_scenario1_data", workspace_dir, delete_existing=True)
    user_prompt = f"Task: {task}"
    while True:
        output = await thread(
            thread_dir=thread_dir,
            user_prompt=user_prompt,
            use_excel_tools=False,
            use_thinking=False,
            use_memory=False,
        )
        if isinstance(output, TaskResult):
            Path(thread_dir / "task_result.md").write_text(output.message)
            return output
        logger.info(f"Output: {output}")
        user_prompt = (
            "The user is not available to provide input for this task. "
            "Please proceed independently by making reasonable assumptions where needed.\n"
            "Compile all your assumptions, any issues encountered, and your solutions in your final comprehensive response.\n"
            "Push through to completion even if you think you need user clarification - just document what you assumed and why."
        )


if __name__ == "__main__":
    res = asyncio.run(
        run_task(
            "Calculate the Average Revenue Per User (ARPU) for customers who were active in January 2023, broken down by industry segment.",
        )
    )
