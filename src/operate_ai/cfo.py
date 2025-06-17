from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, TypedDict
from uuid import uuid4

import duckdb
import logfire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.persistence.file import FileStatePersistence

load_dotenv()

logfire.configure()

SQL_TIMEOUT_SECONDS = 5
PREVIEW_ROWS = 10
MAX_RETRIES = 10
MEMORY_FILE_PATH = os.getenv("MEMORY_FILE_PATH", "memory.json")


def user_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


class Success(BaseModel):
    message: Any


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: RunSQLResult | WriteDataToExcelResult | str
    state_path: str


class AgentDeps(BaseModel):
    data_dir: str
    analysis_dir: str
    results_dir: str
    state_path: str

    @model_validator(mode="after")
    def create_dirs(self: Self) -> Self:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class GraphState:
    chat_messages: list[ChatMessage] = field(default_factory=list)
    message_history: list[ModelMessage] = field(default_factory=list)
    run_sql_attempts: int = 0
    write_sheet_attempts: int = 0


@dataclass
class GraphDeps:
    agent: Agent[AgentDeps, str]
    agent_deps: AgentDeps


def extract_csv_paths(sql_query: str) -> list[str]:
    """Extract CSV file paths from SQL query."""

    # Find read_csv calls with single or multiple CSVs
    read_csv_pattern = r"read_csv\(\s*'([^']*\.csv)'\s*\)|read_csv\(\s*\[([^\]]*)\]\s*\)"
    read_csv_matches = re.findall(read_csv_pattern, sql_query, re.IGNORECASE)

    # Extract paths from both single and multiple CSV cases
    paths: list[str] = []
    for single_path, multiple_paths in read_csv_matches:
        if single_path:
            paths.append(single_path)
        elif multiple_paths:
            # Extract individual paths from the array
            paths.extend(re.findall(r"'([^']*\.csv)'", multiple_paths, re.IGNORECASE))

    return paths


def add_dirs(ctx: RunContext[AgentDeps]) -> str:
    return (
        f"<data_dir>\n{str(Path(ctx.deps.data_dir).expanduser().resolve())}\n</data_dir>\n"
        f"<analysis_dir>\n{str(Path(ctx.deps.analysis_dir).expanduser().resolve())}\n</analysis_dir>\n"
        f"<results_dir>\n{str(Path(ctx.deps.results_dir).expanduser().resolve())}\n</results_dir>\n\n"
        "Write all excel workbooks in the `results_dir`. Use the full absolute path."
    )


def add_current_time() -> str:
    return f"<current_time>\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n</current_time>"


def preview_csv(df: Path | pd.DataFrame, num_rows: int = PREVIEW_ROWS) -> list[dict[Hashable, Any]]:
    """
    Returns the first `num_rows` rows of the specified CSV file.
    Should be used for previewing the data.
    """
    if isinstance(df, pd.DataFrame):
        return df.head(num_rows).to_dict(orient="records")  # type: ignore
    return pd.read_csv(df).head(num_rows).to_dict(orient="records")  # type: ignore


def list_csv_files(ctx: RunContext[AgentDeps]) -> str:
    """
    Lists all available csv files.
    """
    csv_files = [file.expanduser().resolve() for file in Path(ctx.deps.data_dir).glob("*.csv")]
    res = "\n<available_csv_files>\n"
    for file in csv_files:
        res += str(file) + "\n"
        res += f"First {PREVIEW_ROWS} rows for preview:\n"
        res += json.dumps(preview_csv(df=file, num_rows=PREVIEW_ROWS)) + "\n\n"
    return res.strip() + "\n</available_csv_files>"


def temp_file_path(file_dir: Path | str | None = None) -> Path:
    file_dir = file_dir or tempfile.gettempdir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(file_dir) / f"ai_cfo_result_{Path(file_dir).name}_{ts}.csv"


class RunSQL(BaseModel):
    """
    1. Runs an SQL query on csv file(s) using duckdb
    2. Writes the full result to disk
    3. Returns a ResultHandle with a small in-memory preview.
    """

    purpose: str = Field(
        description=(
            "Describe what you want to achieve with the query and why. Relate it to the main task.\n"
            "This will be a user-facing message, so no need to be too technical or mention the internal errors faced if any."
        )
    )
    query: str = Field(
        description=(
            "The SQL query to execute."
            "When reading a csv, use the FULL path with the data_dir included.\n"
            "Example: 'select names from read_csv('workspaces/1/data/orders.csv')'"
        )
    )
    preview_rows: int = Field(description="Number of rows to preview. 2-5 should be enough for most queries.")
    file_name: str | None = Field(
        description=(
            "Descriptive name of the file based on the query to save the result to in the `analysis_dir`.\n"
            "If None, a file path will be created in the `analysis_dir` based on the current timestamp.\n"
            "Example: 'customers_joined_in_2023.csv'\n"
            "'.csv' is optional. It will be added automatically if not provided."
        )
    )


class RunSQLResult(BaseModel):
    """
    A lightweight reference to a (possibly large) query result on disk.
    """

    purpose: str | None = None
    sql_path: str
    csv_path: str
    row_count: int
    preview: list[dict[str, Any]]


async def run_sql(
    analysis_dir: str, query: str, preview_rows: int = 2, file_name: str | None = None
) -> RunSQLResult:
    """
    1. Runs an SQL query on csv file(s) using duckdb
    2. Writes the full result to disk
    3. Returns a ResultHandle with a small in-memory preview.

    Parameters
    ----------
    query : str
        The SQL query to execute.
    preview_rows : int, default 3
        Number of rows to preview. 2-5 should be enough for most queries.
    file_name : str | None, default None
        Descriptive name of the file based on the query to save the result to in the `analysis_dir`.
        If None, a file path will be created in the `analysis_dir` based on the current timestamp.

    Returns
    -------
    RunSQLResult
        A lightweight reference to the result on disk.
    """

    def check_csv_files_exist(paths: list[str]) -> None:
        """Check if all CSV files in the paths exist."""
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"CSV file not found: {path}")

    async def _run_with_timeout():
        query_replaced = query.replace('"', "'").strip()
        logger.info(f"Running SQL query: {query_replaced}")

        # Extract and check CSV files before executing query
        csv_paths = extract_csv_paths(query_replaced)
        if not csv_paths:
            logger.warning("No CSV files found in SQL query")
        else:
            logger.info(f"Found CSV files in query: {csv_paths}")
            check_csv_files_exist(csv_paths)

        try:
            df: pd.DataFrame = duckdb.sql(query_replaced).df()  # type: ignore
        except Exception as e:
            # Handle transaction rollback for DuckDB transaction errors
            try:
                duckdb.sql("ROLLBACK")
                logger.info("Rolled back DuckDB transaction after error")
            except Exception:
                pass  # Rollback might fail if no transaction is active
            raise e

        file_path = (
            Path(analysis_dir) / Path(file_name).name if file_name else temp_file_path(file_dir=analysis_dir)
        )
        file_path = file_path.expanduser().resolve().with_suffix(".csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            file_path = file_path.with_stem(f"{file_path.stem}_{uuid4()}")

        df.to_csv(file_path, index=False)
        file_path.with_suffix(".sql").write_text(query_replaced)

        return RunSQLResult(
            sql_path=str(file_path.with_suffix(".sql")),
            csv_path=str(file_path),
            preview=preview_csv(df=df, num_rows=preview_rows),  # type: ignore
            row_count=len(df),
        )

    try:
        return await asyncio.wait_for(_run_with_timeout(), timeout=SQL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise ModelRetry(f"SQL query execution timed out after {SQL_TIMEOUT_SECONDS} seconds")
    except FileNotFoundError as e:
        raise ModelRetry(str(e))
    except Exception as e:
        raise ModelRetry(str(e))


@dataclass
class RunSQLNode(BaseNode[GraphState, GraphDeps, RunSQLResult]):
    """Run 'run_sql' tool."""

    docstring_notes = True
    purpose: str
    query: str
    preview_rows: int = 2
    file_name: str | None = None

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> RunAgentNode | End[RunSQLResult]:
        try:
            sql_result = await run_sql(
                analysis_dir=ctx.deps.agent_deps.analysis_dir,
                query=self.query,
                preview_rows=self.preview_rows,
                file_name=self.file_name,
            )
            sql_result.purpose = self.purpose
            ctx.state.chat_messages.append(
                {"role": "assistant", "content": sql_result, "state_path": ctx.deps.agent_deps.state_path}
            )
            ctx.state.message_history.append(user_message(sql_result.model_dump_json(exclude={"purpose"})))
            return End(data=sql_result)
        except Exception as e:
            logger.exception("Error running SQL query")
            ctx.state.message_history.append(user_message(content=f"Error in RunSQL: {str(e)}"))
            ctx.state.run_sql_attempts += 1
            return RunAgentNode(user_prompt="")


def calculate_sum(values: list[float]) -> float:
    """Calculate the sum of a list of values."""
    return sum(values)


def calculate_difference(num1: float, num2: float) -> float:
    """Calculate the difference between two numbers by subtracting `num1` from `num2`"""
    return num2 - num1


def calculate_mean(values: list[float]) -> float:
    """Calculate the mean of a list of values."""
    return sum(values) / len(values)


class WriteSheetFromFile(BaseModel):
    """
    Creates an excel workbook and writes a sheet to it.
    1. Reads from the `file_path` of the previously received RunSQLResult
    2. Appends to / creates `workbook_path`.
    """

    file_path: str = Field(
        description="Path to the file to read from. Almost always a `file_path` of a previously received RunSQLResult."
    )
    sheet_name: str = Field(description="The name of the sheet to write to.")
    workbook_name: str | None = Field(description="Name of the workbook to append to / create in `results_dir`.")


class WriteDataToExcelResult(BaseModel):
    file_path: str


def write_sheet_from_file(
    results_dir: str, file_path: str, sheet_name: str, workbook_name: str | None = None
) -> WriteDataToExcelResult:
    """
    Creates an excel workbook and writes a sheet to it.
    1. Reads from the `file_path` of the previously received RunSQLResult
    2. Appends to / creates `workbook_path`.

    Parameters
    ----------
    file_path : str
        Path to the file to read from. Almost always a `file_path` of a previously received RunSQLResult.
    sheet_name : str
        The name of the sheet to write to.
    workbook_name : str | None, default None
        Name of the workbook to append to / create in `results_dir`.
        If None, a path will be created in the `results_dir` using the current timestamp.

    Returns
    -------
    WriteSheetResult
    """

    workbook_path = (
        Path(results_dir) / Path(workbook_name).name if workbook_name else temp_file_path(file_dir=results_dir)
    )
    wb_path = workbook_path.expanduser().resolve().with_suffix(".xlsx")
    wb_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to workbook: {wb_path}")

    df: pd.DataFrame = pd.read_csv(Path(file_path).expanduser().resolve())  # type: ignore

    try:
        mode = "w"
        if_sheet_exists = None
        if wb_path.exists():
            mode = "a"
            if_sheet_exists = "replace"
        with pd.ExcelWriter(wb_path, mode=mode, engine="openpyxl", if_sheet_exists=if_sheet_exists) as w:
            df.to_excel(w, sheet_name=sheet_name[:31] or "Sheet1", index=False)  # type: ignore

        _ = pd.read_excel(wb_path)  # type: ignore
        logger.info(f"Successfully wrote to workbook: {wb_path}")
    except Exception as e:
        logger.warning(f"Failed to write to workbook: {wb_path}")
        raise ModelRetry(
            (
                f"Got this error when trying to read/write the workbook.\n"
                "I have reverted the step. Try again:\n"
                f"{str(e)}"
            )
        )
    return WriteDataToExcelResult(file_path=str(wb_path))


@dataclass
class WriteSheetNode(BaseNode[GraphState, GraphDeps, WriteDataToExcelResult]):
    """Run 'write_sheet_from_file' tool."""

    docstring_notes = True
    file_path: str
    sheet_name: str
    workbook_name: str | None = None

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> RunAgentNode | End[WriteDataToExcelResult]:
        try:
            write_sheet_result = write_sheet_from_file(
                results_dir=ctx.deps.agent_deps.results_dir,
                file_path=self.file_path,
                sheet_name=self.sheet_name,
                workbook_name=self.workbook_name,
            )
            ctx.state.chat_messages.append(
                {"role": "assistant", "content": write_sheet_result, "state_path": ctx.deps.agent_deps.state_path}
            )
            ctx.state.message_history.append(user_message(write_sheet_result.model_dump_json()))
            return End(data=write_sheet_result)
        except Exception as e:
            logger.exception("Error writing sheet")
            ctx.state.message_history.append(user_message(content=f"Error in WriteSheet: {str(e)}"))
            ctx.state.write_sheet_attempts += 1
            return RunAgentNode(user_prompt="")


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


@dataclass
class UserInteractionNode(BaseNode[GraphState, GraphDeps, str]):
    """Pass to End."""

    docstring_notes = True
    message: str

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> End[str]:  # noqa: ARG002
        ctx.state.chat_messages.append(
            {"role": "assistant", "content": self.message, "state_path": ctx.deps.agent_deps.state_path}
        )
        return End(data=self.message)


class TaskResult(BaseModel):
    """
    A task result.
    """

    message: str = Field(description="The final response to the user.")


@dataclass
class TaskResultNode(BaseNode[GraphState, GraphDeps, str]):
    """Pass to End."""

    docstring_notes = True
    message: str

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> End[str]:  # noqa: ARG002
        ctx.state.chat_messages.append(
            {"role": "assistant", "content": self.message, "state_path": ctx.deps.agent_deps.state_path}
        )
        return End(data=self.message)


def create_agent(model: Model | KnownModelName, workspace_dir: Path | str) -> Agent[AgentDeps, str]:
    thinking_server = MCPServerStdio(
        command="npx", args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
    )
    memory_server = MCPServerStdio(
        command="uv",
        args=["run", "./memory_mcp.py"],
        env={"MEMORY_FILE_PATH": str(Path(workspace_dir) / Path(MEMORY_FILE_PATH))},
    )
    excel_server = MCPServerStdio(command="uvx", args=["../../../excel-mcp-server", "stdio"])
    return Agent(
        model=model,
        instructions=[
            Path("./prompts/cfo.md").read_text(),
            Path("./prompts/memory.md").read_text(),
            add_current_time,
            add_dirs,
        ],
        deps_type=AgentDeps,
        retries=MAX_RETRIES,
        tools=[list_csv_files, calculate_sum, calculate_difference, calculate_mean],
        mcp_servers=[thinking_server, memory_server, excel_server],
        output_type=TaskResult | WriteDataToExcelResult | UserInteraction | RunSQL,  # type: ignore
        instrument=True,
    )


@dataclass
class RunAgentNode(BaseNode[GraphState, GraphDeps, RunSQLResult | WriteDataToExcelResult | str]):
    """Run the agent."""

    docstring_notes = True
    user_prompt: str

    async def run(
        self, ctx: GraphRunContext[GraphState, GraphDeps]
    ) -> (
        RunSQLNode
        # | WriteSheetNode
        | UserInteractionNode
        | TaskResultNode
        | End[RunSQLResult | WriteDataToExcelResult | str]
    ):
        if self.user_prompt:
            ctx.state.chat_messages.append(
                {"role": "user", "content": self.user_prompt, "state_path": ctx.deps.agent_deps.state_path}
            )
        error_result = End(data="Ran into an error. Please try again.")
        async with ctx.deps.agent.run_mcp_servers():
            res = await ctx.deps.agent.run(
                user_prompt=self.user_prompt,
                deps=ctx.deps.agent_deps,
                message_history=ctx.state.message_history,
            )
        try:
            ctx.state.message_history += res.new_messages()
            if isinstance(res.output, RunSQL):
                if ctx.state.run_sql_attempts < MAX_RETRIES:
                    return RunSQLNode(
                        purpose=res.output.purpose,
                        query=res.output.query,
                        preview_rows=res.output.preview_rows,
                        file_name=res.output.file_name,
                    )
                return error_result
            # elif isinstance(res.output, WriteSheetFromFile):
            #     if ctx.state.write_sheet_attempts < MAX_RETRIES:
            #         return WriteSheetNode(
            #             file_path=res.output.file_path,
            #             sheet_name=res.output.sheet_name,
            #             workbook_name=res.output.workbook_name,
            #         )
            #     return error_result
            elif isinstance(res.output, UserInteraction):
                return UserInteractionNode(message=res.output.message)
            elif isinstance(res.output, WriteDataToExcelResult):
                ctx.state.chat_messages.append(
                    {"role": "assistant", "content": res.output, "state_path": ctx.deps.agent_deps.state_path}
                )
                return End(data=res.output)
            elif isinstance(res.output, TaskResult):
                return TaskResultNode(message=res.output.message)
            else:
                return error_result
        except Exception:
            logger.exception("Error running agent")
            return error_result


graph = Graph(nodes=(RunAgentNode, RunSQLNode, UserInteractionNode, TaskResultNode), name="CFO Graph")
# try:
#     graph.mermaid_save(Path("cfo_graph.jpg"), direction="LR", highlighted_nodes=RunAgentNode)
# except Exception:
#     logger.exception("Error saving graph")


def setup_thread_dirs(thread_dir: Path | str):
    thread_dir = Path(thread_dir).expanduser().resolve()
    thread_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = thread_dir.parent.parent
    data_dir = workspace_dir / "data"
    analysis_dir = thread_dir / "analysis"
    results_dir = thread_dir / "results"
    states_dir = thread_dir / "states"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
    return


async def load_state(
    persistence: FileStatePersistence[GraphState, RunSQLResult | WriteDataToExcelResult | str],
) -> GraphState:
    if snapshot := await persistence.load_next():
        return snapshot.state
    snapshots = await persistence.load_all()
    return (
        GraphState(
            chat_messages=snapshots[-1].state.chat_messages, message_history=snapshots[-1].state.message_history
        )
        if snapshots
        else GraphState()
    )


def get_prev_state_path(thread_dir: Path | str) -> Path | None:
    thread_dir = Path(thread_dir).expanduser().resolve()
    states_dir = thread_dir / "states"
    prev_states = sorted(
        states_dir.glob("*.json"), key=lambda p: datetime.strptime(p.stem, "%d-%m-%Y_%H-%M-%S"), reverse=True
    )
    return prev_states[0] if prev_states else None


async def load_prev_state(thread_dir: Path | str, prev_state_path: Path | str | None = None) -> GraphState:
    prev_state_path = Path(prev_state_path) if prev_state_path else get_prev_state_path(thread_dir)
    logger.info(f"Previous state path: {prev_state_path}")
    if prev_state_path:
        prev_persistence = FileStatePersistence(json_file=prev_state_path.expanduser().resolve())
        prev_persistence.set_graph_types(graph=graph)
        return await load_state(persistence=prev_persistence)
    return GraphState()


async def thread(
    thread_dir: Path | str, user_prompt: str, prev_state_path: Path | str | None = None
) -> RunSQLResult | WriteDataToExcelResult | str:
    thread_dir = Path(thread_dir).expanduser().resolve()
    setup_thread_dirs(thread_dir)
    data_dir = thread_dir.parent.parent / "data"
    analysis_dir = thread_dir / "analysis"
    results_dir = thread_dir / "results"
    states_dir = thread_dir / "states"

    state = await load_prev_state(thread_dir=thread_dir, prev_state_path=prev_state_path)
    persistence_path = states_dir / f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.json"
    persistence = FileStatePersistence(json_file=persistence_path.expanduser().resolve())
    persistence.set_graph_types(graph=graph)
    model = FallbackModel(
        "openai:gpt-4.1",
        "openai:gpt-4.1-mini",
        "gemini-2.5-flash-preview-05-20",  # type: ignore
        "google-gla:gemini-2.0-flash",
        "anthropic:claude-3-5-sonnet-latest",
    )
    agent_deps = AgentDeps(
        data_dir=str(data_dir),
        analysis_dir=str(analysis_dir),
        results_dir=str(results_dir),
        state_path=str(persistence_path),
    )
    agent = create_agent(model=model, workspace_dir=thread_dir.parent.parent)
    res = await graph.run(
        start_node=RunAgentNode(user_prompt=user_prompt),
        state=state,
        deps=GraphDeps(agent=agent, agent_deps=agent_deps),
        persistence=persistence,
    )
    return res.output


async def run_app():
    input_prompt = ""
    while True:
        # user_prompt = "How many customers in 2022 signed up for a monthly plan?"
        thread_dir = Path("/Users/hamza/dev/operate-ai/workspaces/1/threads/1")
        user_prompt = input(f"{input_prompt} > ")
        if user_prompt.strip().lower() in ["q", ""]:
            return
        input_prompt = await thread(thread_dir=thread_dir, user_prompt=user_prompt)


if __name__ == "__main__":
    asyncio.run(run_app())
