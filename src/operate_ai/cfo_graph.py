import asyncio
import json
import re
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import duckdb
import logfire
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.persistence.file import FileStatePersistence

logfire.configure()

SQL_TIMEOUT_SECONDS = 5
PREVIEW_ROWS = 10


def user_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


class Success(BaseModel):
    message: Any


@dataclass
class GraphState:
    message_history: list[ModelMessage] = field(default_factory=list)


class AgentDeps(BaseModel):
    data_dir: str
    analysis_dir: str
    results_dir: str

    @model_validator(mode="after")
    def create_dirs(self: Self) -> Self:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        return self


thinking_server = MCPServerStdio(command="npx", args=["-y", "@modelcontextprotocol/server-sequential-thinking"])


def extract_csv_paths(sql_query: str) -> list[str]:
    """Extract CSV file paths from SQL query."""

    # Find read_csv calls with single or multiple CSVs
    read_csv_pattern = r"read_csv\(\s*'([^']*\.csv)'\s*\)|read_csv\(\s*\[([^\]]*)\]\s*\)"
    read_csv_matches = re.findall(read_csv_pattern, sql_query, re.IGNORECASE)

    # Extract paths from both single and multiple CSV cases
    paths: list[Any] = []
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
        f"<results_dir>\n{str(Path(ctx.deps.results_dir).expanduser().resolve())}\n</results_dir>"
    )


def add_current_time() -> str:
    return f"<current_time>\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n</current_time>"


def preview_csv(df: Path | str | pd.DataFrame, num_rows: int = PREVIEW_ROWS) -> list[dict[Hashable, Any]]:
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
    csv_files = [str(file.expanduser().resolve()) for file in Path(ctx.deps.data_dir).glob("*.csv")]
    res = "\n<available_csv_files>\n"
    for file in csv_files:
        res += file + "\n"
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

    query: str = Field(description="The SQL query to execute.")
    preview_rows: int = Field(description="Number of rows to preview. 2-5 should be enough for most queries.")
    file_name: str | None = Field(
        description="Descriptive name of the file based on the query to save the result to in the `analysis_dir`."
    )


class RunSQLResult(BaseModel):
    """
    A lightweight reference to a (possibly large) query result on disk.
    """

    file_path: str
    preview: list[dict[str, Any]]
    row_count: int


def run_sql(analysis_dir: str, query: str, preview_rows: int = 2, file_name: str | None = None) -> RunSQLResult:
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
            file_path = Path(path)
            if not file_path.exists():
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

        df: pd.DataFrame = duckdb.sql(query_replaced).df()  # type: ignore

        file_path = Path(analysis_dir) / file_name if file_name else temp_file_path(file_dir=analysis_dir)
        file_path = file_path.expanduser().resolve().with_suffix(".csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(file_path, index=False)
        file_path.with_suffix(".sql").write_text(query_replaced)

        return RunSQLResult(
            file_path=str(file_path),
            preview=preview_csv(df=df, num_rows=preview_rows),  # type: ignore
            row_count=len(df),
        )

    try:
        return asyncio.run(asyncio.wait_for(_run_with_timeout(), timeout=SQL_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        raise ModelRetry(f"SQL query execution timed out after {SQL_TIMEOUT_SECONDS} seconds")
    except FileNotFoundError as e:
        raise ModelRetry(str(e))
    except Exception as e:
        raise ModelRetry(str(e))


@dataclass
class RunSQLNode(BaseNode[GraphState, AgentDeps, str]):
    query: str
    preview_rows: int = 2
    file_name: str | None = None

    async def run(self, ctx: GraphRunContext[GraphState, AgentDeps]) -> End[str]:
        sql_result = run_sql(
            analysis_dir=ctx.deps.analysis_dir,
            query=self.query,
            preview_rows=self.preview_rows,
            file_name=self.file_name,
        )
        ctx.state.message_history.append(user_message(sql_result.model_dump_json()))
        file_path = Path(sql_result.file_path).expanduser().resolve()
        input_prompt = (
            f"Ran SQL query: {file_path.with_suffix('.sql')}\n"
            f"Results are in the file: {file_path}\n"
            "Please Review. Press Enter to continue. Press Q to quit."
        )
        return End(data=input_prompt)


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


class WriteSheetResult(BaseModel):
    file_path: str


def write_sheet_from_file(
    results_dir: str, file_path: str, sheet_name: str, workbook_name: str | None = None
) -> WriteSheetResult:
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

    workbook_path = Path(results_dir) / workbook_name if workbook_name else temp_file_path(file_dir=results_dir)
    wb_path = workbook_path.expanduser().resolve().with_suffix(".xlsx")
    wb_path.parent.mkdir(parents=True, exist_ok=True)

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
    return WriteSheetResult(file_path=str(wb_path))


@dataclass
class WriteSheetNode(BaseNode[GraphState, AgentDeps, str]):
    file_path: str
    sheet_name: str
    workbook_name: str | None = None

    async def run(self, ctx: GraphRunContext[GraphState, AgentDeps]) -> End[str]:
        write_sheet_result = write_sheet_from_file(
            results_dir=ctx.deps.results_dir,
            file_path=self.file_path,
            sheet_name=self.sheet_name,
            workbook_name=self.workbook_name,
        )
        ctx.state.message_history.append(user_message(write_sheet_result.model_dump_json()))
        file_path = Path(write_sheet_result.file_path).expanduser().resolve()
        input_prompt = f"Wrote to workbook: {file_path}\nPlease Review. Press Enter to continue. Press Q to quit."
        return End(data=input_prompt)


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


def user_interaction(message: str) -> str:
    """
    Interacts with the user. Could be:
    - A question
    - A progress update
    - An assumption made that needs to be validated
    - A request for clarification
    - Anything else needed from the user to proceed
    """
    res = input(f"{message}> ")
    return res


class TaskResult(BaseModel):
    """
    A task result.
    """

    message: str = Field(description="The final response to the user.")


fallback_model = FallbackModel(
    "openai:gpt-4.1", "anthropic:claude-3-5-sonnet-latest", "openai:gpt-4.1-mini", "google-gla:gemini-2.0-flash"
)
agent = Agent(
    model=fallback_model,
    instructions=[
        Path("/Users/hamza/dev/operate-ai/src/operate_ai/prompts/cfo.md").read_text(),
        add_current_time,
        add_dirs,
    ],
    deps_type=AgentDeps,
    retries=10,
    tools=[list_csv_files, calculate_sum, calculate_difference, calculate_mean],
    mcp_servers=[thinking_server],
    output_type=TaskResult | UserInteraction | RunSQL | WriteSheetFromFile,  # type: ignore
    instrument=True,
)


@dataclass
class RunAgentNode(BaseNode[GraphState, AgentDeps, str]):
    user_prompt: str

    async def run(self, ctx: GraphRunContext[GraphState, AgentDeps]) -> RunSQLNode | WriteSheetNode | End[str]:
        async with agent.run_mcp_servers():
            res = await agent.run(
                user_prompt=self.user_prompt, deps=ctx.deps, message_history=ctx.state.message_history
            )
            ctx.state.message_history += res.new_messages()
            if isinstance(res.output, RunSQL):
                return RunSQLNode(
                    query=res.output.query, preview_rows=res.output.preview_rows, file_name=res.output.file_name
                )
            elif isinstance(res.output, WriteSheetFromFile):
                return WriteSheetNode(
                    file_path=res.output.file_path,
                    sheet_name=res.output.sheet_name,
                    workbook_name=res.output.workbook_name,
                )
            elif isinstance(res.output, (UserInteraction, TaskResult)):
                return End(data=res.output.message)
            else:
                return End(data=str(res.output))


graph = Graph(nodes=(RunAgentNode, RunSQLNode, WriteSheetNode))
graph.mermaid_save(Path("cfo_graph.jpg"), direction="LR")


def setup_thread_dirs(thread_dir: Path | str) -> tuple[str, str, str]:
    thread_dir = Path(thread_dir).expanduser().resolve()
    thread_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = thread_dir.parent.parent.expanduser().resolve()
    data_dir = workspace_dir / "data"
    analysis_dir = thread_dir / "analysis"
    results_dir = thread_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir), str(analysis_dir), str(results_dir)


async def thread(thread_dir: Path | str, user_prompt: str) -> str:
    data_dir, analysis_dir, results_dir = setup_thread_dirs(thread_dir)
    persistence = FileStatePersistence(json_file=Path(thread_dir) / "state.json")
    logger.info(f"Persistence: {await persistence.load_all()}")
    deps = AgentDeps(data_dir=data_dir, analysis_dir=analysis_dir, results_dir=results_dir)
    state = GraphState()
    res = await graph.run(
        start_node=RunAgentNode(user_prompt=user_prompt), state=state, deps=deps, persistence=persistence
    )
    return res.output


async def run_app():
    input_prompt = ""
    while True:
        # user_prompt = "How many customers in 2023 had a monthly plan?"
        thread_dir = Path("/Users/hamza/dev/operate-ai/workspaces/1/threads/1")
        user_prompt = input(input_prompt)
        if user_prompt.strip().lower() in ["q", ""]:
            return
        input_prompt = await thread(thread_dir=thread_dir, user_prompt=user_prompt)


# if __name__ == "__main__":
#     asyncio.run(run_app())
