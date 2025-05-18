import asyncio
import json
import re
import tempfile
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import duckdb
import logfire
import pandas as pd
from loguru import logger
from pydantic import BaseModel, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessagesTypeAdapter, ToolReturnPart
from pydantic_ai.models.fallback import FallbackModel

logfire.configure()

SQL_TIMEOUT_SECONDS = 5
PREVIEW_ROWS = 10


class Success(BaseModel):
    message: Any


class GraphDeps(BaseModel):
    data_dir: str
    analysis_dir: str
    results_dir: str
    stop: bool = False

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


def add_dirs(ctx: RunContext[GraphDeps]) -> str:
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


def list_csv_files(ctx: RunContext[GraphDeps]) -> str:
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


class RunSQLResult(BaseModel):
    """
    A lightweight reference to a (possibly large) query result on disk.
    """

    file_path: str
    preview: list[dict[str, Any]]
    row_count: int


class WriteSheetResult(BaseModel):
    file_path: str


def run_sql(
    ctx: RunContext[GraphDeps], query: str, preview_rows: int = 2, file_name: str | None = None
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

        file_path = (
            Path(ctx.deps.analysis_dir) / file_name
            if file_name
            else temp_file_path(file_dir=ctx.deps.analysis_dir)
        )
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


def calculate_sum(values: list[float]) -> float:
    """Calculate the sum of a list of values."""
    return sum(values)


def calculate_difference(num1: float, num2: float) -> float:
    """Calculate the difference between two numbers by subtracting `num1` from `num2`"""
    return num2 - num1


def calculate_mean(values: list[float]) -> float:
    """Calculate the mean of a list of values."""
    return sum(values) / len(values)


def write_sheet_from_file(
    ctx: RunContext[GraphDeps], file_path: str, sheet_name: str, workbook_name: str | None = None
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

    workbook_path = (
        Path(ctx.deps.results_dir) / workbook_name
        if workbook_name
        else temp_file_path(file_dir=ctx.deps.results_dir)
    )
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


def user_interaction(ctx: RunContext[GraphDeps], message: str) -> str:
    """
    Interacts with the user. Could be:
    - A question
    - A progress update
    - An assumption made that needs to be validated
    - A request for clarification
    - Anything else needed from the user to proceed
    """
    res = input(f"{message}> ")
    ctx.deps.stop = res.strip().lower() == "q"
    return res


fallback_model = FallbackModel(
    "openai:gpt-4.1", "anthropic:claude-3-5-sonnet-latest", "openai:gpt-4.1-mini", "google-gla:gemini-2.0-flash"
)
agent = Agent(
    model=fallback_model,
    instructions=[Path("./src/operate_ai/prompts/cfo.md").read_text(), add_current_time, add_dirs],
    deps_type=GraphDeps,
    retries=10,
    tools=[
        list_csv_files,
        run_sql,
        write_sheet_from_file,
        user_interaction,
        calculate_sum,
        calculate_difference,
        calculate_mean,
    ],
    mcp_servers=[thinking_server],
    instrument=True,
)


async def new_thread(user_prompt: str):
    deps = GraphDeps(
        data_dir="/Users/hamza/dev/operate-ai/operateai_scenario1_data",
        analysis_dir="/Users/hamza/dev/operate-ai/operateai_scenario1_analysis",
        results_dir="/Users/hamza/dev/operate-ai/operateai_scenario1_results",
        stop=False,
    )
    message_history_path = Path(deps.results_dir) / "message_history.json"
    while True:
        review = None
        async with agent.run_mcp_servers():
            async with agent.iter(
                user_prompt=user_prompt,
                deps=deps,
                message_history=ModelMessagesTypeAdapter.validate_json(message_history_path.read_bytes())
                if message_history_path.exists()
                else None,
            ) as run:
                async for node in run:
                    message_history_path.write_bytes(
                        ModelMessagesTypeAdapter.dump_json(run.ctx.state.message_history)
                    )
                    print("\n\n", node, "\n\n")
                    if deps.stop:
                        break
                    if agent.is_model_request_node(node):
                        for part in node.request.parts:
                            if isinstance(part, ToolReturnPart):
                                if part.tool_name == "run_sql":
                                    file_path = Path(part.content.file_path).expanduser().resolve()
                                    review_prompt = (
                                        f"Ran SQL query: {file_path.with_suffix('.sql')}\n"
                                        f"Results are in the file: {file_path}\n"
                                        "Please Review. Press Enter to continue. Press Q to quit."
                                    )
                                    review = input(f"{review_prompt} > ")
                                    if review.strip():
                                        break
                                elif part.tool_name == "write_sheet_from_file":
                                    file_path = Path(part.content.file_path).expanduser().resolve()
                                    review_prompt = (
                                        f"Wrote to workbook: {file_path}\n"
                                        "Please Review. Press Enter to continue. Press Q to quit."
                                    )
                                    review = input(f"{review_prompt} > ")
                                    if review.strip():
                                        break

        message_history_path.write_bytes(ModelMessagesTypeAdapter.dump_json(run.ctx.state.message_history))
        if deps.stop or run.result is None:
            break
        user_prompt = review or input(f"{run.result.output} > ")
        if user_prompt.strip().lower() in ["q", ""]:
            break


if __name__ == "__main__":
    user_prompt = "How many customers in 2023 had a monthly plan?"
    asyncio.run(new_thread(user_prompt=user_prompt))


"""
This is all of my code for my CFO AGENT
Now, I have to actually make it into a fastapi + streamlit app
I will use sqlite3 for the database.

# Tables

## Workspace
no users. one workspace per company. multiple threads per workspace.
the workspace will have an id, a name, a created_at, updated_at, and a workspace_dir
the workspace_dir will have the data dir and the threads dir. like workspace_dir/data and workspace_dir/threads
data will be uploaded to the workspace and shared across threads.

## Thread
a thread will have an id, a workspace id, a created_at, an updated_at, and a thread_dir
the thread_dir will have the analysis dir, the results dir, the message_history.json, a memory.json. this thread dir will be like workspace_dir/threads/<thread_id>
a person can reuse/continue a thread whenever they want. like selecting a previous chat from the list of chats. so no point in having a status field.
no thread specific data is uploaded by the user.
the data in the analysis and results dir is thread specific.

# Flow

- a person logs into a workspace
- they can either create a new thread or continue an existing thread
- the user sends a prompt
- if it's new thread, a new thread is created with an empty thread_dir
- if it's an existing thread, the thread_dir is loaded from the workspace_dir/threads/<thread_id>
- the agent runs
- if the agent decides to call the user_interaction tool, the flow stops and the user is asked to provide a response
- the flow wont continue until the user provides a response
- the response could be text entered into a box, or the user clicking 'continue' or the user clicking the read square stop button
- after the agent uses the run_sql tool, the flow stops and the user is asked to review the results
- the flow wont continue until the user provides a response
- the response could be text entered into a box, or the user clicking 'continue' or the user clicking the read square stop button
- after the agent uses the write_sheet_from_file tool, the flow stops and the user is asked to review the results
- the flow wont continue until the user provides a response
- the response could be text entered into a box, or the user clicking 'continue' or the user clicking the read square stop button
- in any other case/tool call, the flow continues without any user intervention
- once the agent is done, it sends a final response to the user
- but even then, unless the user starts a new thread (which for now in our code is just entering q), the thread is still alive
- so if the user sends a new message, the thread continues

the complexity is, how do we handle this user interaction?
becasue once a new thread is cerated using post and the user pormpt, we go back to the client
now, the server decides to engage the user in the scenarios we've defined above.
then, the server has sent a request to the user, but then it gets the request returns. we still need to wait for the user to respond







"""
