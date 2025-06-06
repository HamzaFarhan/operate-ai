from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from loguru import logger
from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from operate_ai.cfo_graph import TaskResult, thread

OutputsT = TypeVar("OutputsT")
PrevQueriesT = TypeVar("PrevQueriesT")

DELTA = 0.025

@dataclass
class PrevQuery[PrevQueriesT]:
    query: str
    result: PrevQueriesT


@dataclass
class Query[OutputsT]:
    query: str
    output_type: type[OutputsT]
    # prev_queries: list[PrevQuery[PrevQueriesT]] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        res = f"Task: {self.query}"
        # if self.prev_queries:
        #     res += "\n\nPrevious Queries:\n"
        #     for prev_query in self.prev_queries:
        #         res += f"Task: {prev_query.query}\nResult: {prev_query.result}\n"
        return res


@dataclass
class EqEvaluator(Evaluator[Query[OutputsT], OutputsT]):
    async def evaluate(self, ctx: EvaluatorContext[Query[OutputsT], OutputsT]) -> bool:
        if isinstance(ctx.output, (int, float)) and isinstance(ctx.expected_output, (int, float)):
            output = round(ctx.output, 2) if isinstance(ctx.output, float) else ctx.output
            expected = (
                round(ctx.expected_output, 2) if isinstance(ctx.expected_output, float) else ctx.expected_output
            )
            return abs(output - expected) < DELTA
        return ctx.output == ctx.expected_output


async def eval_task[OutputT](
    query: Query[OutputT],
    name: str | None = None,
    use_excel_tools: bool = False,
    use_thinking: bool = False,
    use_memory: bool = False,
    limit: int = 10,
) -> OutputT:
    thread_dir = Path(f"/Users/hamza/dev/operate-ai/workspaces/1/threads/{name or uuid4()}")
    output = None
    user_prompt = query.prompt
    step = 0
    while not isinstance(output, TaskResult) and step < limit:
        output = await thread(
            thread_dir=thread_dir,
            user_prompt=user_prompt,
            do_user_interaction=False,
            use_excel_tools=use_excel_tools,
            use_thinking=use_thinking,
            use_memory=use_memory,
        )
        logger.info(f"Output: {output}")
        user_prompt = "go on"
        step += 1
    if output is None:
        raise ValueError("Output is None")
    typer_agent = Agent(
        name="typer",
        model="openai:gpt-4.1-nano",
        output_type=query.output_type,
        instructions=f"Convert the result into this format: {query.output_type}",
        instrument=True,
    )
    user_prompt = (
        f"Task: {query.query}\n\nResult: {output.model_dump_json() if not isinstance(output, str) else output}"
    )
    res = await typer_agent.run(user_prompt=user_prompt)
    return res.output
