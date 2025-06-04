from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from loguru import logger
from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from operate_ai.cfo_graph import TaskResult, thread

OutputsT = TypeVar("OutputsT")
PrevQueriesT = TypeVar("PrevQueriesT")


@dataclass
class PrevQuery[PrevQueriesT]:
    query: str
    result: PrevQueriesT


@dataclass
class Query[OutputsT, PrevQueriesT]:
    query: str
    output_type: type[OutputsT]
    prev_queries: list[PrevQuery[PrevQueriesT]] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        res = f"Task: {self.query}"
        if self.prev_queries:
            res += "\n\nPrevious Queries:\n"
            for prev_query in self.prev_queries:
                res += f"Task: {prev_query.query}\nResult: {prev_query.result}\n"
        return res


@dataclass
class EqEvaluator(Evaluator[Query[OutputsT, PrevQueriesT], OutputsT]):
    async def evaluate(self, ctx: EvaluatorContext[Query[OutputsT, PrevQueriesT], OutputsT]) -> bool:
        return ctx.output == ctx.expected_output


async def eval_task[OutputT, PrevQueriesT](query: Query[OutputT, PrevQueriesT]) -> OutputT:
    thread_dir = Path("/Users/hamza/dev/operate-ai/workspaces/1/threads/1")
    output = None
    limit = 10
    user_prompt = query.prompt
    while not isinstance(output, TaskResult) and limit > 0:
        output = await thread(thread_dir=thread_dir, user_prompt=user_prompt, do_user_interaction=False)
        logger.info(f"Output: {output}")
        user_prompt = "go on"
        limit -= 1
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
