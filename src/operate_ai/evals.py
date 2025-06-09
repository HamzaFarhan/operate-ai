from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.models import KnownModelName
from pydantic_ai.models.fallback import FallbackModel
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from operate_ai.cfo_graph import Query, run_task

DELTA = 0.05


@dataclass
class EqEvaluator[OutputT](Evaluator[Query[OutputT], OutputT]):
    async def evaluate(self, ctx: EvaluatorContext[Query[OutputT], OutputT]) -> bool:
        if isinstance(ctx.output, (int, float)) and isinstance(ctx.expected_output, (int, float)):
            output = round(ctx.output, 2) if isinstance(ctx.output, float) else ctx.output
            expected = (
                round(ctx.expected_output, 2) if isinstance(ctx.expected_output, float) else ctx.expected_output
            )
            return abs(output - expected) < DELTA
        return ctx.output == ctx.expected_output


async def eval_task[OutputT](
    query: Query[OutputT],
    workspace_dir: Path | str,
    name: str | None = None,
    model: KnownModelName | FallbackModel | None = None,
    use_excel_tools: bool = False,
    use_thinking: bool = False,
    use_memory: bool = False,
    step_limit: int = 10,
) -> OutputT:
    res = await run_task(
        query=query,
        workspace_dir=workspace_dir,
        name=name,
        model=model,
        use_excel_tools=use_excel_tools,
        use_thinking=use_thinking,
        use_memory=use_memory,
        step_limit=step_limit,
    )
    return res  # type: ignore[return-value]
