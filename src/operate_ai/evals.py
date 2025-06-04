from dataclasses import dataclass
from typing import TypeVar

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

OutputsT = TypeVar("OutputsT")


@dataclass
class Query[OutputsT]:
    query: str
    output_type: type[OutputsT]


@dataclass
class EqEvaluator(Evaluator[Query[OutputsT], OutputsT]):
    async def evaluate(self, ctx: EvaluatorContext[Query[OutputsT], OutputsT]) -> bool:
        return ctx.output == ctx.expected_output
