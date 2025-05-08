from textwrap import dedent
from typing import Literal

from loguru import logger

THINKING_INSTRUCTIONS = dedent(
    """\
    You have access to the `think` and `analyze` tools to work through problems step-by-step and structure your thought process. You must ALWAYS `think` before making a tool call or generating an answer.

    1. **Think** (scratchpad):
        - Purpose: Use the `think` tool as a scratchpad to break down complex problems, outline steps, and decide on immediate actions within your reasoning flow. Use this to structure your internal monologue.
        - Usage: Call `think` multiple times to build a chain of thought. Detail your reasoning for each step and specify the intended action (e.g., "make a tool call", "perform calculation", "ask clarifying question").
        - You must always `think` before making a tool call or generating an answer.

    2. **Analyze** (evaluation):
        - Purpose: Evaluate the result of a think step or tool call. Assess if the result is expected, sufficient, or requires further investigation.
        - Usage: Call `analyze` after a `think` step or a tool call. Determine the `next_action` based on your analysis: `continue`, `validate` (seek external confirmation/validation if possible), or `final_answer` (ready to conclude).
        - Also note your reasoning about whether it's correct/sufficient.

    ## IMPORTANT GUIDELINES
    - **Always Think First:** You MUST use the `think` tool before any other action (like calling another tool or giving the final answer). This is your first step.
    - **Iterate to Solve:** Use the `think` and `analyze` tools iteratively to build a clear reasoning path. The typical flow is `Think` -> [`Tool Call` if needed] -> [`Analyze` if needed] -> ... -> `final_answer`. Repeat this cycle until you reach a satisfactory conclusion.
    - **Keep Thoughts Internal:** The reasoning steps (thoughts and analyses) are for your internal process only. Do not share them directly with the user.
    - **Conclude Clearly:** When your analysis determines the `next_action` is `final_answer`, provide a concise and accurate final answer to the user."""
)

THINKING_FEW_SHOT_EXAMPLES = dedent(
    """\
    Below are examples demonstrating how to use the `think` and `analyze` tools.

    ### Examples

    **Example 1: Simple Fact Retrieval**

    *User Request:* How many continents are there on Earth?

    *Agent's Internal Process:*

    ```tool_call
    think(
        title="Understand Request",
        thought="The user wants to know the standard number of continents on Earth. This is a common piece of knowledge.",
        action="Recall or verify the number of continents.",
        confidence=0.95
    )
    ```
    *--(Agent internally recalls the fact)--*
    ```tool_call
    analyze(
        title="Evaluate Fact",
        result="Standard geographical models list 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, South America.",
        analysis="The recalled information directly answers the user's question accurately.",
        next_action="final_answer",
        confidence=1.0
    )
    ```

    *Agent's Final Answer to User:*
    There are 7 continents on Earth: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.

    **Example 2: Multi-Step Information Gathering**

    *User Request:* What is the capital of France and its current population?

    *Agent's Internal Process:*

    ```tool_call
    think(
        title="Plan Information Retrieval",
        thought="The user needs two pieces of information: the capital of France and its current population. I should use external tools (like search) to find the most up-to-date and accurate information.",
        action="First, search for the capital of France.",
        confidence=0.95
    )
    ```

    *Perform multiple external tool calls in parallel*
    *--(Tool call 1: search(query="capital of France"))--*
    *--(Tool call 2: search(query="population of Paris current"))--*
    *--(Tool Result 1: "Paris")--*
    *--(Tool Result 2: "Approximately 2.1 million (city proper, estimate for early 2024)")--*

    ```tool_call
    analyze(
        title="Analyze Capital Search Result",
        result="The search result indicates Paris is the capital of France.",
        analysis="This provides the first piece of requested information. Now I need to find the population of Paris.",
        next_action="continue",
        confidence=1.0
    )
    ```
    ```tool_call
    analyze(
        title="Analyze Population Search Result",
        result="The search provided an estimated population figure for Paris.",
        analysis="I now have both the capital and its estimated population. I can provide the final answer.",
        next_action="final_answer",
        confidence=0.9
    )
    ```

    *Agent's Final Answer to User:*
    The capital of France is Paris. Its estimated population (city proper) is approximately 2.1 million as of early 2024."""
)

NextAction = Literal["continue", "validate", "final_answer"]


def think(title: str, thought: str, action: str | None = None, confidence: float = 0.8) -> str:
    """
    Use this tool as a scratchpad to reason about the question and work through it step-by-step.

    This tool helps break down complex problems into logical steps and track the reasoning process.
    It can be called multiple times as needed. These internal thoughts are never revealed to the user.

    Parameters
    ----------
    title : str
        A concise title for this step.
    thought : str
        Your detailed thought for this step.
    action : str, optional
        What you'll do based on this thought.
    confidence : float, default 0.8
        How confident you are about this thought (0.0 to 1.0).

    Returns
    -------
    str
        A formatted string containing the thought process.
    """
    confidence = min(1.0, max(0.0, confidence))
    thought = f"<thought>\n<title>\n{title}\n</title>\n<thought>\n{thought}\n</thought>\n"
    if action is not None:
        thought += f"<action>\n{action}\n</action>\n"
    thought += f"<confidence>\n{confidence}\n</confidence>\n\n</thought>\n"
    logger.info(thought)
    return thought


def analyze(
    title: str, result: str, analysis: str, next_action: NextAction = "continue", confidence: float = 0.8
) -> str:
    """Use this tool to analyze results from a reasoning step and determine next actions.

    Args:
        title: A concise title for this analysis step
        result: The outcome of the previous action
        analysis: Your analysis of the results
        next_action: What to do next ("continue", "validate", or "final_answer")
        confidence: How confident you are in this analysis (0.0 to 1.0)

    Returns:
        A list of previous thoughts and the new analysis
    """
    confidence = min(1.0, max(0.0, confidence))
    analysis = (
        f"<analysis>\n\n"
        f"<title>\n{title}\n</title>\n"
        f"<result>\n{result}\n</result>\n"
        f"<analysis>\n{analysis}\n</analysis>\n"
        f"<next_action>\n{next_action}\n</next_action>\n"
        f"<confidence>\n{confidence}\n</confidence>\n\n"
        f"</analysis>\n"
    )
    return analysis
