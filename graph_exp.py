from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

load_dotenv()

# Rich console setup for dark theme
custom_theme = Theme(
    {
        "info": "cyan",
        "user": "green",
        "ai": "magenta",
        "error": "bold red",
        "success": "bold green",
    }
)
console = Console(theme=custom_theme)

MODEL = "google-gla:gemini-2.0-flash"

SuccessT = TypeVar("SuccessT")
UntilSuccessT = TypeVar("UntilSuccessT")


class Success(BaseModel):
    message: str


AgentOutputT = UntilSuccessT | SuccessT


class AgentPrompt(BaseModel):
    prompt: str
    tool: str


@dataclass
class GraphDeps:
    available_tools: dict[str, Tool[GraphDeps]]


AnalysisType = Literal[
    "Cohort Analysis",
    "Customer LTV",
    "CAC Analysis",
    "Payback Model",
    "Retention & Churn Analysis",
    "Marketing Efficiency",
    "Other",
]

SegmentationDimension = Literal[
    "Acquisition Month",
    "Geography",
    "Industry Segment",
    "Customer Type",
    "Acquisition Channel",
    "Product Tier",
    "Subscription Type",
    "Lead Source",
    "Time Period",
]


class TimePeriod(BaseModel):
    """Defines the time period for the analysis."""

    start_date: date | None = Field(None, description="Start date for the analysis period.")
    end_date: date | None = Field(None, description="End date for the analysis period.")
    relative_period: str | None = Field(None, description="Relative time period (e.g., 'Last Quarter', 'YTD').")


class AnalyticalTask(BaseModel):
    """
    Represents the user's goal for a financial or data analysis task,
    gathered through interaction before planning and execution.

    This model structure helps ensure all necessary components for
    analysis are captured systematically.
    """

    goal_description: str = Field(
        ...,
        description="High-level description of what the user wants to achieve, captured from initial interaction.",
    )
    analysis_type: AnalysisType = Field(
        ..., description="The primary type of analysis requested (e.g., LTV, Cohort)."
    )
    specific_metrics: list[str] = Field(
        default_factory=list,
        description="Specific metrics the user is interested in (e.g., 'Monthly Recurring Revenue', 'Churn Rate'). Allows for precise targeting.",
    )
    segmentation_dimensions: list[SegmentationDimension] = Field(
        default_factory=list,
        description="How the data should be segmented or grouped (e.g., 'Geography', 'Product Tier'). Enables multi-dimensional analysis.",
    )
    time_period: TimePeriod | None = Field(
        None, description="The time frame relevant for the analysis (e.g., specific dates, relative periods)."
    )
    additional_context: list[str] = Field(
        default_factory=list,
        description="Any other relevant context or specific instructions from the user (e.g., specific assumptions, formatting requests).",
    )


financial_data_agent = Agent(
    name="financial_data_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions="Use the tools at your disposal to fetch specific financial data points based on the prompt.",
)


@financial_data_agent.tool_plain
def get_company_revenue(company_name: str) -> float:
    """
    Returns the annual revenue for the specified company in USD millions.
    """
    console.print(
        Panel(
            f"[info]Getting revenue for: [bold]{company_name}[/bold][/info]",
            title="Internal Log",
            border_style="dim blue",
        )
    )
    revenue_dict = {"Cloud Pro": 150.5, "Tech Solutions": 75.2, "DefaultCorp": 10.0}
    return revenue_dict.get(company_name, 5.0)


@financial_data_agent.tool_plain
def get_stock_price(ticker: str) -> float:
    """
    Returns the current stock price for the given ticker symbol.
    """
    console.print(
        Panel(
            f"[info]Getting stock price for: [bold]{ticker}[/bold][/info]",
            title="Internal Log",
            border_style="dim blue",
        )
    )
    stock_dict = {"CLDP": 125.50, "TSOL": 88.75, "DFLT": 5.20}
    return stock_dict.get(ticker.upper(), 1.00)


async def get_financial_data(ctx: RunContext[GraphDeps], prompt: str) -> str:
    """
    Processes the given prompt to retrieve specific financial data (e.g., revenue, stock price).
    Use this tool if the required agent prompt targets specific financial data points.

    Args:
        ctx: The graph context.
        prompt: The prompt describing what financial data is needed.

    Returns:
        str: The requested financial data or an error message.
    """
    console.print(
        Panel(
            f"[info]Financial Data Agent received prompt: [bold]{prompt}[/bold][/info]",
            title="Internal Log",
            border_style="dim blue",
        )
    )
    res = await financial_data_agent.run(user_prompt=prompt, deps=ctx.deps)
    return res.output


market_analysis_agent = Agent(
    name="market_analysis_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions="Use the tools at your disposal to perform market analysis based on the prompt.",
)


@market_analysis_agent.tool_plain
def get_market_trend(sector: str) -> str:
    """
    Returns a market trend analysis for the given sector.
    """
    console.print(
        Panel(
            f"[info]Getting market trend for: [bold]{sector}[/bold][/info]",
            title="Internal Log",
            border_style="dim blue",
        )
    )
    trends = {
        "Tech": "Positive growth driven by AI adoption.",
        "Healthcare": "Stable growth with focus on telemedicine.",
        "Retail": "Mixed outlook, e-commerce focus essential.",
        "SaaS": "Strong growth, increasing competition.",
    }
    return trends.get(sector, "General market conditions are stable.")


@market_analysis_agent.tool_plain
def get_competitor_analysis(company_name: str) -> str:
    """
    Returns a competitor analysis for the specified company.
    """
    console.print(
        Panel(
            f"[info]Getting competitor analysis for: [bold]{company_name}[/bold][/info]",
            title="Internal Log",
            border_style="dim blue",
        )
    )
    analysis = {
        "Cloud Pro": "Competes with major cloud providers and niche SaaS tools. Key differentiator is ease of use.",
        "Tech Solutions": "Faces competition from established IT consultancies. Focuses on mid-market segment.",
    }
    return analysis.get(company_name, "Primary competitors are established players in the relevant sector.")


async def get_market_analysis(ctx: RunContext[GraphDeps], prompt: str) -> str:
    """
    Processes the given prompt to retrieve market analysis (e.g., trends, competitor info).
    Use this tool if the required agent prompt targets market analysis.

    Args:
        ctx: The graph context.
        prompt: The prompt describing what market analysis is needed.

    Returns:
        str: The requested market analysis or an error message.
    """
    console.print(
        Panel(
            f"[info]Market Analysis Agent received prompt: [bold]{prompt}[/bold][/info]",
            title="Internal Log",
            border_style="dim blue",
        )
    )
    res = await market_analysis_agent.run(user_prompt=prompt, deps=ctx.deps)
    return res.output


task_info_agent = Agent(
    name="task_info_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=(
        "Engage the user in a natural, friendly conversation to understand their analytical request.\n"
        "The user might be non-technical (e.g., a CEO or manager), so use clear, simple language and avoid jargon.\n"
        "Ask clarifying questions as needed. Your internal goal is to gather all necessary details to construct a complete `AnalyticalTask` object.\n"
        "**Crucially, do not reveal this technical goal or mention 'AnalyticalTask' or data structures to the user.**\n"
        "If the user isn't interested in clarifying further, you can make some assumptions and proceed.\n"
        "Focus on making the interaction feel like talking to a helpful human assistant who wants to understand their needs thoroughly.\n"
        "Return the AnalyticalTask object when you're ready.\n"
    ),
    output_type=AgentOutputT[str, AnalyticalTask],
)


planner_agent = Agent(
    name="planner_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=(
        "Based on the user's analytical task, create a list of prompts for specialized agents.\n"
        "Each item in the list should be an 'AgentPrompt' containing:\n"
        "- prompt: A specific instruction or question for a specialized agent.\n"
        "- tool: The name of the single tool (which represents an agent) that can best handle that prompt.\n"
        "Your plan will be given to an executor agent that routes the prompts to the correct tool/agent.\n"
        "Refer to the <available_tools> list to know which tools/agents the executor has access to.\n"
        "Do not try to execute the prompts yourself. Just create the list of AgentPrompt objects."
    ),
    output_type=list[AgentPrompt],
)


executor_agent = Agent(
    name="executor_agent",
    model=MODEL,
    deps_type=GraphDeps,
    instructions=(
        "You will receive an analytical task and a list of specific prompts targeted at different tools/agents.\n"
        "Execute the prompts sequentially using the specified tool for each.\n"
        "You may add relevant data/context from completed tasks to subsequent prompts as needed.\n"
        "Pass the 'prompt' from the AgentPrompt object to the corresponding tool, "
        "updating it with new context when appropriate.\n"
        "Gather the results from each tool execution.\n"
        "Once all prompts are executed, synthesize the results and return a final answer in a `Success` object.\n"
        "Use the results from previous steps if needed for subsequent prompts."
    ),
    output_type=AgentOutputT[str, Success],
    tools=[get_financial_data, get_market_analysis],
)


@financial_data_agent.instructions
@market_analysis_agent.instructions
@executor_agent.instructions
@planner_agent.instructions
@task_info_agent.instructions
def user_data_context(ctx: RunContext[GraphDeps]) -> str:
    return f"\n\n{ctx.deps}\n\n<current_time>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</current_time>\n\n"


@dataclass
class TaskInfoNode(BaseNode[None, GraphDeps]):
    user_prompt: str

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Planner:
        console.print(
            Panel(
                f"[info]TaskInfoNode received prompt: [bold]{self.user_prompt}[/bold][/info]",
                title="Internal Log",
                border_style="dim blue",
            )
        )
        message_history = None
        while True:
            run = await task_info_agent.run(
                user_prompt=self.user_prompt, deps=ctx.deps, message_history=message_history
            )
            if isinstance(run.output, AnalyticalTask):
                return Planner(analytical_task=run.output)
            else:
                console.print(Panel(f"[ai]{run.output}[/ai]", title="AI Message", border_style="magenta"))
                self.user_prompt = Prompt.ask("[user]Your response[/user]", console=console)
                message_history = run.all_messages()


@dataclass
class Planner(BaseNode[None, GraphDeps]):
    analytical_task: AnalyticalTask

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> Executor:
        tools_str = "<available_tools>\n"
        for tool in ctx.deps.available_tools.values():
            tools_str += f"- {tool.name}: {tool.description}\n\n"
        tools_str += "</available_tools>"
        user_prompt = (
            f"<analytical_task>\n{self.analytical_task.model_dump_json()}\n</analytical_task>\n\n{tools_str}\n\n"
        )
        console.print(
            Panel(
                f"[info]Planner received prompt:\n[bold]{user_prompt}[/bold][/info]",
                title="Internal Log",
                border_style="dim blue",
            )
        )
        plan_res = await planner_agent.run(user_prompt=user_prompt, deps=ctx.deps)
        return Executor(analytical_task=self.analytical_task, prompts=plan_res.output)


@dataclass
class Executor(BaseNode[None, GraphDeps, str]):
    analytical_task: AnalyticalTask
    prompts: list[AgentPrompt]

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> End[str]:
        prompts_str = "<prompts>\n"
        for i, agent_prompt in enumerate(self.prompts):
            prompts_str += f"{i + 1}. Tool: {agent_prompt.tool}, Prompt: {agent_prompt.prompt}\n"
        prompts_str += "</prompts>"
        user_prompt = (
            f"<analytical_task>\n{self.analytical_task.model_dump_json()}\n</analytical_task>\n\n{prompts_str}\n\n"
        )
        console.print(
            Panel(
                f"[info]Executor received prompt:\n[bold]{user_prompt}[/bold][/info]",
                title="Internal Log",
                border_style="dim blue",
            )
        )
        message_history = None
        while True:
            run = await executor_agent.run(user_prompt=user_prompt, deps=ctx.deps, message_history=message_history)
            if isinstance(run.output, Success):
                return End(run.output.message)
            message_history = run.all_messages()
            console.print(Panel(f"[ai]{run.output}[/ai]", title="AI Message", border_style="magenta"))
            user_goal = Prompt.ask("[user]Your response[/user]", console=console)
            if user_goal == "q":
                return End(run.output)


async def main():
    graph = Graph(nodes=[TaskInfoNode, Planner, Executor])
    graph_deps = GraphDeps(available_tools=executor_agent._function_tools)
    # first_prompt = "Get the company revenue and competitor analysis of 'Tech Solutions'"

    # Display welcome message
    console.print(
        Panel(
            "[ai]Welcome to OperateAI![/ai]",
            title="AI Assistant",
            border_style="magenta",
        )
    )

    # Get user input with rich prompt
    user_prompt = Prompt.ask("[user]What can I help you with?[/user]", console=console)

    # Log user input with rich panel
    console.print(Panel(f"[user]{user_prompt}[/user]", title="User Request", border_style="green"))

    res = await graph.run(start_node=TaskInfoNode(user_prompt=user_prompt), deps=graph_deps)
    return res.output


if __name__ == "__main__":
    res = asyncio.run(main())
    console.print(Panel(f"[ai]{res}[/ai]", title="Final Result", border_style="bold magenta", padding=(1, 2)))
