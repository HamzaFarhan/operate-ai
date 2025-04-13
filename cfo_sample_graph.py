from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai import messages as _messages
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.persistence.file import FileStatePersistence


@dataclass
class CFOGraphState:
    data_ingestion_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    financial_modeler_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    insights_generator_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    validator_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    output_generator_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    planner_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    clarification_message_history: list[_messages.ModelMessage] = field(default_factory=list)


@dataclass
class CFOGraphDeps:
    user_id: str
    user_query: str
    user_persona: str = "startup_founder"  # Can be "startup_founder", "startup_cfo", or "financial_analyst"
    data_sources: list[Path] = field(default_factory=list)
    web_search_enabled: bool = True


class DataIngestionResult(BaseModel):
    processed_data: dict[str, Any]
    data_summary: str
    identified_metrics: list[str]

    def __str__(self) -> str:
        metrics_str = "\n".join([f"- {metric}" for metric in self.identified_metrics])
        return f"<data_summary>\n{self.data_summary}\n</data_summary>\n\n<identified_metrics>\n{metrics_str}\n</identified_metrics>"


class FinancialModelResult(BaseModel):
    model_structure: dict[str, Any]
    formula_explanations: dict[str, str]
    assumptions: dict[str, Any]
    model_summary: str

    def __str__(self) -> str:
        formula_explanations_str = "\n".join(
            [f"{key}: {value}" for key, value in self.formula_explanations.items()]
        )
        assumptions_str = "\n".join([f"{key}: {value}" for key, value in self.assumptions.items()])
        return (
            f"<model_summary>\n{self.model_summary}\n</model_summary>\n\n"
            f"<assumptions>\n{assumptions_str}\n</assumptions>\n\n"
            f"<formula_explanations>\n{formula_explanations_str}\n</formula_explanations>"
        )


class InsightsResult(BaseModel):
    key_insights: list[str]
    visualizations: list[dict[str, Any]]
    recommendations: list[str]

    def __str__(self) -> str:
        insights_str = "\n".join([f"- {insight}" for insight in self.key_insights])
        recommendations_str = "\n".join([f"- {rec}" for rec in self.recommendations])
        return (
            f"<key_insights>\n{insights_str}\n</key_insights>\n\n"
            f"<recommendations>\n{recommendations_str}\n</recommendations>"
        )


class ValidationResult(BaseModel):
    is_valid: bool
    validation_issues: list[str] = []
    corrected_model: dict[str, Any] | None = None
    confidence_score: float

    def __str__(self) -> str:
        issues_str = "\n".join([f"- {issue}" for issue in self.validation_issues])
        return (
            f"<validation_result>\nIs valid: {self.is_valid}\nConfidence score: {self.confidence_score}\n</validation_result>\n\n"
            f"<validation_issues>\n{issues_str}\n</validation_issues>"
        )


class OutputResult(BaseModel):
    excel_model: dict[str, Any]  # This would contain structured data for Excel generation
    report_text: str
    visualizations: list[dict[str, Any]]
    saas_metrics: SaaSMetricsResult | None = None
    sources: list[str]

    def __str__(self) -> str:
        sources_str = "\n".join([f"- {source}" for source in self.sources])
        saas_metrics_str = (
            f"\n\n<saas_metrics>\n{str(self.saas_metrics)}\n</saas_metrics>" if self.saas_metrics else ""
        )
        return f"<report>\n{self.report_text}\n</report>{saas_metrics_str}\n\n<sources>\n{sources_str}\n</sources>"


class SaaSMetricsResult(BaseModel):
    cohort_analysis: dict[str, Any] | None = None
    ltv_analysis: dict[str, Any] | None = None
    cac_analysis: dict[str, Any] | None = None
    payback_analysis: dict[str, Any] | None = None
    retention_analysis: dict[str, Any] | None = None
    marketing_efficiency: dict[str, Any] | None = None
    summary: str

    def __str__(self) -> str:
        metrics: list[str] = []
        if self.cohort_analysis:
            metrics.append("Cohort Analysis")
        if self.ltv_analysis:
            metrics.append("LTV Analysis")
        if self.cac_analysis:
            metrics.append("CAC Analysis")
        if self.payback_analysis:
            metrics.append("Payback Analysis")
        if self.retention_analysis:
            metrics.append("Retention Analysis")
        if self.marketing_efficiency:
            metrics.append("Marketing Efficiency")

        metrics_str = ", ".join(metrics)
        return f"<saas_metrics_summary>\n{self.summary}\n</saas_metrics_summary>\n\n<metrics_included>\n{metrics_str}\n</metrics_included>"


# Initialize the agents
data_ingestion_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You are a data ingestion specialist for a financial analysis system. "
        "You analyze uploaded data files and web data to extract relevant financial information. "
        "Your job is to clean, process, and structure the data for financial modeling."
    ),
    deps_type=CFOGraphDeps,
)

financial_modeler_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You are an expert financial modeler specializing in SaaS metrics and analysis. "
        "Given processed data and a user query, you create sophisticated financial models "
        "with appropriate formulas and assumptions. You are skilled at modeling SaaS-specific "
        "metrics including cohort analysis, LTV, CAC, payback periods, retention/churn analysis, "
        "and marketing efficiency metrics. You select the most appropriate formulas for each scenario "
        "and create dynamic, scalable models with single-cell drivers for key assumptions."
    ),
    deps_type=CFOGraphDeps,
)

insights_generator_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You are an insights generator for SaaS financial data. Given a financial model, "
        "you identify key insights about metrics like LTV, CAC, retention rates, cohort performance, "
        "payback periods, and marketing efficiency. You create appropriate visualizations like cohort tables, "
        "waterfall charts, and segment analyses, and provide actionable recommendations. "
        "You communicate complex SaaS metrics clearly, tailoring your analysis to the user's persona."
    ),
    deps_type=CFOGraphDeps,
)

validator_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You are a financial model validator. You check financial models for errors, "
        "logical inconsistencies, and anomalies. You provide a validation report with a confidence score "
        "and suggest corrections where necessary."
    ),
    deps_type=CFOGraphDeps,
)

output_generator_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You are an output generator for financial analysis. Given a validated financial model and insights, "
        "you create a polished report with clear explanations, visualizations, and Excel model structures. "
        "You tailor your output to the user's persona and ensure all sources are properly cited."
    ),
    deps_type=CFOGraphDeps,
)


@data_ingestion_agent.tool_plain
@financial_modeler_agent.tool_plain
@insights_generator_agent.tool_plain
@validator_agent.tool_plain
@output_generator_agent.tool_plain
def web_search(query: str) -> str:
    """Search the web for financial data and information."""
    # In a real implementation, this would connect to a web search API
    return f"Web search results for: {query}"


@data_ingestion_agent.tool_plain
def parse_csv(file_path: str) -> dict[str, Any]:
    """Parse a CSV file and return structured data."""
    # In a real implementation, this would use pandas to read CSV files
    return {"data": f"Parsed data from {file_path}"}


@data_ingestion_agent.tool_plain
def parse_excel(file_path: str) -> dict[str, Any]:
    """Parse an Excel file and return structured data."""
    # In a real implementation, this would use pandas or openpyxl to read Excel files
    return {"data": f"Parsed data from {file_path}"}


@output_generator_agent.tool_plain
def generate_excel_model(model_structure: dict[str, Any]) -> str:
    """Generate an Excel file from the model structure."""
    # In a real implementation, this would use openpyxl to create an Excel file
    return f"Excel model generated based on structure with {len(model_structure)} sheets"


@output_generator_agent.tool_plain
def generate_visualization(data: dict[str, Any], chart_type: str) -> str:
    """Generate a visualization from data."""
    # In a real implementation, this would use matplotlib or another visualization library
    return f"{chart_type} visualization generated"


@output_generator_agent.tool_plain
def generate_cohort_analysis(cohort_data: dict[str, Any], metric: str) -> str:
    """Generate a cohort analysis table and visualization for retention or other metrics"""
    # In a real implementation, this would use pandas and matplotlib/seaborn
    return f"Cohort analysis for {metric} generated"


@output_generator_agent.tool_plain
def generate_waterfall_chart(data: dict[str, Any], title: str) -> str:
    """Generate a waterfall chart for metrics like revenue, customer growth, etc."""
    # In a real implementation, this would use matplotlib or another visualization library
    return f"Waterfall chart for {title} generated"


@output_generator_agent.tool_plain
def generate_customer_segment_analysis(segment_data: dict[str, Any], metrics: list[str]) -> str:
    """Generate analysis of metrics across different customer segments"""
    # In a real implementation, this would use pandas and visualization libraries
    return f"Customer segment analysis for {', '.join(metrics)} generated"


@data_ingestion_agent.system_prompt
@financial_modeler_agent.system_prompt
@insights_generator_agent.system_prompt
@validator_agent.system_prompt
@output_generator_agent.system_prompt
def get_system_prompt(ctx: RunContext[CFOGraphDeps]) -> str:
    persona_descriptions = {
        "startup_founder": "The user is a startup founder looking for quick financial analysis for investor meetings. They have limited time and may lack deep excel knowledge.",
        "startup_cfo": "The user is a startup CFO who needs to answer questions from the CEO and create analysis and charts for board decks. They have limited time and resources.",
        "financial_analyst": "The user is a financial analyst who needs to build accurate, scalable financial models. They need help with error-prone formulas and manual data entry.",
    }

    persona_description = persona_descriptions.get(ctx.deps.user_persona, persona_descriptions["startup_founder"])

    return f"<user_id>\n{ctx.deps.user_id}\n</user_id>\n\n<user_persona>\n{persona_description}\n</user_persona>\n\n<web_search_enabled>\n{ctx.deps.web_search_enabled}\n</web_search_enabled>"


@dataclass
class WebSearchNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """If needed, perform web search after user approval."""

    docstring_notes = True

    user_query: str
    data_ingestion_result: DataIngestionResult
    search_query: str
    requires_approval: bool = True

    async def run(self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]) -> FinancialModelingNode:
        if self.requires_approval:
            # In a real implementation, this would wait for user approval
            logger.info(f"Waiting for user approval to search: {self.search_query}")
            # Simulate waiting for approval
            await asyncio.sleep(1)

        search_results = await web_search(self.search_query)

        # Update the data ingestion result with web search data
        updated_data = self.data_ingestion_result.processed_data.copy()
        updated_data["web_search_results"] = search_results

        updated_result = DataIngestionResult(
            processed_data=updated_data,
            data_summary=f"{self.data_ingestion_result.data_summary}\nWeb search results incorporated.",
            identified_metrics=self.data_ingestion_result.identified_metrics,
        )

        return FinancialModelingNode(user_query=self.user_query, data_ingestion_result=updated_result)


@dataclass
class DataIngestionNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """Process and structure data for financial modeling."""

    docstring_notes = True

    user_query: str

    async def run(
        self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]
    ) -> WebSearchNode | FinancialModelingNode:
        response = await data_ingestion_agent.run(
            user_prompt=self.user_query,
            message_history=ctx.state.data_ingestion_message_history,
            deps=ctx.deps,
            result_type=DataIngestionResult,
        )
        ctx.state.data_ingestion_message_history = response.all_messages()

        # Check if web search is needed and enabled
        if ctx.deps.web_search_enabled and any(
            keyword in self.user_query.lower()
            for keyword in ["market", "industry", "competitor", "trend", "forecast", "benchmark"]
        ):
            search_query = f"financial data {self.user_query}"
            return WebSearchNode(
                user_query=self.user_query, data_ingestion_result=response.data, search_query=search_query
            )

        return FinancialModelingNode(user_query=self.user_query, data_ingestion_result=response.data)


@dataclass
class FinancialModelingNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """Generate a financial model based on processed data and user query."""

    docstring_notes = True

    user_query: str
    data_ingestion_result: DataIngestionResult

    async def run(self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]) -> InsightsGenerationNode:
        user_prompt = f"<user_query>\n{self.user_query}\n</user_query>\n\n<processed_data>\n{str(self.data_ingestion_result)}\n</processed_data>"
        response = await financial_modeler_agent.run(
            user_prompt=user_prompt,
            message_history=ctx.state.financial_modeler_message_history,
            deps=ctx.deps,
            result_type=FinancialModelResult,
        )
        ctx.state.financial_modeler_message_history = response.all_messages()
        return InsightsGenerationNode(
            user_query=self.user_query,
            data_ingestion_result=self.data_ingestion_result,
            financial_model_result=response.data,
        )


@dataclass
class InsightsGenerationNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """Generate insights, visualizations, and recommendations from the financial model."""

    docstring_notes = True

    user_query: str
    data_ingestion_result: DataIngestionResult
    financial_model_result: FinancialModelResult

    async def run(self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]) -> ValidationNode:
        user_prompt = (
            f"<user_query>\n{self.user_query}\n</user_query>\n\n"
            f"<processed_data>\n{str(self.data_ingestion_result)}\n</processed_data>\n\n"
            f"<financial_model>\n{str(self.financial_model_result)}\n</financial_model>"
        )
        response = await insights_generator_agent.run(
            user_prompt=user_prompt,
            message_history=ctx.state.insights_generator_message_history,
            deps=ctx.deps,
            result_type=InsightsResult,
        )
        ctx.state.insights_generator_message_history = response.all_messages()
        return ValidationNode(
            user_query=self.user_query,
            data_ingestion_result=self.data_ingestion_result,
            financial_model_result=self.financial_model_result,
            insights_result=response.data,
        )


@dataclass
class ValidationNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """Validate the financial model and insights for accuracy and consistency."""

    docstring_notes = True

    user_query: str
    data_ingestion_result: DataIngestionResult
    financial_model_result: FinancialModelResult
    insights_result: InsightsResult

    async def run(
        self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]
    ) -> OutputGenerationNode | FinancialModelingRevisionNode:
        user_prompt = (
            f"<user_query>\n{self.user_query}\n</user_query>\n\n"
            f"<financial_model>\n{str(self.financial_model_result)}\n</financial_model>\n\n"
            f"<insights>\n{str(self.insights_result)}\n</insights>"
        )
        response = await validator_agent.run(
            user_prompt=user_prompt,
            message_history=ctx.state.validator_message_history,
            deps=ctx.deps,
            result_type=ValidationResult,
        )
        ctx.state.validator_message_history = response.all_messages()

        if response.data.is_valid:
            return OutputGenerationNode(
                user_query=self.user_query,
                data_ingestion_result=self.data_ingestion_result,
                financial_model_result=self.financial_model_result,
                insights_result=self.insights_result,
                validation_result=response.data,
            )
        else:
            return FinancialModelingRevisionNode(
                user_query=self.user_query,
                data_ingestion_result=self.data_ingestion_result,
                financial_model_result=self.financial_model_result,
                insights_result=self.insights_result,
                validation_result=response.data,
            )


@dataclass
class FinancialModelingRevisionNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """Revise the financial model based on validation feedback."""

    docstring_notes = True

    user_query: str
    data_ingestion_result: DataIngestionResult
    financial_model_result: FinancialModelResult
    insights_result: InsightsResult
    validation_result: ValidationResult

    async def run(self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]) -> ValidationNode:
        user_prompt = (
            f"<user_query>\n{self.user_query}\n</user_query>\n\n"
            f"<processed_data>\n{str(self.data_ingestion_result)}\n</processed_data>\n\n"
            f"<financial_model>\n{str(self.financial_model_result)}\n</financial_model>\n\n"
            f"<validation_feedback>\n{str(self.validation_result)}\n</validation_feedback>"
        )
        response = await financial_modeler_agent.run(
            user_prompt=user_prompt,
            message_history=ctx.state.financial_modeler_message_history,
            deps=ctx.deps,
            result_type=FinancialModelResult,
        )
        ctx.state.financial_modeler_message_history = response.all_messages()

        # After model revision, regenerate insights
        insights_prompt = (
            f"<user_query>\n{self.user_query}\n</user_query>\n\n"
            f"<processed_data>\n{str(self.data_ingestion_result)}\n</processed_data>\n\n"
            f"<financial_model>\n{str(response.data)}\n</financial_model>"
        )
        insights_response = await insights_generator_agent.run(
            user_prompt=insights_prompt,
            message_history=ctx.state.insights_generator_message_history,
            deps=ctx.deps,
            result_type=InsightsResult,
        )
        ctx.state.insights_generator_message_history = insights_response.all_messages()

        return ValidationNode(
            user_query=self.user_query,
            data_ingestion_result=self.data_ingestion_result,
            financial_model_result=response.data,
            insights_result=insights_response.data,
        )


@dataclass
class OutputGenerationNode(BaseNode[CFOGraphState, CFOGraphDeps, str]):
    """Generate the final output including report, Excel model, and visualizations."""

    docstring_notes = True

    user_query: str
    data_ingestion_result: DataIngestionResult
    financial_model_result: FinancialModelResult
    insights_result: InsightsResult
    validation_result: ValidationResult

    async def run(self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]) -> End[str]:
        user_prompt = (
            f"<user_query>\n{self.user_query}\n</user_query>\n\n"
            f"<financial_model>\n{str(self.financial_model_result)}\n</financial_model>\n\n"
            f"<insights>\n{str(self.insights_result)}\n</insights>\n\n"
            f"<validation>\n{str(self.validation_result)}\n</validation>"
        )
        response = await output_generator_agent.run(
            user_prompt=user_prompt,
            message_history=ctx.state.output_generator_message_history,
            deps=ctx.deps,
            result_type=OutputResult,
        )
        ctx.state.output_generator_message_history = response.all_messages()

        # Format the final response for the user
        final_response = (
            f"# Financial Analysis Report\n\n"
            f"{response.data.report_text}\n\n"
            f"## Key Insights\n\n"
            f"{chr(10).join(['- ' + insight for insight in self.insights_result.key_insights])}\n\n"
            f"## Recommendations\n\n"
            f"{chr(10).join(['- ' + rec for rec in self.insights_result.recommendations])}\n\n"
            f"## Sources\n\n"
            f"{chr(10).join(['- ' + source for source in response.data.sources])}"
        )

        return End(final_response)


@dataclass
class PlanningNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """Analyze query, data, and determine required sub-tasks."""

    docstring_notes = True

    async def run(
        self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]
    ) -> InteractiveClarificationNode | DataIngestionNode:
        assessment_prompt = (
            f"<user_query>\n{ctx.deps.user_query}\n</user_query>\n"
            f"<data_sources>\n{len(ctx.deps.data_sources)} files provided\n</data_sources>"
        )

        response = await data_ingestion_agent.run(
            user_prompt=assessment_prompt,
            message_history=ctx.state.planner_message_history,
            deps=ctx.deps,
            result_type=DataIngestionResult,
        )
        ctx.state.planner_message_history = response.all_messages()

        # Determine if we need clarifications
        if not ctx.deps.data_sources and "forecast" in ctx.deps.user_query.lower():
            return InteractiveClarificationNode()
        return DataIngestionNode(user_query=ctx.deps.user_query)


@dataclass
class InteractiveClarificationNode(BaseNode[CFOGraphState, CFOGraphDeps]):
    """Collect additional information through interactive questions."""

    docstring_notes = True
    required_clarifications: list[str] = field(
        default_factory=lambda: [
            "What's the time horizon for this analysis?",
            "Which key metrics are most important to track?",
            "Are there any known assumptions or constraints?",
        ]
    )

    async def run(self, ctx: GraphRunContext[CFOGraphState, CFOGraphDeps]) -> DataIngestionNode:
        # In real implementation, collect user responses
        # Simulating responses for now
        clarifications = {
            "time_horizon": "3 years",
            "key_metrics": ["ARR", "CAC", "Gross Margin"],
            "assumptions": "15% monthly growth rate",
        }

        updated_query = (
            f"{ctx.deps.user_query}\n\n"
            f"Clarifications:\n"
            f"- Time Horizon: {clarifications['time_horizon']}\n"
            f"- Key Metrics: {', '.join(clarifications['key_metrics'])}\n"
            f"- Assumptions: {clarifications['assumptions']}"
        )

        return DataIngestionNode(user_query=updated_query)


@financial_modeler_agent.tool_plain
def calculate_ltv(arpu: float, churn_rate: float, profit_margin: float) -> float:
    """Calculate Customer Lifetime Value using the formula LTV = (ARPU / Churn Rate) × Profit Margin"""
    return (arpu / churn_rate) * profit_margin


@financial_modeler_agent.tool_plain
def calculate_cac(marketing_cost: float, new_customers: int) -> float:
    """Calculate Customer Acquisition Cost using the formula CAC = Marketing Cost / New Customers"""
    return marketing_cost / new_customers


@financial_modeler_agent.tool_plain
def calculate_payback_period(cac: float, mrr: float, gross_margin: float) -> float:
    """Calculate CAC Payback Period using the formula Payback Period = CAC / (MRR × Gross Margin)"""
    return cac / (mrr * gross_margin)


@financial_modeler_agent.tool_plain
def calculate_retention_rate(customers_start: int, customers_end: int, new_customers: int) -> float:
    """Calculate Retention Rate using the formula Retention = (Customers End - New Customers) / Customers Start"""
    return (customers_end - new_customers) / customers_start


@financial_modeler_agent.tool_plain
def calculate_churn_rate(customers_start: int, customers_churned: int) -> float:
    """Calculate Churn Rate using the formula Churn Rate = Customers Churned / Customers Start"""
    return customers_churned / customers_start


# Create the CFO graph
cfo_graph = Graph(
    nodes=[
        PlanningNode,
        InteractiveClarificationNode,
        DataIngestionNode,
        WebSearchNode,
        FinancialModelingNode,
        InsightsGenerationNode,
        ValidationNode,
        FinancialModelingRevisionNode,
        OutputGenerationNode,
    ],
    auto_instrument=False,
)

# Save graph visualization
cfo_graph.mermaid_save("cfo_graph.jpg", direction="TB")


async def run_cfo_graph(
    user_query: str,
    user_id: str,
    user_persona: str = "startup_founder",
    data_sources: list[Path] | None = None,
    web_search_enabled: bool = True,
    state_path: Path | None = None,
) -> str:
    """Run the CFO graph with the given input parameters."""
    state_path = state_path or Path(f"cfo_agent_{str(uuid4())}.json")
    persistence = FileStatePersistence(state_path)

    deps = CFOGraphDeps(
        user_id=user_id,
        user_query=user_query,
        user_persona=user_persona,
        data_sources=data_sources or [],
        web_search_enabled=web_search_enabled,
    )

    persistence.set_graph_types(cfo_graph)

    if state_path.exists():
        try:
            print((await persistence.load_all())[-1])
            async with cfo_graph.iter_from_persistence(persistence=persistence, deps=deps) as run:
                async for node in run:
                    logger.info(node)
            if run.result:
                return run.result.output
            else:
                raise ValueError("No result from graph")
        except Exception as e:
            logger.error(f"Error loading from persistence: {e}")
            # Fall back to starting a new graph

    async with cfo_graph.iter(
        start_node=PlanningNode(),
        state=CFOGraphState(),
        deps=deps,
        persistence=persistence,
    ) as run:
        async for node in run:
            logger.info(node)

    if run.result:
        return run.result.output
    else:
        raise ValueError("No result from graph")


# if __name__ == "__main__":
#     import asyncio

#     sample_query = "Create a 3-year financial projection for my SaaS startup with $50k monthly recurring revenue growing at 15% annually. Include costs, cash flow, and key metrics."

#     res = asyncio.run(run_cfo_graph(user_query=sample_query, user_id="user123", user_persona="startup_founder"))

#     logger.success(res)

#     res = asyncio.run(run_cfo_graph(user_query=sample_query, user_id="user123", user_persona="startup_founder"))

#     logger.success(res)
