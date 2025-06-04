from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset

from operate_ai.evals import EqEvaluator, Query


class CustomersPerSubscriptionType(BaseModel):
    monthly: int = Field(description="The number of customers with a monthly subscription.")
    annual: int = Field(description="The number of customers with an annual subscription.")


class CustomersPerPlanType(BaseModel):
    basic: int = Field(description="The number of customers with a basic plan.")
    pro: int = Field(description="The number of customers with a pro plan.")
    enterprise: int = Field(description="The number of customers with an enterprise plan.")


class CustomersPerIndustry(BaseModel):
    retail: int = Field(description="The number of customers in the retail industry.")
    tech: int = Field(description="The number of customers in the tech industry.")
    healthcare: int = Field(description="The number of customers in the healthcare industry.")
    education: int = Field(description="The number of customers in the education industry.")
    other: int = Field(description="The number of customers in the other industry.")


class CustomersPerAcquisitionChannel(BaseModel):
    paid_search: int = Field(description="The number of customers acquired through paid search.")
    social_media: int = Field(description="The number of customers acquired through social media.")
    email: int = Field(description="The number of customers acquired through email.")
    affiliate: int = Field(description="The number of customers acquired through affiliate marketing.")
    content: int = Field(description="The number of customers acquired through content marketing.")


type ResultT = (
    int
    | CustomersPerSubscriptionType
    | CustomersPerPlanType
    | CustomersPerIndustry
    | CustomersPerAcquisitionChannel
)

step1_dataset = Dataset[Query[ResultT], ResultT](
    cases=[
        Case(
            name="step1_1",
            inputs=Query(query="How many customers were active in January 2023?", output_type=int),
            expected_output=54,
        ),
        Case(
            name="step1_2",
            inputs=Query(
                query="Show me the breakdown of active Jan 2023 customers by their initial subscription type (Monthly vs Annual)",
                output_type=CustomersPerSubscriptionType,
            ),
            expected_output=CustomersPerSubscriptionType(monthly=39, annual=15),
        ),
        Case(
            name="step1_3",
            inputs=Query(
                query="Break down active Jan 2023 customers by their initial plan type (Basic, Pro, Enterprise)",
                output_type=CustomersPerPlanType,
            ),
            expected_output=CustomersPerPlanType(basic=26, pro=20, enterprise=8),
        ),
        Case(
            name="step1_4",
            inputs=Query(
                query="Show the industry distribution of customers who were active in January 2023",
                output_type=CustomersPerIndustry,
            ),
            expected_output=CustomersPerIndustry(retail=17, tech=16, healthcare=9, education=8, other=4),
        ),
        Case(
            name="step1_5",
            inputs=Query(
                query="What are the acquisition channels for customers active in January 2023?",
                output_type=CustomersPerAcquisitionChannel,
            ),
            expected_output=CustomersPerAcquisitionChannel(
                paid_search=17, social_media=15, email=12, affiliate=9, content=1
            ),
        ),
    ],
    evaluators=[EqEvaluator()],
)
