"""All LLM prompts and their structured output schemas, in one place.

Pydantic field descriptions are part of the prompt: the OpenAI SDK sends the schema,
including descriptions, to the model.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tagging: document metadata from the first pages of a report
# ---------------------------------------------------------------------------

# Same industry list as the organizers' question generator, so results are comparable.
Industry = Literal[
    "Technology", "Financial Services", "Healthcare", "Automotive",
    "Retail", "Energy and Utilities", "Hospitality", "Telecommunications",
    "Media & Entertainment", "Pharmaceuticals", "Aerospace & Defense",
    "Transport & Logistics", "Food & Beverage",
]  # fmt: skip


class EndOfPeriod(BaseModel):
    year: int
    month: int = Field(ge=1, le=12)


class DocumentMetadata(BaseModel):
    company_name: str = Field(description="Full legal name of the reporting company")
    aliases: list[str] = Field(
        description="Other names used for the company: short names, former names, "
        "stock tickers with and without exchange prefix (e.g. 'HCA', 'NYSE: HCA')"
    )
    currency: str = Field(description="ISO 4217 code of the main reporting currency, e.g. USD")
    major_industry: Industry
    end_of_period: EndOfPeriod = Field(description="End of the fiscal period the report covers")


TAGGING_SYSTEM_PROMPT = (
    "You extract metadata from the first pages of a company annual report. "
    "Use only information stated in the text. For the industry, choose the closest option. "
    "For end_of_period, prefer the 'Period statements' section: it lists the most frequent "
    "'year ended <date>' phrases in the whole report."
)
