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


# ---------------------------------------------------------------------------
# Reranking: score retrieved pages for relevance to the question
# Prompt and schema from IlyaRice/RAG-Challenge-2 (MIT), src/prompts.py
# ---------------------------------------------------------------------------

RERANK_SYSTEM_PROMPT = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""


class BlockRanking(BaseModel):
    """Rank retrieved text block relevance to a query."""

    # Added to the original schema: without it, scores are matched to blocks by list
    # position, and the model sometimes returns them in a different order.
    block_number: int = Field(description="Number of the ranked block, N in 'Block N'")
    reasoning: str = Field(
        description="Analysis of the block, identifying key information and how it relates to the query"
    )
    relevance_score: float = Field(
        description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant"
    )


class BlockRankings(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""

    block_rankings: list[BlockRanking] = Field(
        description="A list of text blocks and their associated relevance scores."
    )


def rerank_user_prompt(query: str, blocks: list[str]) -> str:
    formatted = "\n\n---\n\n".join(
        f'Block {i + 1}:\n\n"""\n{text}\n"""' for i, text in enumerate(blocks)
    )
    return (
        f'Here is the query: "{query}"\n\n'
        "Here are the retrieved text blocks:\n"
        f"{formatted}\n\n"
        f"You should provide exactly {len(blocks)} rankings, in order."
    )
