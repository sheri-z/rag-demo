"""All LLM prompts and their structured output schemas, in one place.

Pydantic field descriptions are part of the prompt: the OpenAI SDK sends the schema,
including descriptions, to the model.
"""

from typing import Literal

from pydantic import BaseModel, Field, create_model

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


# ---------------------------------------------------------------------------
# Answering: one prompt and schema per answer type, structured output with CoT
# Prompts, examples and schemas from IlyaRice/RAG-Challenge-2 (MIT), src/prompts.py
# ---------------------------------------------------------------------------

ANSWER_INSTRUCTION = """
You are a RAG (Retrieval-Augmented Generation) answering system.
Your task is to answer the given question based only on information from the company's annual report, which is uploaded in the format of relevant pages extracted using RAG.

Before giving a final answer, carefully think out loud and step by step. Pay special attention to the wording of the question.
- Keep in mind that the content containing the answer may be worded differently than the question.
- The question was autogenerated from a template, so it may be meaningless or not applicable to the given company.
"""

ANSWER_USER_PROMPT = """
Here is the context:
\"\"\"
{context}
\"\"\"

---

Here is the question:
"{question}"
"""


class NameAnswer(BaseModel):
    step_by_step_analysis: str = Field(
        description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested value, but only a similar one."
    )
    reasoning_summary: str = Field(
        description="Concise summary of the step-by-step reasoning process. Around 50 words."
    )
    relevant_pages: list[int] = Field(
        description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
"""
    )
    final_answer: str | Literal["N/A"] = Field(
        description="""
If it is a company name, should be extracted exactly as it appears in question.
If it is a person name, it should be their full name.
If it is a product name, it should be extracted exactly as it appears in the context.
Without any extra information, words or comments.
- Return 'N/A' if information is not available in the context
"""
    )


class NumberAnswer(BaseModel):
    step_by_step_analysis: str = Field(
        description="""
Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.
**Strict Metric Matching Required:**    

1. Determine the precise concept the question's metric represents. What is it actually measuring?
2. Examine potential metrics in the context. Don't just compare names; consider what the context metric measures.
3. Accept ONLY if: The context metric's meaning *exactly* matches the target metric. Synonyms are acceptable; conceptual differences are NOT.
4. Reject (and use 'N/A') if:
    - The context metric covers more or less than the question's metric.
    - The context metric is a related concept but not the *exact* equivalent (e.g., a proxy or a broader category).
    - Answering requires calculation, derivation, or inference.
    - Aggregation Mismatch: The question needs a single value but the context offers only an aggregated total
5. No Guesswork: If any doubt exists about the metric's equivalence, default to `N/A`."
"""
    )
    reasoning_summary: str = Field(
        description="Concise summary of the step-by-step reasoning process. Around 50 words."
    )
    relevant_pages: list[int] = Field(
        description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
"""
    )
    final_answer: float | int | Literal["N/A"] = Field(
        description="""
An exact metric number is expected as the answer.
- Example for percentages:
    Value from context: 58,3%
    Final answer: 58.3

Pay special attention to any mentions in the context about whether metrics are reported in units, thousands, or millions to adjust number in final answer with no changes, three zeroes or six zeroes accordingly.
Pay attention if value wrapped in parentheses, it means that value is negative.

- Example for negative values:
    Value from context: (2,124,837) CHF
    Final answer: -2124837

- Example for numbers in thousands:
    Value from context: 4970,5 (in thousands $)
    Final answer: 4970500

- Return 'N/A' if metric provided is in a different currency than mentioned in the question
    Example of value from context: 780000 USD, but question mentions EUR
    Final answer: 'N/A'

- Return 'N/A' if metric is not directly stated in context EVEN IF it could be calculated from other metrics in the context
    Example: Requested metric: Dividend per Share; Only available metrics from context: Total Dividends Paid ($5,000,000), and Number of Outstanding Shares (1,000,000); Calculated DPS = Total Dividends / Outstanding Shares.
    Final answer: 'N/A'

- Return 'N/A' if information is not available in the context
"""
    )


class BooleanAnswer(BaseModel):
    step_by_step_analysis: str = Field(
        description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested value, but only a similar one."
    )
    reasoning_summary: str = Field(
        description="Concise summary of the step-by-step reasoning process. Around 50 words."
    )
    relevant_pages: list[int] = Field(
        description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
"""
    )
    # Fixed a slip in the original description, which said to return False when the
    # information IS in the context.
    final_answer: bool = Field(
        description="""
A boolean value (True or False) extracted from the context that precisely answers the question.
If the question asks whether something happened and the context has no information about it, return False.
"""
    )


class NamesAnswer(BaseModel):
    step_by_step_analysis: str = Field(
        description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested entity, but only a similar one."
    )
    reasoning_summary: str = Field(
        description="Concise summary of the step-by-step reasoning process. Around 50 words."
    )
    relevant_pages: list[int] = Field(
        description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
"""
    )
    final_answer: list[str] | Literal["N/A"] = Field(
        description="""
Each entry should be extracted exactly as it appears in the context.

If the question asks about positions (e.g., changes in positions), return ONLY position titles, WITHOUT names or any additional information. Appointments on new leadership positions also should be counted as changes in positions. If several changes related to position with same title are mentioned, return title of such position only once. Position title always should be in singular form.
Example of answer ['Chief Technology Officer', 'Board Member', 'Chief Executive Officer']

If the question asks about names, return ONLY the full names exactly as they are in the context.
Example of answer ['Carly Kennedy', 'Brian Appelgate Jr.']

If the question asks about new launched products, return ONLY the product names exactly as they are in the context. Candidates for new products or products in testing phase not counted as new launched products.
Example of answer ['EcoSmart 2000', 'GreenTech Pro']

- Return 'N/A' if information is not available in the context
"""
    )


NAME_EXAMPLE = r"""
Example:
Question: 
"Who was the CEO of 'Southwest Airlines Co.'?" 

Answer: 
```
{
  "step_by_step_analysis": "1. The question asks for the CEO of 'Southwest Airlines Co.'. The CEO is typically the highest-ranking executive responsible for the overall management of the company, sometimes referred to as the President or Managing Director.\n2. My source of information is a document that appears to be 'Southwest Airlines Co.''s annual report. This document will be used to identify the individual holding the CEO position.\n3. Within the provided document, there is a section that identifies Robert E. Jordan as the President & Chief Executive Officer of 'Southwest Airlines Co.'. The document confirms his role since February 2022.\n4. Therefore, based on the information found in the document, the CEO of 'Southwest Airlines Co.' is Robert E. Jordan.",
  "reasoning_summary": "'Southwest Airlines Co.''s annual report explicitly names Robert E. Jordan as President & Chief Executive Officer since February 2021. This directly answers the question.",
  "relevant_pages": [58],
  "final_answer": "Robert E. Jordan"
}
```
"""


NUMBER_EXAMPLE = r"""
Example 1:
Question:
"What was the total assets of 'Waste Connections Inc.' in the fiscal year 2022?"

Answer:
```
{
  "step_by_step_analysis": "1. **Metric Definition:** The question asks for 'total assets' for 'Waste Connections Inc.' in fiscal year 2022.  'Total assets' represents the sum of all resources owned or controlled by the company, expected to provide future economic benefits.\n2. **Context Examination:** The context includes 'Consolidated Balance Sheets' (page 78), a standard financial statement that reports a company's assets, liabilities, and equity.\n3. **Metric Matching:** On page 78, under 'December 31, 2022', a line item labeled 'Total assets' exists.  This directly matches the concept requested in the question.\n4. **Value Extraction and Adjustment:** The value for 'Total assets' is '$18,500,342'. The context indicates this is in thousands of dollars.  Therefore, the full value is 18,500,342,000.\n5. **Confirmation**: No calculation beyond unit adjustment was needed. The reported metric directly matches the question.",
  "reasoning_summary": "The 'Total assets' value for fiscal year 2022 was directly found on the 'Consolidated Balance Sheets' (page 78). The reported value was in thousands, requiring multiplication by 1000 for the final answer.",
  "relevant_pages": [78],
  "final_answer": 18500342000
}
```


Example 2:
Question:
"For Ritter Pharmaceuticals, Inc., what was the value of Research and development equipment, at cost at the end of the period listed in annual report?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks for 'Research and development equipment, at cost' for Ritter Pharmaceuticals, Inc. This indicates a specific value from the balance sheet, representing the *original purchase price* of equipment specifically used for R&D, *without* any accumulated depreciation.\n2. The context (page 35) shows 'Property and equipment, net' at $12,500.  This is a *net* value (after depreciation), and it's a *broader* category, encompassing all property and equipment, not just R&D equipment.\n3. The context (page 37) also mentions 'Accumulated Depreciation' of $110,000 for 'Machinery and Equipment'. This represents the total *depreciation*, not the original cost, and, importantly, it doesn't specify that this equipment is *exclusively* for R&D.\n4. Neither of these metrics *exactly* matches the requested metric. 'Property and equipment, net' is too broad and represents the depreciated value. 'Accumulated Depreciation' only shows depreciation, not cost, and lacks R&D specificity.\n5. Since the context doesn't provide the *original cost* of *only* R&D equipment, and we cannot make assumptions, perform calculations, or combine information, the answer is 'N/A'.",
  "reasoning_summary": "The context lacks a specific line item for 'Research and development equipment, at cost.' 'Property and equipment, net' is depreciated and too broad, while 'Accumulated Depreciation' only represents depreciation, not original cost, and is not R&D-specific. Strict matching requires 'N/A'.",
  "relevant_pages": [ 35, 37 ],
  "final_answer": "N/A"
}
```
"""


BOOLEAN_EXAMPLE = r"""
Question:
"Did W. P. Carey Inc. announce any changes to its dividend policy in the annual report?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks whether W. P. Carey Inc. announced changes to its dividend policy.\n2. The phrase 'changes to its dividend policy' requires careful interpretation. It means any adjustment to the framework, rules, or stated intentions that dictate how the company determines and distributes dividends.\n3. The context (page 12, 18) states that the company increased its annualized dividend to $4.27 per share in the fourth quarter of 2023, compared to $4.22 per share in the same period of 2022. Page 45 mentions further details about dividend.\n4. Consistent, incremental increases throughout the year, with explicit mentions of maintaining a 'steady and growing' dividend, indicates no changes to *policy*, though the *amount* increased as planned within the existing policy.",
  "reasoning_summary": "The context highlights consistent, small increases to the dividend throughout the year, consistent with a stated policy of providing a 'steady and growing' dividend. While the dividend *amount* changed, the *policy* governing those increases remained consistent. The question asks about *policy* changes, not amount changes.",
  "relevant_pages": [12, 18, 45],
  "final_answer": False
}
```
"""


NAMES_EXAMPLE = r"""
Example:
Question:
"What are the names of all new executives that took on new leadership positions in company?"

Answer:
```
{
    "step_by_step_analysis": "1. The question asks for the names of all new executives who took on new leadership positions in the company.\n2. Exhibit 10.9 and 10.10, as listed in the Exhibit Index on page 89, mentions new Executive Agreements with Carly Kennedy and Brian Appelgate.\n3. Exhibit 10.9, Employment Agreement with Carly Kennedy, states her start date as April 4, 2022, and her position as Executive Vice President and General Counsel.\n4. Exhibit 10.10, Offer Letter with Brian Appelgate shows that his new role within the company is Interim Chief Operations Officer, and he was accepting the offer on November 8, 2022.\n5. Based on the documents, Carly Kennedy and Brian Appelgate are named as the new executives.",
    "reasoning_summary": "Exhibits 10.9 and 10.10 of the annual report, described as Employment Agreement and Offer Letter, explicitly name Carly Kennedy and Brian Appelgate taking on new leadership roles within the company in 2022.",
    "relevant_pages": [
        89
    ],
    "final_answer": [
        "Carly Kennedy",
        "Brian Appelgate"
    ]
}
```
"""


def build_answer_system_prompt(example: str) -> str:
    return ANSWER_INSTRUCTION.strip() + "\n\n---\n\n" + example.strip()


AnswerKind = Literal["name", "number", "boolean", "names"]

ANSWER_PROMPTS: dict[str, tuple[str, type[BaseModel]]] = {
    "name": (build_answer_system_prompt(NAME_EXAMPLE), NameAnswer),
    "number": (build_answer_system_prompt(NUMBER_EXAMPLE), NumberAnswer),
    "boolean": (build_answer_system_prompt(BOOLEAN_EXAMPLE), BooleanAnswer),
    "names": (build_answer_system_prompt(NAMES_EXAMPLE), NamesAnswer),
}


# ---------------------------------------------------------------------------
# Web answering: fallback when the knowledge base cannot answer
# Same answer types and final_answer rules as above, with web results as the context
# ---------------------------------------------------------------------------

WEB_ANSWER_INSTRUCTION = """
You are a question answering system that works from web search results.
The question could not be answered from the company annual reports in the knowledge base, so
you receive numbered web search results (title, URL, snippet) instead.

Answer only from these results. Prefer official sources such as the company's own website,
investor relations pages and regulatory filings over news and aggregator sites.
If the results do not contain the answer, say so: return 'N/A', or False for yes/no questions.
Numbers must be written out in full units: "$394.3 billion" -> 394300000000, "$2.1 million" -> 2100000.

Before giving a final answer, carefully think out loud and step by step.
"""

WEB_ANSWER_USER_PROMPT = """
Here are the web search results:
\"\"\"
{results}
\"\"\"

---

Here is the question:
"{question}"
"""


def _web_schema(kind_schema: type[BaseModel]) -> type[BaseModel]:
    """Web answer schema for one answer type, reusing that type's final_answer field."""
    final = kind_schema.model_fields["final_answer"]
    return create_model(
        f"Web{kind_schema.__name__}",
        step_by_step_analysis=(
            str,
            Field(description="Step-by-step analysis of the search results, at least 3 steps."),
        ),
        reasoning_summary=(
            str,
            Field(description="Concise summary of the reasoning. Around 50 words."),
        ),
        used_results=(
            list[int],
            Field(
                description="Numbers of the search results the answer is based on. Empty if the answer is N/A."
            ),
        ),
        final_answer=(final.annotation, Field(description=final.description)),
    )


WEB_ANSWER_SCHEMAS: dict[str, type[BaseModel]] = {
    kind: _web_schema(schema) for kind, (_, schema) in ANSWER_PROMPTS.items()
}


# ---------------------------------------------------------------------------
# Answer type classification: for free-form questions that come without a kind
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """
You classify a question about a company by the type of answer it expects.

- number: a single numeric value (amount, count, percentage, ratio, per-share value)
- boolean: a yes/no question ("Did ...", "Does ...", "Is there ...")
- name: one name (a person, company, product, position or place)
- names: a list of names ("Which ...", "What are the names of ...", "List ...")

If several types fit, choose the most specific one the question asks for.
"""


class QuestionKind(BaseModel):
    kind: Literal["number", "boolean", "name", "names"]
