"""Answer a question from retrieved pages, with a prompt chosen by answer type.

Each answer type (name, number, boolean, names) has its own system prompt, example and
structured output schema. The schema makes the model reason step by step first, then
summarize, then list the pages it used, then give the final answer, in that order.

Page references are checked against the retrieved pages: pages the model cites but never
saw are dropped. (The original also padded references to at least 2 pages, a scoring
tactic for the competition, which is left out here.)

Usage:
    uv run python -m rag_demo.answer --kind boolean "Did AA Limited announce new product launches?"
"""

import argparse
from dataclasses import asdict, dataclass, field

from rag_demo.config import get_openai, get_settings
from rag_demo.prompts import ANSWER_PROMPTS, ANSWER_USER_PROMPT, AnswerKind
from rag_demo.retrieve import RetrievedPage, retrieve
from rag_demo.router import route


@dataclass
class Answer:
    question: str
    kind: str
    value: str | float | int | bool | list[str]
    references: list[dict] = field(default_factory=list)  # [{"sha1", "page"}]
    reasoning_summary: str = ""
    step_by_step_analysis: str = ""
    company: str | None = None
    retrieved_pages: list[int] = field(default_factory=list)
    dropped_pages: list[int] = field(default_factory=list)  # cited but not retrieved

    def to_dict(self) -> dict:
        return asdict(self)


def format_context(pages: list[RetrievedPage]) -> str:
    parts = [f'Text retrieved from page {p.page}: \n"""\n{p.text}\n"""' for p in pages]
    return "\n\n---\n\n".join(parts)


def generate(question: str, kind: AnswerKind, pages: list[RetrievedPage]):
    system_prompt, schema = ANSWER_PROMPTS[kind]
    response = get_openai().chat.completions.parse(
        model=get_settings().answer_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": ANSWER_USER_PROMPT.format(
                    context=format_context(pages), question=question
                ),
            },
        ],
        response_format=schema,
    )
    return response.choices[0].message.parsed


def answer(question: str, kind: AnswerKind) -> Answer:
    companies = route(question)
    pages = retrieve(question)
    if not pages:
        return Answer(
            question,
            kind,
            value=False if kind == "boolean" else "N/A",
            reasoning_summary="The question does not name a company in the knowledge base.",
        )

    result = generate(question, kind, pages)
    sha1 = pages[0].sha1
    retrieved = [p.page for p in pages]
    cited = list(dict.fromkeys(result.relevant_pages))  # dedupe, keep order
    return Answer(
        question=question,
        kind=kind,
        value=result.final_answer,
        references=[{"sha1": sha1, "page": p} for p in cited if p in retrieved],
        reasoning_summary=result.reasoning_summary,
        step_by_step_analysis=result.step_by_step_analysis,
        company=companies[0].name,
        retrieved_pages=retrieved,
        dropped_pages=[p for p in cited if p not in retrieved],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--kind", required=True, choices=sorted(ANSWER_PROMPTS))
    args = parser.parse_args()

    result = answer(args.question, args.kind)
    print(f"answer:     {result.value!r}")
    print(f"pages:      {[r['page'] for r in result.references]}")
    if result.dropped_pages:
        print(f"dropped:    {result.dropped_pages} (cited but not retrieved)")
    print(f"summary:    {result.reasoning_summary}")


if __name__ == "__main__":
    main()
