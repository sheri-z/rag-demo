"""Extract document metadata with an LLM and build the company routing list.

For each report, the first pages (cover, table of contents, intro) go to the LLM with a
structured output schema. The result is compared with the organizer metadata in subset.csv,
which the organizers also extracted with an LLM.

The routing list maps every name a question might use (organizer name, legal name,
short names, tickers) to the report's sha1. Questions in this dataset use the organizer
names, so those are always included.

Output: data/erc2/metadata.json

Usage:
    uv run python -m rag_demo.tag
"""

import csv
import json
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from rag_demo.clean import CLEAN_DIR
from rag_demo.config import get_openai, get_settings
from rag_demo.download import DATA_DIR

METADATA_PATH = DATA_DIR / "metadata.json"

# Same industry list as the organizers' question generator, so results are comparable.
Industry = Literal[
    "Technology", "Financial Services", "Healthcare", "Automotive",
    "Retail", "Energy and Utilities", "Hospitality", "Telecommunications",
    "Media & Entertainment", "Pharmaceuticals", "Aerospace & Defense",
    "Transport & Logistics", "Food & Beverage",
]  # fmt: skip

MAX_CHARS = 15_000  # enough for cover + contents + intro, about 4k tokens
MAX_PAGES = 8


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


SYSTEM_PROMPT = (
    "You extract metadata from the first pages of a company annual report. "
    "Use only information stated in the text. For the industry, choose the closest option. "
    "For end_of_period, prefer the 'Period statements' section: it lists the most frequent "
    "'year ended <date>' phrases in the whole report."
)

# The first pages often name the report year without the exact period end
# (e.g. AA "Annual Report 2021" covers the year ended 31 January 2022).
PERIOD_PHRASE = re.compile(
    r"(?:year|period|months|financial year) end(?:ed|ing)\s+(?:on\s+)?"
    r"(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
)


def load_pages(sha1: str) -> list[dict]:
    return json.loads((CLEAN_DIR / f"{sha1}.json").read_text())["pages"]


def period_statements(pages: list[dict], top: int = 3) -> str:
    counts = Counter(m for p in pages for m in PERIOD_PHRASE.findall(p["text"]))
    return "\n".join(f"- '{phrase}' ({n} times)" for phrase, n in counts.most_common(top))


def build_input(sha1: str) -> str:
    pages = load_pages(sha1)
    return f"{first_pages(pages)}\n\n--- Period statements ---\n{period_statements(pages)}"


def first_pages(pages: list[dict]) -> str:
    parts, total = [], 0
    for p in pages[:MAX_PAGES]:
        if total >= MAX_CHARS:
            break
        text = p["text"][: MAX_CHARS - total]
        parts.append(f"--- page {p['page']} ---\n{text}")
        total += len(text)
    return "\n\n".join(parts)


def extract(sha1: str) -> DocumentMetadata:
    response = get_openai().chat.completions.parse(
        model=get_settings().llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_input(sha1)},
        ],
        response_format=DocumentMetadata,
    )
    return response.choices[0].message.parsed


LEGAL_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "ltd", "limited", "plc"}


def normalize(name: str) -> str:
    return " ".join(name.lower().replace(",", " ").replace(".", " ").split())


def same_company(a: str, b: str) -> bool:
    """Compare names ignoring legal suffixes, so 'HCA Healthcare' == 'HCA Healthcare, Inc.'."""

    def core(name: str) -> list[str]:
        words = normalize(name).split()
        while len(words) > 1 and words[-1] in LEGAL_SUFFIXES:
            words.pop()
        return words

    return core(a) == core(b)


def routing_names(organizer_name: str, meta: DocumentMetadata) -> list[str]:
    names = [organizer_name, meta.company_name, *meta.aliases]
    seen, result = set(), []
    for n in names:
        key = normalize(n)
        if key and key not in seen:
            seen.add(key)
            result.append(n.strip())
    return result


def main() -> None:
    organizer = {r["sha1"]: r for r in csv.DictReader(open(DATA_DIR / "subset.csv"))}
    results = []
    print(f"{'report':8}  {'name':5} {'cur':5} {'industry':8}  extracted")
    for sha1, row in organizer.items():
        meta = extract(sha1)
        checks = {
            "company_name": same_company(meta.company_name, row["company_name"]),
            "currency": meta.currency == row["cur"],
            "major_industry": meta.major_industry == row["major_industry"],
        }
        results.append(
            {
                "sha1": sha1,
                "organizer": {
                    "company_name": row["company_name"],
                    "currency": row["cur"],
                    "major_industry": row["major_industry"],
                },
                "extracted": meta.model_dump(),
                "matches_organizer": checks,
                "routing_names": routing_names(row["company_name"], meta),
            }
        )
        mark = lambda ok: "ok" if ok else "DIFF"  # noqa: E731
        print(
            f"{sha1[:8]}  {mark(checks['company_name']):5} {mark(checks['currency']):5} "
            f"{mark(checks['major_industry']):8}  {meta.company_name} | {meta.currency} | "
            f"{meta.major_industry} | {meta.end_of_period.year}-{meta.end_of_period.month:02}"
        )

    METADATA_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for field in ["company_name", "currency", "major_industry"]:
        n = sum(r["matches_organizer"][field] for r in results)
        print(f"{field}: {n}/{len(results)} match the organizer metadata")
    print(f"-> {METADATA_PATH}")


if __name__ == "__main__":
    main()
