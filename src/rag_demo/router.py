"""Route a question to the company reports it asks about.

Uses the routing list from the tagging step (organizer names, legal names, aliases, tickers).
As in the original approach, no LLM is needed: plain string matching is enough when the
list of company names is known.

Matching rules:
    - whole words only, so "AA" does not match inside "AAA" or "Saab"
    - names of 4 characters or less (tickers like "AA", "HCA", "AIN") match case-sensitively,
      so "ain" or "hca" in ordinary text are ignored; longer names match case-insensitively
    - longer names win: in "AA Limited", "AA" is not counted as a separate match

Usage:
    uv run python -m rag_demo.router "Did AA Limited announce any new product launches?"
"""

import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

from rag_demo.paths import METADATA_PATH

SHORT_NAME_MAX = 4


@dataclass(frozen=True)
class Company:
    sha1: str
    name: str  # organizer name, as used in the questions


@dataclass(frozen=True)
class _Pattern:
    regex: re.Pattern
    length: int
    company: Company


def _compile(name: str) -> re.Pattern:
    flags = 0 if len(name) <= SHORT_NAME_MAX else re.IGNORECASE
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", flags)


@lru_cache
def _patterns() -> list[_Pattern]:
    if not METADATA_PATH.exists():
        raise SystemExit(f"{METADATA_PATH} not found; run rag_demo.tag first")
    patterns = []
    for doc in json.loads(METADATA_PATH.read_text()):
        company = Company(sha1=doc["sha1"], name=doc["organizer"]["company_name"])
        for name in doc["routing_names"]:
            patterns.append(_Pattern(_compile(name), len(name), company))
    return sorted(patterns, key=lambda p: p.length, reverse=True)


def companies() -> list[Company]:
    seen = {p.company.sha1: p.company for p in _patterns()}
    return list(seen.values())


def route(question: str) -> list[Company]:
    """Return the companies mentioned in the question, in order of first mention."""
    taken: list[tuple[int, int]] = []
    found: dict[str, tuple[int, Company]] = {}
    for p in _patterns():  # longest names first
        for m in p.regex.finditer(question):
            span = m.span()
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue  # part of a longer name already matched
            taken.append(span)
            sha1 = p.company.sha1
            if sha1 not in found or span[0] < found[sha1][0]:
                found[sha1] = (span[0], p.company)
    return [company for _, company in sorted(found.values(), key=lambda x: x[0])]


def main() -> None:
    question = " ".join(sys.argv[1:])
    for company in route(question):
        print(f"{company.sha1[:8]}  {company.name}")


if __name__ == "__main__":
    main()
