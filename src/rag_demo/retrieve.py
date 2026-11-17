"""Retrieve the most relevant pages for a question.

Pipeline, following the winning configuration of the original project:
    1. route the question to the company report(s) it names
    2. vector search: top 30 chunks in that company's index
    3. parent pages: map chunks to their pages, deduplicated, each page keeps its best chunk score
    4. LLM reranking: gpt-4o-mini scores pages 2 at a time (0 to 1),
       combined score = 0.7 * LLM score + 0.3 * vector score
    5. return the top 10 pages

Usage:
    uv run python -m rag_demo.retrieve "Did AA Limited announce any new product launches?"
    uv run python -m rag_demo.retrieve --no-rerank "..."
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np

from rag_demo.config import get_openai, get_settings
from rag_demo.ingest import embed
from rag_demo.paths import CLEAN_DIR, INDEX_DIR
from rag_demo.prompts import RERANK_SYSTEM_PROMPT, BlockRankings, rerank_user_prompt
from rag_demo.router import route

SAMPLE_CHUNKS = 30  # chunks taken from vector search before reranking
TOP_PAGES = 10  # pages returned after reranking
RERANK_BATCH = 2  # pages per reranking request (the original code's default)
LLM_WEIGHT = 0.7
RERANK_WORKERS = 8


@dataclass
class RetrievedPage:
    sha1: str
    page: int
    text: str
    vector_score: float
    llm_score: float | None = None
    reasoning: str = ""

    @property
    def score(self) -> float:
        if self.llm_score is None:
            return self.vector_score
        return LLM_WEIGHT * self.llm_score + (1 - LLM_WEIGHT) * self.vector_score


@lru_cache
def _index(sha1: str) -> faiss.Index:
    return faiss.read_index(str(INDEX_DIR / f"{sha1}.faiss"))


@lru_cache
def _chunks(sha1: str) -> list[dict]:
    return json.loads((INDEX_DIR / f"{sha1}.chunks.json").read_text())


@lru_cache
def _pages(sha1: str) -> dict[int, str]:
    pages = json.loads((CLEAN_DIR / f"{sha1}.json").read_text())["pages"]
    return {p["page"]: p["text"] for p in pages}


def vector_search(
    sha1: str, query_vector: np.ndarray, k: int = SAMPLE_CHUNKS
) -> list[RetrievedPage]:
    """Top-k chunks, expanded to their parent pages (best chunk score per page)."""
    scores, ids = _index(sha1).search(query_vector.reshape(1, -1), k)
    best: dict[int, float] = {}
    for score, i in zip(scores[0], ids[0]):
        if i < 0:  # fewer than k chunks in the index
            continue
        page = _chunks(sha1)[i]["page"]
        best[page] = max(best.get(page, -1.0), float(score))
    pages = _pages(sha1)
    results = [RetrievedPage(sha1, p, pages[p], s) for p, s in best.items()]
    return sorted(results, key=lambda r: r.score, reverse=True)


def _rank_batch(query: str, batch: list[RetrievedPage]) -> None:
    response = get_openai().chat.completions.parse(
        model=get_settings().llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": rerank_user_prompt(query, [p.text for p in batch])},
        ],
        response_format=BlockRankings,
    )
    rankings = response.choices[0].message.parsed.block_rankings
    by_number = {r.block_number: r for r in rankings}
    if sorted(by_number) != list(range(1, len(batch) + 1)):
        by_number = dict(enumerate(rankings, 1))  # numbers unusable, fall back to order
    for i, page in enumerate(batch, 1):
        ranking = by_number.get(i)
        page.llm_score = ranking.relevance_score if ranking else 0.0
        page.reasoning = ranking.reasoning if ranking else "missing ranking"


def rerank(query: str, pages: list[RetrievedPage]) -> list[RetrievedPage]:
    batches = [pages[i : i + RERANK_BATCH] for i in range(0, len(pages), RERANK_BATCH)]
    with ThreadPoolExecutor(max_workers=RERANK_WORKERS) as pool:
        list(pool.map(lambda b: _rank_batch(query, b), batches))
    return sorted(pages, key=lambda r: r.score, reverse=True)


def retrieve(question: str, top_n: int = TOP_PAGES, use_rerank: bool = True) -> list[RetrievedPage]:
    """Pages for a single-company question. Empty if the question names no known company."""
    companies = route(question)
    if not companies:
        return []
    if len(companies) > 1:
        raise NotImplementedError("questions about several companies are not supported yet")
    query_vector = embed([question])[0]
    pages = vector_search(companies[0].sha1, query_vector)
    if use_rerank:
        pages = rerank(question, pages)
    return pages[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top", type=int, default=TOP_PAGES)
    args = parser.parse_args()

    pages = retrieve(args.question, top_n=args.top, use_rerank=not args.no_rerank)
    if not pages:
        print("no known company in the question")
    for p in pages:
        llm = "  -" if p.llm_score is None else f"{p.llm_score:.1f}"
        print(f"page {p.page:>3}  score={p.score:.3f}  vector={p.vector_score:.3f}  llm={llm}")
        if p.reasoning:
            print(f"          {p.reasoning[:150]}")


if __name__ == "__main__":
    main()
