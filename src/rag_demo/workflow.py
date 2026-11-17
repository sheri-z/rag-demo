"""LangGraph workflow: answer from the knowledge base, fall back to web search over MCP.

    START -> route -> retrieve -> answer_from_kb -> END
               |          |
               |          +-- (no page relevant enough) --+
               +-- (no known company) --------------------+-> web_search -> answer_from_web -> END

Fallback happens only when the knowledge base clearly cannot answer:
    - the question names no company in the knowledge base (or several, not supported yet)
    - the best reranked page scores below MIN_RELEVANCE
An N/A answer from the knowledge base does not trigger a web search: for many questions N/A is the
correct answer ("the report does not state this"), and a number from the web would be wrong.

Web search goes through our own MCP server (mcp_server.py), started as a subprocess and called
via langchain-mcp-adapters, the same way any MCP client would call it.

Usage:
    uv run python -m rag_demo.workflow --kind number "What was Apple's revenue in fiscal 2022?"
    uv run python -m rag_demo.workflow --mermaid      # print the graph as a Mermaid diagram
"""

import argparse
import asyncio
import sys
from typing import TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from rag_demo.answer import Answer, answer_from_pages
from rag_demo.config import get_openai, get_settings
from rag_demo.prompts import (
    ANSWER_PROMPTS,
    WEB_ANSWER_INSTRUCTION,
    WEB_ANSWER_SCHEMAS,
    WEB_ANSWER_USER_PROMPT,
    AnswerKind,
)
from rag_demo.retrieve import RetrievedPage, retrieve
from rag_demo.router import route

# Reranking scale: 0.3 = "slightly relevant". On the 17 test questions the best page always scores
# at least 0.4, including questions about metrics the report does not contain, so a higher
# threshold would send answerable and correctly-N/A questions to the web.
MIN_RELEVANCE = 0.3
WEB_RESULTS = 5

MCP_SERVERS = {
    "web": {
        "command": sys.executable,
        "args": ["-m", "rag_demo.mcp_server"],
        "transport": "stdio",
    }
}


class State(TypedDict, total=False):
    question: str
    kind: AnswerKind
    company: str | None
    pages: list[RetrievedPage]
    fallback_reason: str | None
    web_results: list[dict]
    answer: Answer
    trace: list[str]


def _trace(state: State, step: str) -> list[str]:
    return [*state.get("trace", []), step]


# --- nodes -------------------------------------------------------------------


def route_node(state: State) -> State:
    companies = route(state["question"])
    if not companies:
        return {
            "company": None,
            "fallback_reason": "no company from the knowledge base in the question",
            "trace": _trace(state, "route: no known company"),
        }
    if len(companies) > 1:
        names = ", ".join(c.name for c in companies)
        return {
            "company": None,
            "fallback_reason": f"several companies ({names}), not supported yet",
            "trace": _trace(state, f"route: several companies ({names})"),
        }
    return {"company": companies[0].name, "trace": _trace(state, f"route: {companies[0].name}")}


async def retrieve_node(state: State) -> State:
    pages = await asyncio.to_thread(retrieve, state["question"])
    best = max((p.llm_score or 0.0) for p in pages)
    update: State = {
        "pages": pages,
        "trace": _trace(state, f"retrieve: {len(pages)} pages, best relevance {best:.1f}"),
    }
    if best < MIN_RELEVANCE:
        update["fallback_reason"] = f"best page relevance {best:.1f} < {MIN_RELEVANCE}"
    return update


async def answer_from_kb_node(state: State) -> State:
    result = await asyncio.to_thread(
        answer_from_pages, state["question"], state["kind"], state["pages"], state["company"]
    )
    return {"answer": result, "trace": _trace(state, "answer: knowledge base")}


async def web_search_node(state: State) -> State:
    tools = await MultiServerMCPClient(MCP_SERVERS).get_tools()
    web_search = next(t for t in tools if t.name == "web_search")
    message = await web_search.ainvoke(
        {
            "type": "tool_call",
            "id": "web_search",
            "name": "web_search",
            "args": {"query": state["question"], "max_results": WEB_RESULTS},
        }
    )
    results = (message.artifact or {}).get("structured_content", {}).get("result", [])
    return {
        "web_results": results,
        "trace": _trace(state, f"web_search (MCP): {len(results)} results"),
    }


def _answer_from_web(question: str, kind: AnswerKind, results: list[dict]):
    formatted = "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}" for i, r in enumerate(results, 1)
    )
    response = get_openai().chat.completions.parse(
        model=get_settings().answer_model,
        temperature=0,
        messages=[
            {"role": "system", "content": WEB_ANSWER_INSTRUCTION.strip()},
            {
                "role": "user",
                "content": WEB_ANSWER_USER_PROMPT.format(results=formatted, question=question),
            },
        ],
        response_format=WEB_ANSWER_SCHEMAS[kind],
    )
    return response.choices[0].message.parsed


async def answer_from_web_node(state: State) -> State:
    question, kind, results = state["question"], state["kind"], state.get("web_results", [])
    if not results:
        value = False if kind == "boolean" else "N/A"
        result = Answer(question, kind, value, reasoning_summary="Web search found nothing.")
    else:
        parsed = await asyncio.to_thread(_answer_from_web, question, kind, results)
        used = [results[i - 1] for i in parsed.used_results if 1 <= i <= len(results)]
        if parsed.final_answer == "N/A":
            used = []
        result = Answer(
            question=question,
            kind=kind,
            value=parsed.final_answer,
            reasoning_summary=parsed.reasoning_summary,
            step_by_step_analysis=parsed.step_by_step_analysis,
            company=state.get("company"),
            source="web",
            web_sources=[{"title": r["title"], "url": r["url"]} for r in used],
        )
    return {"answer": result, "trace": _trace(state, "answer: web")}


# --- edges -------------------------------------------------------------------


def after_route(state: State) -> str:
    return "web_search" if state.get("fallback_reason") else "retrieve"


def after_retrieve(state: State) -> str:
    return "web_search" if state.get("fallback_reason") else "answer_from_kb"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer_from_kb", answer_from_kb_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("answer_from_web", answer_from_web_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ["retrieve", "web_search"])
    graph.add_conditional_edges("retrieve", after_retrieve, ["answer_from_kb", "web_search"])
    graph.add_edge("answer_from_kb", END)
    graph.add_edge("web_search", "answer_from_web")
    graph.add_edge("answer_from_web", END)
    return graph.compile()


GRAPH = build_graph()


async def run(question: str, kind: AnswerKind) -> State:
    return await GRAPH.ainvoke({"question": question, "kind": kind, "trace": []})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--kind", choices=sorted(ANSWER_PROMPTS), default="name")
    parser.add_argument("--mermaid", action="store_true", help="print the graph and exit")
    args = parser.parse_args()

    if args.mermaid:
        print(GRAPH.get_graph().draw_mermaid())
        return
    if not args.question:
        parser.error("question is required")

    state = asyncio.run(run(args.question, args.kind))
    a = state["answer"]
    for step in state["trace"]:
        print(f"  - {step}")
    if state.get("fallback_reason"):
        print(f"fallback:  {state['fallback_reason']}")
    print(f"answer:    {a.value!r}  (source: {a.source})")
    if a.source == "knowledge_base":
        print(f"pages:     {[r['page'] for r in a.references]}")
    for s in a.web_sources:
        print(f"web:       {s['title']} <{s['url']}>")
    print(f"summary:   {a.reasoning_summary}")


if __name__ == "__main__":
    main()
