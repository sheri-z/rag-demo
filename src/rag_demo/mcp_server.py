"""MCP server exposing a web search tool.

The RAG workflow calls this server when the knowledge base cannot answer: the question names
no company in the knowledge base, or no retrieved page is relevant enough.

Any MCP client can use it (the LangGraph workflow, Kiro, Claude Desktop, ...). It speaks MCP
over stdio: the client starts this script as a subprocess and exchanges JSON-RPC messages
on stdin/stdout. So nothing may be printed to stdout here except protocol messages.

The search backend is DuckDuckGo (no API key). Only `search()` depends on it, so switching
to another provider such as Tavily means replacing that one function.

Usage:
    uv run python -m rag_demo.mcp_server                  # start the server (stdio)
    uv run mcp dev src/rag_demo/mcp_server.py             # try it in the MCP Inspector
"""

import logging
import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

MAX_RESULTS_LIMIT = 10
SEARCH_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0

logger = logging.getLogger(__name__)

mcp = FastMCP("web-search", log_level="WARNING")

# The search library logs every HTTP request at INFO level.
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("ddgs").setLevel(logging.WARNING)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


def search(query: str, max_results: int) -> list[SearchResult]:
    # ddgs queries several search engines and raises "No results found" when they all come
    # back empty, which happens now and then for ordinary queries. Retry before giving up,
    # and report "no results" as an empty list rather than as a tool error.
    for attempt in range(SEARCH_ATTEMPTS):
        try:
            hits = DDGS().text(query, max_results=max_results)
            break
        except DDGSException as exc:
            logger.warning("search attempt %d failed: %s", attempt + 1, exc)
            time.sleep(RETRY_DELAY_SECONDS)
    else:
        return []
    return [SearchResult(title=h["title"], url=h["href"], snippet=h["body"]) for h in hits]


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web and return result titles, URLs and text snippets.

    Use for questions the annual report knowledge base cannot answer, such as companies
    outside the knowledge base or recent events.

    Args:
        query: Search query in plain words.
        max_results: Number of results, 1 to 10.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return search(query, max(1, min(max_results, MAX_RESULTS_LIMIT)))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
