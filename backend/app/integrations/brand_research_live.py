"""Live brand research: a small tool-calling agent over Jina Search.

The agent is a LangGraph loop of ``bind_tools`` and ``ToolNode``. Code enforces
the search budget: extra tool calls are dropped, and the loop ends when the
budget is spent. A separate step then writes the notes from the search results,
with no tools bound and no tool messages in the prompt: asked to answer inside
the loop with ``tool_choice="none"``, the live model wrote fake tool calls or
junk instead. Notes exist only when a search found something, so they cannot
come from the model's memory alone.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..models import FacebookPage
from ..prompts import research_messages, research_notes_messages
from .brand_research import BrandResearch, WebSearchConfigError, WebSearchError
from .llm_live import (
    LlmError,
    LlmSettings,
    _reasoning_extra_body,
    _with_reasoning_message,
    redact_secret,
)
from .web_reader_live import _is_jina_retryable_status

# Each Jina search costs about 10,000 tokens from the key's balance.
MAX_SEARCHES = 3

_SEARCH_URL = "https://s.jina.ai/"
_MAX_HITS = 5
_SNIPPET_CHARS = 500
_NOTES_CHARS = 4_000


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


class WebSearch(Protocol):
    def search(self, query: str) -> list[SearchHit]:
        """Return the top results for one query."""
        ...


class JinaWebSearch:
    """Titles, URLs, and snippets from s.jina.ai, without reading each page."""

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff: float = 1.0,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.getenv("JINA_API_KEY", "")
        ).strip()
        if not self._api_key:
            raise WebSearchConfigError(
                "JINA_API_KEY is not set. Export it before using live web search."
            )
        self._client = client or httpx.Client(timeout=timeout)
        self._max_retries = max_retries
        self._backoff = backoff

    def search(self, query: str) -> list[SearchHit]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "X-Respond-With": "no-content",
        }
        attempts = self._max_retries + 1
        reason = ""
        for attempt in range(attempts):
            try:
                resp = self._client.get(
                    _SEARCH_URL, params={"q": query}, headers=headers
                )
            except httpx.TransportError as exc:
                reason = type(exc).__name__
            else:
                if resp.status_code < 300:
                    return _parse_hits(resp)
                reason = f"HTTP {resp.status_code}"
                if not _is_jina_retryable_status(resp.status_code):
                    raise WebSearchError(f"Jina search failed: {reason}")
            if attempt < attempts - 1:
                time.sleep(self._backoff * 2**attempt)
        raise WebSearchError(f"Jina search failed after {attempts} attempts: {reason}")


def _parse_hits(resp: httpx.Response) -> list[SearchHit]:
    try:
        payload = resp.json()
    except ValueError:
        raise WebSearchError("Jina search returned invalid JSON") from None
    rows = payload.get("data") if isinstance(payload, dict) else None
    hits: list[SearchHit] = []
    for row in rows if isinstance(rows, list) else []:
        url = row.get("url") if isinstance(row, dict) else None
        if not url:
            continue
        hits.append(
            SearchHit(
                title=str(row.get("title") or ""),
                url=str(url),
                snippet=str(row.get("description") or "")[:_SNIPPET_CHARS],
            )
        )
    return hits[:_MAX_HITS]


def make_web_search_tool(search: WebSearch) -> BaseTool:
    """Build the tool. The artifact records the URLs each search really returned."""

    @tool(response_format="content_and_artifact")
    def web_search(query: str) -> tuple[str, list[str]]:
        """Search the public web. Returns the title, URL, and snippet of top results."""
        hits = search.search(query)
        if not hits:
            return "No results.", []
        text = "\n\n".join(
            f"{hit.title}\n{hit.url}\n{hit.snippet}" for hit in hits
        )
        return text, [hit.url for hit in hits]

    return web_search


def _tool_messages(state: MessagesState) -> list[ToolMessage]:
    return [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]


def _capped(reply: AIMessage, remaining: int) -> AIMessage:
    """Drop tool calls past the budget, since a model can ask for many at once."""
    if len(reply.tool_calls) <= remaining:
        return reply
    extra = {
        key: value
        for key, value in reply.additional_kwargs.items()
        if key != "tool_calls"
    }
    return reply.model_copy(
        update={"tool_calls": reply.tool_calls[:remaining], "additional_kwargs": extra}
    )


def build_research_agent(
    model: BaseChatModel,
    search: WebSearch,
    *,
    max_searches: int = MAX_SEARCHES,
    prepare: Callable[[list[dict[str, str]]], list[dict[str, str]]] = list,
):
    """Compile the search loop and the step that writes notes from its results."""
    tools = [make_web_search_tool(search)]
    may_search = model.bind_tools(tools)

    def agent(state: MessagesState) -> dict[str, Any]:
        remaining = max_searches - len(_tool_messages(state))
        return {"messages": [_capped(may_search.invoke(state["messages"]), remaining)]}

    def after_tools(state: MessagesState) -> Literal["agent", "write_notes"]:
        return "agent" if len(_tool_messages(state)) < max_searches else "write_notes"

    def write_notes(state: MessagesState) -> dict[str, Any]:
        results = [m.text for m in _tool_messages(state) if m.artifact]
        if not results:
            return {}
        brand = next(m for m in state["messages"] if isinstance(m, HumanMessage))
        return {
            "messages": [model.invoke(prepare(research_notes_messages(brand.text, results)))]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    # A failed search goes back to the model. Anything else is a bug and raises.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=(WebSearchError,)))
    graph.add_node("write_notes", write_notes)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: "write_notes"}
    )
    graph.add_conditional_edges("tools", after_tools, ["agent", "write_notes"])
    graph.add_edge("write_notes", END)
    return graph.compile()


class LiveBrandResearcher:
    """Runs the research agent for one brand and returns its notes."""

    def __init__(
        self,
        settings: LlmSettings | None = None,
        *,
        api_key: str | None = None,
        model: BaseChatModel | None = None,
        search: WebSearch | None = None,
        max_searches: int = MAX_SEARCHES,
    ) -> None:
        self._settings = settings or LlmSettings.from_env(api_key=api_key)
        self._agent = build_research_agent(
            model or _chat_model(self._settings),
            search or JinaWebSearch(),
            max_searches=max_searches,
            prepare=self._prepare,
        )

    def _prepare(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return _with_reasoning_message(messages, self._settings)

    def research(
        self,
        website_url: str,
        website_content: str | None,
        facebook_page: FacebookPage | None,
    ) -> BrandResearch:
        messages = self._prepare(
            research_messages(website_url, website_content, facebook_page)
        )
        try:
            final = self._agent.invoke({"messages": messages})
        except openai.OpenAIError as exc:
            # Drop the cause, which can keep the Authorization header.
            detail = redact_secret(f"{type(exc).__name__}: {exc}", self._settings.api_key)
            raise LlmError(f"LLM request failed: {detail[:500]}") from None

        tool_messages = _tool_messages(final)
        found = any(message.artifact for message in tool_messages)
        notes = final["messages"][-1].text.strip() if found else ""
        return BrandResearch(
            notes=notes[:_NOTES_CHARS] or None,
            searches=len(tool_messages),
            llm_calls=sum(isinstance(m, AIMessage) for m in final["messages"]),
        )


def _chat_model(settings: LlmSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        extra_body=_reasoning_extra_body(settings),
    )
