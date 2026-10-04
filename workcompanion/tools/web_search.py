"""Web search tool.

Research answers must separate *document evidence* from *web evidence*, so this
module is deliberately small and dependency-free: it queries DuckDuckGo's
keyless HTML endpoint and returns :class:`WebSource` records that the Research
Agent labels as ``SourceKind.WEB``.

Everything here fails soft.  No network, a blocked request or a parse miss
returns an empty result plus a human-readable reason, never an exception that
would break a research run.
"""

from __future__ import annotations

import html
import re
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.schemas.research import WebSource

logger = get_logger(__name__)

_DDG_LITE = "https://lite.duckduckgo.com/lite/"

_RESULT_RE = re.compile(
    r'<a[^>]+class="result-link"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<td[^>]+class="result-snippet"[^>]*>(?P<snippet>.*?)</td>', re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


class WebSearchTool:
    """Keyless web search with strict timeouts and graceful degradation."""

    name = "duckduckgo"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._last_error: str = ""

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self._settings.enable_web_research) and self._settings.web_search_provider != "none"

    @property
    def last_error(self) -> str:
        return self._last_error

    def describe(self) -> dict[str, Any]:
        return {
            "tool": "web_search",
            "provider": self._settings.web_search_provider,
            "enabled": self.enabled,
            "max_results": self._settings.web_max_results,
            "timeout_s": self._settings.web_search_timeout,
        }

    # ------------------------------------------------------------------
    def search(self, query: str, *, max_results: int | None = None) -> list[WebSource]:
        """Run a search and return normalised sources (possibly empty)."""
        self._last_error = ""
        query = (query or "").strip()
        if not query:
            return []
        if not self.enabled:
            self._last_error = "Web research is disabled in Settings."
            return []

        limit = max(1, min(max_results or self._settings.web_max_results, 20))
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx is a hard dependency
            self._last_error = "httpx is not installed, so web research is unavailable."
            return []

        try:
            response = httpx.post(
                _DDG_LITE,
                data={"q": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=self._settings.web_search_timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            sources = self._parse(response.text, limit=limit)
        except Exception as exc:  # network, DNS, TLS, HTTP status - all non-fatal
            self._last_error = f"Web search failed: {type(exc).__name__}: {exc}"
            logger.warning("[web] search failed for %r: %s", query, exc)
            return []

        if not sources:
            self._last_error = f"DuckDuckGo returned no usable results for {query!r}."
        logger.info("[web] %d result(s) for %r", len(sources), query)
        return sources

    def search_many(
        self, queries: Sequence[str], *, max_results: int | None = None
    ) -> list[WebSource]:
        """Run several queries and merge them, de-duplicated by URL."""
        merged: list[WebSource] = []
        seen: set[str] = set()
        for query in queries:
            if not query.strip():
                continue
            for source in self.search(query, max_results=max_results):
                key = _normalise_url(source.url)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(source)
            time.sleep(0.3)  # be polite to the endpoint
        return merged

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(payload: str, *, limit: int) -> list[WebSource]:
        sources: list[WebSource] = []
        snippets = [_clean(match.group("snippet")) for match in _SNIPPET_RE.finditer(payload)]

        for rank, match in enumerate(_RESULT_RE.finditer(payload), start=1):
            href = html.unescape(match.group("href").strip())
            url = _unwrap_redirect(href)
            if not url.startswith(("http://", "https://")):
                continue
            sources.append(
                WebSource(
                    title=_clean(match.group("title")) or urlparse(url).netloc,
                    url=url,
                    snippet=snippets[rank - 1] if rank - 1 < len(snippets) else "",
                    provider="duckduckgo",
                    rank=rank,
                )
            )
            if len(sources) >= limit:
                break
        return sources


def _clean(html_fragment: str) -> str:
    text = _TAG_RE.sub(" ", html_fragment or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps results in ``/l/?uddg=<encoded target>``."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


def _normalise_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.netloc}{parsed.path.rstrip('/')}".lower()
