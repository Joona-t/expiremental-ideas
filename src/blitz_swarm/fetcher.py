"""HTTP fetching utilities for explicit URL ingestion."""

from __future__ import annotations

import asyncio
import re
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.request import Request, urlopen

from blitz_swarm.models import DocumentSource
from blitz_swarm.utils import stable_id

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    httpx = None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.title: str = ""
        self._in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


class HTTPUrlFetcher:
    async def fetch(self, url: str) -> DocumentSource:
        content_type, body = await self._fetch_bytes(url)
        text, title = self._extract_text(body, content_type)
        return DocumentSource(
            source_id=stable_id("src", url),
            uri=url,
            source_type="url",
            title=title or url,
            content=text,
            metadata={"content_type": content_type},
        )

    async def _fetch_bytes(self, url: str) -> tuple[str, str]:
        if httpx is not None:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.headers.get("content-type", "text/plain"), response.text
        return await asyncio.to_thread(self._fetch_bytes_sync, url)

    def _fetch_bytes_sync(self, url: str) -> tuple[str, str]:
        request = Request(url, headers={"User-Agent": "BlitzSwarm/0.1"})
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get_content_type()
                return content_type, body
        except URLError as exc:  # pragma: no cover - network behavior
            raise RuntimeError(f"Failed to fetch URL '{url}': {exc}") from exc

    def _extract_text(self, body: str, content_type: str) -> tuple[str, str]:
        if "html" not in content_type:
            return body.strip(), ""
        parser = _HTMLTextExtractor()
        parser.feed(body)
        return parser.text(), parser.title
