# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Bounded, task-local references from native PDF pages for image summaries."""

import asyncio
import re
from typing import Any
from urllib.parse import unquote

from openviking.parse.image_rewrite import HTML_IMG_PATTERN

PDF_IMAGE_NAME = re.compile(r"^page([1-9]\d*)_img[1-9]\d*\.(?:png|jpe?g)$", re.I)
_PAGE = re.compile(r"<!--\s*Page ([1-9]\d*)\s*-->")
_MEDIA = re.compile(r"<!--\s*Page \d+ (?:Image|Table) \d+\s*-->")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^\s()]+)\)")
_MAX_CONTEXT = 1200


def extract_pdf_image_contexts(
    content: str, parent_uri: str, image_names: set[str]
) -> dict[str, str]:
    """Associate only actual sibling references with their matching source page."""
    pages = list(_PAGE.finditer(content))
    result: dict[str, str] = {}
    for index, marker in enumerate(pages):
        end = pages[index + 1].start() if index + 1 < len(pages) else len(content)
        page_content = content[marker.end() : end]
        # Native PDF paragraphs precede extracted tables/images. Keep the original
        # text, avoiding duplicate table renderings and image URI noise.
        text = _MEDIA.split(page_content, maxsplit=1)[0]
        text = HTML_IMG_PATTERN.sub("", text)
        text = _MD_IMAGE.sub("", text)
        text = re.sub(r"<!--[\s\S]*?-->", "", text).strip()
        if len(text) < 80:
            continue
        if len(text) > _MAX_CONTEXT:
            boundary = text.rfind("\n", 0, _MAX_CONTEXT + 1)
            text = text[: boundary if boundary > _MAX_CONTEXT // 2 else _MAX_CONTEXT]
        context = f"Source PDF page {marker.group(1)} text (not image OCR):\n{text}"
        refs = [m.group(2) for m in HTML_IMG_PATTERN.finditer(page_content)]
        refs.extend(m.group(1) for m in _MD_IMAGE.finditer(page_content))
        for ref in refs:
            ref = unquote(ref)
            name = ref.rsplit("/", 1)[-1]
            image = PDF_IMAGE_NAME.fullmatch(name)
            if (
                name not in image_names
                or image is None
                or image.group(1) != marker.group(1)
                or ref not in (name, f"./{name}", f"{parent_uri.rstrip('/')}/{name}")
            ):
                continue
            # A page split across native chunks may have a shorter partial copy.
            if len(context) > len(result.get(name, "")):
                result[name] = context
    return result


class PdfImageContextCache:
    """One executor owns this cache; no data survives or crosses task/ACL scope."""

    MAX_READ_BYTES = 256 * 1024
    MAX_TOTAL_READ_BYTES = 2 * 1024 * 1024
    MAX_MARKDOWN_FILES = 32
    MAX_DIRECTORIES = 64
    MAX_REFERENCES = 2048
    MAX_TOTAL_MARKDOWN_FILES = 128

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()
        self._read_bytes = 0
        self._references = 0
        self._read_files = 0

    async def get(
        self, parent_uri: str, image_name: str, file_paths: list[str], fs: Any, ctx: Any
    ) -> str:
        if not PDF_IMAGE_NAME.fullmatch(image_name):
            return ""
        async with self._lock:
            if parent_uri not in self._cache:
                if len(self._cache) >= self.MAX_DIRECTORIES:
                    return ""
                image_names = {
                    path.rsplit("/", 1)[-1]
                    for path in file_paths
                    if PDF_IMAGE_NAME.fullmatch(path.rsplit("/", 1)[-1])
                }
                contexts: dict[str, str] = {}
                for path in sorted(p for p in file_paths if p.lower().endswith(".md"))[
                    : self.MAX_MARKDOWN_FILES
                ]:
                    budget = min(
                        self.MAX_READ_BYTES, self.MAX_TOTAL_READ_BYTES - self._read_bytes
                    )
                    if (
                        budget <= 0
                        or self._references >= self.MAX_REFERENCES
                        or self._read_files >= self.MAX_TOTAL_MARKDOWN_FILES
                    ):
                        break
                    # Charge attempted reads, including failures, to the fixed budget.
                    self._read_bytes += budget
                    self._read_files += 1
                    try:
                        raw = await fs.read(path, offset=0, size=budget, ctx=ctx)
                        if not isinstance(raw, bytes):
                            continue
                        self._read_bytes -= budget - min(len(raw), budget)
                        text = raw[:budget].decode("utf-8", errors="replace")
                        found = extract_pdf_image_contexts(text, parent_uri, image_names)
                    except Exception:
                        continue  # Context is optional; keep normal image processing.
                    for name, context in found.items():
                        if name in contexts:
                            if len(context) > len(contexts[name]):
                                contexts[name] = context
                        elif self._references < self.MAX_REFERENCES:
                            contexts[name] = context
                            self._references += 1
                self._cache[parent_uri] = contexts
            return self._cache[parent_uri].get(image_name, "")
