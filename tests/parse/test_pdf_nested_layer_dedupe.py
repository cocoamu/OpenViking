# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""扫描页「整页底图 + 嵌套文字块图」的去重单测（EXEPD-230436）。

现场：dev《终端员工奖惩管理办法--签批版.pdf》7 页共 86 张图，每页都是一张整页底图
外加十几张完全落在底图内的文字块图。检索命中 page1_img4（2.1~2.2.2 第一行），
块边界正好切在「一次大功奖励人民币｜300 元。」之间，模型只看到前半句，
回答「大功金额文字截断」。而整页底图 page1_img1 的摘要里是完整的 300 元。

这里锁定四件事：
1. 无文本层 + 整页底图 + 完全包含 ⇒ 嵌套层不再单独出图，只留底图；
2. 有文本层的页面不启用该规则（图可能是有用的插图）；
3. 整页底图抽不出来时 fail-open，逐块照旧，绝不丢整页内容；
4. 部分越出底图的图（不是重复层）照常保留。
"""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openviking.parse.parsers.pdf import PDFParser

PAGE_W = 595.0
PAGE_H = 841.0
COVER = {"x0": 0.0, "top": -0.7, "x1": 595.2, "bottom": 841.0}
STRIP_A = {"x0": 40.3, "top": 55.0, "x1": 529.9, "bottom": 271.0}
STRIP_B = {"x0": 69.1, "top": 338.2, "x1": 543.4, "bottom": 469.7}


class _FakePage:
    """最小 pdfplumber page stub。"""

    def __init__(self, text: str, width: float = PAGE_W, height: float = PAGE_H):
        self._text = text
        self.width = width
        self.height = height
        self.images = []

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return []

    def flush_cache(self):
        pass

    def close(self):
        self.flush_cache()


def _storage() -> MagicMock:
    storage = MagicMock()
    storage.media_dir = Path("/media")

    def save(_resource_name, _data, filename=None):
        return Path(f"/media/dummy/images/{filename}.png")

    storage.save_image.side_effect = save
    return storage


async def _convert(parser: PDFParser, pages, renders):
    fake_pdf = SimpleNamespace(pages=pages)
    fake_pdfplumber = SimpleNamespace(open=lambda _path: nullcontext(fake_pdf))
    with (
        patch("openviking.parse.parsers.pdf.lazy_import", return_value=fake_pdfplumber),
        patch.object(parser, "_extract_bookmarks", return_value=[]),
        patch.object(parser, "_detect_headings_by_font", return_value=[]),
        patch.object(
            parser,
            "_extract_image_from_page",
            side_effect=lambda _page, img: renders.get((img["x0"], img["top"])),
        ),
    ):
        markdown, meta = await parser._convert_local(
            "dummy.pdf", storage=_storage(), resource_name="dummy"
        )
    return markdown, meta


@pytest.mark.asyncio
async def test_scanned_page_keeps_only_page_covering_image():
    """纯扫描页：只保留整页底图，完全落在它内部的文字块图不再单独出图。"""
    page = _FakePage("")
    page.images = [COVER, STRIP_A, STRIP_B]
    renders = {
        (COVER["x0"], COVER["top"]): b"cover",
        (STRIP_A["x0"], STRIP_A["top"]): b"a",
        (STRIP_B["x0"], STRIP_B["top"]): b"b",
    }

    markdown, meta = await _convert(PDFParser(), [page], renders)

    assert meta["images_extracted"] == 1
    assert meta["images_nested_dropped"] == 2
    assert markdown.count("![Page") == 1
    assert "![Page 1 Image 1]" in markdown


@pytest.mark.asyncio
async def test_text_page_keeps_nested_layers():
    """有文本层的页面不启用：图可能是真实插图，不能被底图规则吞掉。"""
    page = _FakePage("这是正文")
    page.images = [COVER, STRIP_A, STRIP_B]
    renders = {
        (COVER["x0"], COVER["top"]): b"cover",
        (STRIP_A["x0"], STRIP_A["top"]): b"a",
        (STRIP_B["x0"], STRIP_B["top"]): b"b",
    }

    markdown, meta = await _convert(PDFParser(), [page], renders)

    assert meta["images_extracted"] == 3
    assert meta["images_nested_dropped"] == 0


@pytest.mark.asyncio
async def test_cover_extraction_failure_keeps_all_layers():
    """底图抽不出来 ⇒ fail-open，逐块照旧，整页内容不能丢。"""
    page = _FakePage("")
    page.images = [COVER, STRIP_A, STRIP_B]
    fake_pdf = SimpleNamespace(pages=[page])
    fake_pdfplumber = SimpleNamespace(open=lambda _path: nullcontext(fake_pdf))

    def extract(_page, img):
        if (img["x0"], img["top"]) == (COVER["x0"], COVER["top"]):
            raise RuntimeError("render failed")
        return f"strip-{img['x0']}".encode()

    parser = PDFParser()
    with (
        patch("openviking.parse.parsers.pdf.lazy_import", return_value=fake_pdfplumber),
        patch.object(parser, "_extract_bookmarks", return_value=[]),
        patch.object(parser, "_detect_headings_by_font", return_value=[]),
        patch.object(parser, "_extract_image_from_page", side_effect=extract),
    ):
        markdown, meta = await parser._convert_local(
            "dummy.pdf", storage=_storage(), resource_name="dummy"
        )

    assert meta["images_nested_dropped"] == 0
    assert meta["images_extracted"] == 2
    assert markdown.count("![Page") == 2


@pytest.mark.asyncio
async def test_image_partly_outside_cover_is_kept():
    """不是完全包含的图（越出底图边界）照常保留。"""
    page = _FakePage("")
    outside = {"x0": 500.0, "top": 300.0, "x1": 700.0, "bottom": 500.0}
    page.images = [COVER, STRIP_A, outside]
    renders = {
        (COVER["x0"], COVER["top"]): b"cover",
        (STRIP_A["x0"], STRIP_A["top"]): b"a",
        (outside["x0"], outside["top"]): b"outside",
    }

    markdown, meta = await _convert(PDFParser(), [page], renders)

    assert meta["images_nested_dropped"] == 1
    assert meta["images_extracted"] == 2
    assert "![Page 1 Image 3]" in markdown


def test_page_covering_index_requires_near_full_page():
    images = [
        {"x0": 0.0, "top": 0.0, "x1": 100.0, "bottom": 100.0},
        {"x0": 10.0, "top": 10.0, "x1": 90.0, "bottom": 90.0},
    ]
    assert PDFParser._page_covering_image_index(images, 100.0, 100.0) == 0
    assert PDFParser._page_covering_image_index(images[1:], 100.0, 100.0) is None
    assert PDFParser._page_covering_image_index(images, 0.0, 100.0) is None
