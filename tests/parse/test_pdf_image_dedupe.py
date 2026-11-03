# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""PDF 图片跨页去重与装饰图剔除单测（EXEPD-228951 A-1）。

现场：同一份 PDF 每页页眉是同一张 116×116 logo，逐字节相同但 URI 不同，
旧实现把去重集合建在页内，于是每页都留下一条图片资产，EAIP 把它们当成三张不同的图下发。

这里锁定四件事：
1. 跨页同字节只留第一次出现；
2. 「跨 ≥3 页 + 最短边 ≤200px」的模板图整类剔除，并删掉已保存的文件；
3. 单张小图（只出现一次）与大图（重复但够大）都不能被尺寸规则误删；
4. 图片注释前缀保持不变，追加 w/h/sha256 供下游按内容去重。
"""

import struct
import zlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openviking.parse.parsers.pdf import (
    DECORATIVE_MAX_SHORT_SIDE_PT,
    DECORATIVE_MAX_SHORT_SIDE_PX,
    DECORATIVE_REPEAT_PAGES,
    PDFParser,
)


class _FakePage:
    """Minimal pdfplumber page stub（与 test_pdf_bookmark_extraction 同形态）。"""

    def __init__(self, text: str):
        self._text = text
        self.images = []

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return []

    def flush_cache(self):
        pass

    def close(self):
        self.flush_cache()


def _png(width: int, height: int) -> bytes:
    """生成一张只有 IHDR 有意义的合法 PNG（尺寸解析只读 IHDR）。"""
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _bbox(top: float, left: float = 0.0, width: float = 100.0) -> dict:
    return {"x0": left, "top": top, "x1": left + width, "bottom": top + width}


def _storage() -> MagicMock:
    storage = MagicMock()
    storage.media_dir = Path("/media")

    def save(_resource_name, _data, filename=None):
        return Path(f"/media/dummy/images/{filename}.png")

    storage.save_image.side_effect = save
    return storage


async def _convert(parser: PDFParser, pages, renders, storage=None):
    fake_pdf = SimpleNamespace(pages=pages)
    fake_pdfplumber = SimpleNamespace(open=lambda _path: nullcontext(fake_pdf))
    storage = storage or _storage()
    with (
        patch("openviking.parse.parsers.pdf.lazy_import", return_value=fake_pdfplumber),
        patch.object(parser, "_extract_bookmarks", return_value=[]),
        patch.object(parser, "_detect_headings_by_font", return_value=[]),
        patch.object(
            parser,
            "_extract_image_from_page",
            side_effect=lambda _page, img: renders[(img["x0"], img["top"])],
        ),
    ):
        markdown, meta = await parser._convert_local(
            "dummy.pdf", storage=storage, resource_name="dummy"
        )
    return markdown, meta, storage


@pytest.mark.asyncio
async def test_repeated_small_logo_is_dropped_entirely():
    """页眉 logo 每页重复且很小 ⇒ 整类剔除；真实外观图保留。"""
    logo = _png(116, 116)
    figure = _png(1750, 1254)
    pages = [_FakePage(f"Page {n}") for n in range(1, 4)]
    # 页眉 logo 每页都有；真实外观图只出现在第 1 页
    pages[0].images = [_bbox(0.0), _bbox(200.0, width=400.0)]
    pages[1].images = [_bbox(0.0)]
    pages[2].images = [_bbox(0.0)]
    renders = {(0.0, 0.0): logo, (0.0, 200.0): figure}

    markdown, meta, storage = await _convert(PDFParser(), pages, renders)

    assert DECORATIVE_REPEAT_PAGES == 3
    assert DECORATIVE_MAX_SHORT_SIDE_PT == 120
    assert DECORATIVE_MAX_SHORT_SIDE_PX == 600
    # 3 页各一张 logo：整类按模板图剔除（3 页都计入剔除，不再重复计去重）
    assert meta["images_deduplicated"] == 0
    assert meta["images_decorative_dropped"] == 3
    assert meta["images_extracted"] == 1
    assert "sha256=" + __import__("hashlib").sha256(figure).hexdigest() in markdown
    assert "w=1750 h=1254" in markdown
    assert markdown.count("![Page") == 1
    assert markdown.count("<!-- Page 1 Image") == 1


@pytest.mark.asyncio
async def test_single_small_image_is_not_removed_by_size_alone():
    """只出现一次的小图（接口图/图标）不能被尺寸阈值误删。"""
    small = _png(150, 150)
    page = _FakePage("Page one")
    page.images = [_bbox(300.0, width=120.0)]

    markdown, meta, _ = await _convert(PDFParser(), [page], {(0.0, 300.0): small})

    assert meta["images_decorative_dropped"] == 0
    assert meta["images_extracted"] == 1
    assert "w=150 h=150" in markdown


@pytest.mark.asyncio
async def test_large_repeated_image_is_kept_once_not_dropped():
    """大图即使跨页重复也只按内容去重，不能套用装饰图规则。"""
    figure = _png(1750, 1254)
    pages = [_FakePage(f"Page {n}") for n in range(1, 4)]
    for page in pages:
        page.images = [_bbox(100.0, width=400.0)]

    markdown, meta, _ = await _convert(PDFParser(), pages, {(0.0, 100.0): figure})

    assert meta["images_deduplicated"] == 2
    assert meta["images_decorative_dropped"] == 0
    assert meta["images_extracted"] == 1
    assert markdown.count("![Page") == 1


@pytest.mark.asyncio
async def test_image_comment_keeps_legacy_prefix_and_adds_metadata():
    """旧前缀 `<!-- Page {n} Image {i}` 必须原样保留，新字段只追加。"""
    figure = _png(640, 480)
    page = _FakePage("Page one")
    page.images = [_bbox(50.0, width=300.0)]

    markdown, _meta, _ = await _convert(PDFParser(), [page], {(0.0, 50.0): figure})

    assert "<!-- Page 1 Image 1 w=640 h=480 sha256=" in markdown
    assert "![Page 1 Image 1](" in markdown


def test_pixel_size_reads_png_header_and_fails_open():
    assert PDFParser._image_pixel_size(_png(116, 116)) == (116, 116)
    assert PDFParser._image_pixel_size(b"not-a-png") == (0, 0)
    assert PDFParser._image_pixel_size(b"") == (0, 0)

def _file_storage(tmp_path: Path) -> MagicMock:
    """真把图片写到磁盘的 storage，用来核对「被剔除的图是否真的删了 / 保留的图还在」。"""
    media = tmp_path / "media"
    storage = MagicMock()
    storage.media_dir = media

    def save(_resource_name, data, filename=None):
        path = media / "images" / f"{filename}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    storage.save_image.side_effect = save
    return storage


@pytest.mark.asyncio
async def test_mixed_size_occurrences_keep_the_whole_digest(tmp_path):
    """同一字节先以小版面出现、后以大版面出现 ⇒ 整类保留（2026-11-03 复核 P1）。

    只看首见那次（旧实现）会把这块字节的全部出现判成页眉 logo 并删文件；
    保守规则要求「每一次出现都小」才剔除。
    """
    icon = _png(116, 116)
    pages = [_FakePage(f"Page {n}") for n in range(1, 4)]
    pages[0].images = [_bbox(0.0)]  # 100pt 小版面：像页眉
    pages[1].images = [_bbox(0.0)]  # 100pt 小版面
    # 第 3 页同一份字节当整页插图用：400pt 版面，远超 DECORATIVE_MAX_SHORT_SIDE_PT
    pages[2].images = [_bbox(400.0, left=40.0, width=400.0)]
    renders = {(0.0, 0.0): icon, (40.0, 400.0): icon}

    markdown, meta, storage = await _convert(
        PDFParser(), pages, renders, storage=_file_storage(tmp_path)
    )

    assert meta["images_decorative_dropped"] == 0, "有一次大版面就不能判成模板图"
    assert meta["images_deduplicated"] == 2, "跨页同字节仍按内容去重，只剩首见那次"
    assert meta["images_extracted"] == 1
    saved = list((tmp_path / "media" / "images").glob("*.png"))
    assert len(saved) == 1, "保留下来的图必须还在盘上"
    assert saved[0].read_bytes() == icon
    assert markdown.count("![Page") == 1
    assert markdown.count("<!-- Page 1 Image") == 1
    assert "w=116 h=116" in markdown
