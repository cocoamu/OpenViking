# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""EXEPD-230545: every parser must publish the same image-metadata comment.

knowledge-service only reads image page/size/content-hash from the HTML comment
immediately before the Markdown link, so the comment format is a cross-service
contract. These tests pin the format and the two parser wirings.
"""

import hashlib
import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from openviking.parse.image_metadata import (
    ImageMetadata,
    describe_image,
    format_comment,
    markdown_with_comment,
)
from openviking.parse.parsers.anydoc_renderer import _AnyDocMarkdownRenderer

# Mirrors knowledge-service SnippetText: the comment must contain a standalone
# "Image" word, w=/h= are 1-5 digits, sha256= is exactly 64 hex chars.
IMAGE_MARKUP = re.compile(r"<!--[^>]*\bImage\b[^>]*-->|!\[([^\]]*)]\(([^)]*)\)", re.S)
COMMENT_PAGE = re.compile(r"Page\s+(\d{1,9})(?!\d)")
COMMENT_WIDTH = re.compile(r"\bw=(\d{1,5})\b")
COMMENT_HEIGHT = re.compile(r"\bh=(\d{1,5})\b")
COMMENT_CONTENT_HASH = re.compile(r"\bsha256=([0-9a-fA-F]{64})\b")


def _png(width: int = 40, height: int = 30) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeStorage:
    def __init__(self, media_dir: Path):
        self.media_dir = media_dir

    def save_image(self, resource_name, image_data, filename=None, extension=".png"):
        media_dir = Path(self.media_dir)
        media_dir.mkdir(parents=True, exist_ok=True)
        path = media_dir / f"{filename or 'image'}{extension}"
        path.write_bytes(image_data)
        return path


def _renderer(tmp_path: Path, image_bytes: bytes, *, media_type: str = "image/png"):
    document = SimpleNamespace(blocks=[], notes=[], assets=[])
    renderer = _AnyDocMarkdownRenderer(
        document,
        source_format="pptx",
        resource_name="课件",
        storage=_FakeStorage(tmp_path / "media"),
    )
    renderer.assets = {7: SimpleNamespace(id=7, media_type=media_type, data=image_bytes,
                                          origin_part="ppt/media/image1.png")}
    image = SimpleNamespace(alt="", source=SimpleNamespace(kind="asset", asset_id=7))
    return renderer, image


def _render(renderer, image, *, context="block", in_label=False):
    return renderer._render_image(image, context=context, in_label=in_label)


class TestCommentFormat:
    def test_office_comment_is_parseable_by_knowledge_service(self):
        data = _png()
        comment = format_comment(describe_image(data))

        assert IMAGE_MARKUP.fullmatch(comment), comment
        assert COMMENT_CONTENT_HASH.search(comment).group(1) == hashlib.sha256(data).hexdigest()
        assert COMMENT_WIDTH.search(comment).group(1) == "40"
        assert COMMENT_HEIGHT.search(comment).group(1) == "30"
        assert COMMENT_PAGE.search(comment) is None  # Office has no page concept

    def test_pdf_comment_keeps_page_label_and_gains_identity(self):
        comment = format_comment(ImageMetadata(sha256="a" * 64, width=1750, height=1254),
                                 page=7, index=2)

        assert IMAGE_MARKUP.fullmatch(comment), comment
        assert COMMENT_PAGE.search(comment).group(1) == "7"
        assert COMMENT_WIDTH.search(comment).group(1) == "1750"
        assert COMMENT_CONTENT_HASH.search(comment).group(1) == "a" * 64

    def test_undecodable_bytes_still_yield_content_hash(self):
        data = b"not-an-image"
        metadata = describe_image(data)

        assert metadata.width is None and metadata.height is None
        comment = format_comment(metadata)
        assert "w=" not in comment and "h=" not in comment
        assert COMMENT_CONTENT_HASH.search(comment).group(1) == hashlib.sha256(data).hexdigest()

    def test_markdown_glue_keeps_comment_adjacent(self):
        glued = markdown_with_comment("<!-- Image sha256=" + "a" * 64 + " -->", "![x](y)")

        assert glued.splitlines() == ["<!-- Image sha256=" + "a" * 64 + " -->", "![x](y)"]


class TestAnydocRendererWiring:
    def test_block_image_gets_metadata_comment(self, tmp_path):
        rendered = _render(*_renderer(tmp_path, _png()))

        lines = rendered.splitlines()
        assert IMAGE_MARKUP.fullmatch(lines[0]), rendered
        assert lines[1].startswith("![")
        assert "anydoc_asset_7.png" in lines[1]

    def test_inline_and_label_images_stay_plain(self, tmp_path):
        renderer, image = _renderer(tmp_path, _png())

        assert not _render(renderer, image, context="table").startswith("<!--")
        assert not _render(renderer, image, in_label=True).startswith("<!--")

    def test_rejected_image_has_no_comment(self, tmp_path):
        # Too small to ingest (image_validation rejects < 14px sides).
        rendered = _render(*_renderer(tmp_path, _png(4, 4)))

        # 被 image_validation 拒绝的图不能留下"注释 + 无图片"的悬空元数据
        assert "<!--" not in rendered
        assert "anydoc_asset_7" not in rendered


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
