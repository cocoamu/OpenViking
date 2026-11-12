# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""EXEPD-230092：PDF 抽出的图片要能带上内容描述。

背景：alt 一直是 "Page N Image M" 这种占位，检索侧拿不到"这张图画的是什么"，
于是模型看不见没有正文说明的实物图（主唛样式图、罐体对比图）。
"""

import asyncio
from pathlib import Path

import pytest

from openviking.parse.parsers.pdf import PDFParser
from openviking_cli.utils.config.parser_config import PDFConfig


class _FakeStorage:
    def __init__(self, media_dir: Path):
        self.media_dir = media_dir


class _FakeResult:
    def __init__(self, abstract: str):
        self.abstract = abstract


class _FakeProcessor:
    def __init__(self, abstract: str = "一排七个罐体，标注 500FP 到 1000FP"):
        self.abstract = abstract
        self.calls = 0
        self.last_context = ""
        self.last_instruction = ""

    async def understand_image(self, image, context="", instruction=""):
        self.calls += 1
        self.last_context = context
        self.last_instruction = instruction
        return _FakeResult(self.abstract)


@pytest.fixture()
def media(tmp_path):
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "page10_img2.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return _FakeStorage(tmp_path / "media")


def _parser(caption_on: bool) -> PDFParser:
    return PDFParser(PDFConfig(image_caption=caption_on, image_caption_max_chars=40))


def test_alt_keeps_placeholder_when_switch_off(media, monkeypatch):
    parser = _parser(False)
    md = "<!-- Page 10 Image 2 -->\n![Page 10 Image 2](page10_img2.png)"
    out = asyncio.run(parser._caption_images(md, media))
    assert out == md, "开关关着时逐字不变"


def test_alt_gets_caption_when_switch_on(media, monkeypatch):
    import openviking.parse.vlm as vlm_module

    fake = _FakeProcessor()
    monkeypatch.setattr(vlm_module, "VLMProcessor", lambda *a, **k: fake)

    parser = _parser(True)
    md = "<!-- Page 10 Image 2 -->\n![Page 10 Image 2](page10_img2.png)"
    out = asyncio.run(parser._caption_images(md, media))

    assert "![一排七个罐体" in out, out
    assert "page10_img2.png" in out, "链接不能变，只换 alt"
    assert fake.calls == 1


def test_placeholder_abstract_is_not_written(media, monkeypatch):
    import openviking.parse.vlm as vlm_module

    monkeypatch.setattr(vlm_module, "VLMProcessor", lambda *a, **k: _FakeProcessor("[Image]"))

    parser = _parser(True)
    md = "![Page 10 Image 2](page10_img2.png)"
    out = asyncio.run(parser._caption_images(md, media))
    assert out == md, "模型没识别出来时保留原 alt，不写占位符"


def test_missing_file_keeps_placeholder(media, monkeypatch):
    import openviking.parse.vlm as vlm_module

    monkeypatch.setattr(vlm_module, "VLMProcessor", lambda *a, **k: _FakeProcessor())

    parser = _parser(True)
    md = "![Page 3 Image 1](not_there.png)"
    out = asyncio.run(parser._caption_images(md, media))
    assert out == md


def test_normal_parse_resolves_default_storage(media, monkeypatch, tmp_path):
    import openviking.parse.vlm as vlm_module
    import openviking_cli.utils.storage as storage_module

    fake = _FakeProcessor()
    monkeypatch.setattr(vlm_module, "VLMProcessor", lambda: fake)
    monkeypatch.setattr(storage_module, "get_storage", lambda: media)
    parser = _parser(True)
    md = "<!-- Page 10 Image 2 -->\n![Page 10 Image 2](page10_img2.png)"
    monkeypatch.setattr(parser, "_convert_local_sync", lambda *args: (md, {}))
    out, _ = asyncio.run(parser._convert_to_markdown(tmp_path / "input.pdf"))
    assert out == "<!-- Page 10 Image 2 -->\n![一排七个罐体，标注 500FP 到 1000FP](page10_img2.png)"
    assert fake.calls == 1


def test_caption_is_grounded_in_same_page_text(media, monkeypatch):
    """EXEPD-230261：图注要带上同页正文里的说法，模型才知道叉号是「不可水洗」。"""
    import openviking.parse.vlm as vlm_module

    fake = _FakeProcessor("不可水洗——水洗槽打叉")
    monkeypatch.setattr(vlm_module, "VLMProcessor", lambda *a, **k: fake)

    parser = _parser(True)
    md = (
        "<!-- Page 81 Table 1 -->\n"
        "| 洗涤标识认知 | 水洗 | 不可水洗 | 最高洗涤30℃ | 最高洗涤40℃ |\n"
        "<!-- Page 81 Image 2 -->\n"
        "![Page 81 Image 2](page10_img2.png)"
    )
    out = asyncio.run(parser._caption_images(md, media))

    assert "![不可水洗——水洗槽打叉]" in out, out
    assert "不可水洗" in fake.last_context, "同页正文必须作为图注上下文：" + fake.last_context
    assert "名称" in fake.last_instruction, "要显式要求先给名称：" + fake.last_instruction


def test_page_text_context_stops_at_next_page(media):
    parser = _parser(True)
    md = (
        "<!-- Page 1 -->\n第一页正文：悬挂滴干\n"
        "<!-- Page 2 -->\n第二页正文：不可熨烫\n"
        "![Page 1 Image 1](page10_img2.png)"
    )
    ctx = parser._page_text_context(md, "1")
    assert "悬挂滴干" in ctx
    assert "不可熨烫" not in ctx, "不能把下一页的正文串进来：" + ctx
