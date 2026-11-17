# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""EXEPD-230092 / EXEPD-230261：PDF 抽出的图片必须带可读语义 alt。

契约：凡是进入入库 markdown 的图片引用，其 alt 必须非空、非 "Page N Image M"、
非 "[Image]"、非纯文档名/文件名。唯一例外是「同页无文本线索且 VLM 失败」的图，
此时保留占位但必须计入 uncaptioned 并以聚合日志暴露。

分层：已有摘要复用 → VLM（最多两轮）→ 同页标题/首句兜底 → 计数保留。
"""

import asyncio
from pathlib import Path

import pytest

from openviking.parse.parsers.pdf import (
    ImageCaptionInvariantError,
    ImageCaptionReport,
    PDFParser,
    assert_semantic_image_alts,
    audit_image_alts,
    is_semantic_caption,
)
from openviking_cli.utils.config.parser_config import PDFConfig

DOC_NAME = "常用产品面料"

PAGE_MD = """# 2. 洗涤说明

<!-- Page 10 -->
最高洗涤30℃，不可漂白，悬挂滴干。

<!-- Page 10 Image 2 -->
![Page 10 Image 2](page10_img2.png)"""


class _FakeStorage:
    def __init__(self, media_dir: Path):
        self.media_dir = media_dir


class _FakeResult:
    def __init__(self, abstract: str):
        self.abstract = abstract


class _FakeProcessor:
    """按脚本返回 abstract；Exception 实例会被抛出，callable 会被 await。"""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = 0
        self.contexts = []
        self.instructions = []

    async def understand_image(self, image, context="", instruction=""):
        self.calls += 1
        self.contexts.append(context)
        self.instructions.append(instruction)
        item = self.results.pop(0) if self.results else "[Image]"
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return _FakeResult(item)


@pytest.fixture()
def media(tmp_path):
    directory = tmp_path / "media"
    directory.mkdir()
    for name in ("page10_img2.png", "page11_img1.png", "page11_img2.png", "page12_img1.png"):
        (directory / name).write_bytes(name.encode())
    return _FakeStorage(directory)


@pytest.fixture()
def install_vlm(monkeypatch):
    import openviking.parse.vlm as vlm_module

    def install(processor):
        monkeypatch.setattr(vlm_module, "VLMProcessor", lambda *args, **kwargs: processor)
        return processor

    return install


def _parser(**overrides) -> PDFParser:
    settings = {"image_caption": True, "image_caption_max_chars": 40}
    settings.update(overrides)
    return PDFParser(PDFConfig(**settings))


def _caption(parser, markdown, media, meta=None):
    return asyncio.run(
        parser._caption_images(markdown, media, meta=meta, document_name=DOC_NAME)
    )


# ── 不变量 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "alt",
    [
        "",
        "   ",
        "W",
        "→",
        "40",
        "40.5",
        "40%",
        "→→",
        "Page 10 Image 2",
        "PAGE 2 IMAGE 1",
        "page 10",
        "[Image]",
        "[image]",
        "[图片]",
        "图片",
        "image",
        "undefined",
        "null",
        "N/A",
        "page10_img2.png",
        "page10_img2",
        "a/b.png",
        "0123456789abcdef0123",
    ],
)
def test_contract_rejects_non_semantic_alts(alt):
    assert not is_semantic_caption(alt, DOC_NAME)


@pytest.mark.parametrize(
    "alt",
    [
        "悬挂滴干（图 3）",
        "最高洗涤30℃",
        "最高洗涤40℃（图 2）",
        "一排七个罐体，标注 500FP 到 1000FP",
        "不可水洗",
        "W30",
        "30℃",
        "水柔棉/丝光棉/精梳棉",
    ],
)
def test_contract_accepts_semantic_alts(alt):
    assert is_semantic_caption(alt, DOC_NAME)


def test_contract_rejects_document_name_and_image_filename_context():
    assert not is_semantic_caption("常用产品面料", DOC_NAME)
    assert not is_semantic_caption("常用产品面料.pdf", "常用产品面料.pdf")
    assert not is_semantic_caption("page10_img2", DOC_NAME, "page10_img2.png")
    assert is_semantic_caption("最高洗涤40℃（图 2）", DOC_NAME, "page10_img2.png")


def test_contract_truncates_long_caption_before_judging():
    long_caption = "最高洗涤40℃" + "说" * 200
    assert is_semantic_caption(long_caption, DOC_NAME)
    assert not is_semantic_caption("→" * 200)
    assert not is_semantic_caption("a")


def test_audit_lists_invalid_alts_and_assertion_raises():
    markdown = (
        "![Page 1 Image 1](a.png)\n"
        "![](b.png)\n"
        "![Image](c.png)\n"
        "![常用产品面料](d.png)\n"
        "![悬挂滴干（图 3）](e.png)"
    )
    audit = audit_image_alts(markdown, DOC_NAME)
    assert audit.total == 5
    assert audit.semantic == 1
    assert audit.uncaptioned == 4
    assert not audit.valid
    with pytest.raises(ImageCaptionInvariantError):
        assert_semantic_image_alts(markdown, DOC_NAME)


def test_report_count_equation_is_asserted():
    report = ImageCaptionReport(images_total=4, captioned=1, fallback=1, reused=1, uncaptioned=1)
    report.assert_valid()
    broken = ImageCaptionReport(images_total=4, captioned=1, fallback=1, reused=1, uncaptioned=0)
    assert broken.problems()
    with pytest.raises(ImageCaptionInvariantError):
        broken.assert_valid()


# ── 分层行为 ──────────────────────────────────────────────────────────────


def test_switch_off_is_verbatim(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["不该被调用"]))
    parser = _parser(image_caption=False)
    meta = {}
    out = _caption(parser, PAGE_MD, media, meta)
    assert out == PAGE_MD
    assert meta == {}
    assert processor.calls == 0


def test_first_attempt_success_and_meta_counts(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["悬挂滴干"]))
    parser = _parser()
    meta = {}
    out = _caption(parser, PAGE_MD, media, meta)

    assert out.endswith("![悬挂滴干](page10_img2.png)")
    assert processor.calls == 1
    assert meta["images_total"] == 1
    assert meta["captioned"] == 1
    assert meta["fallback"] == 0
    assert meta["reused"] == 0
    assert meta["uncaptioned"] == 0
    assert meta["vlm_calls"] == 1
    assert meta["vlm_retries"] == 0
    assert_semantic_image_alts(out, DOC_NAME)


def test_vlm_context_carries_page_heading_and_body(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["悬挂滴干"]))
    _caption(_parser(), PAGE_MD, media)
    assert "2. 洗涤说明" in processor.contexts[0]
    assert "最高洗涤30℃" in processor.contexts[0]


def test_placeholder_then_retry_with_stricter_prompt(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["[Image]", "悬挂滴干"]))
    parser = _parser()
    meta = {}
    out = _caption(parser, PAGE_MD, media, meta)

    assert out.endswith("![悬挂滴干](page10_img2.png)")
    assert processor.calls == 2
    assert processor.instructions[0] != processor.instructions[1]
    assert "再仔细看这张图" in processor.instructions[1]
    assert meta["captioned"] == 1
    assert meta["vlm_calls"] == 2
    assert meta["vlm_retries"] == 1


def test_retry_is_bounded_and_falls_back_to_section_heading(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["[Image]", ""]))
    parser = _parser()
    meta = {}
    out = _caption(parser, PAGE_MD, media, meta)

    assert processor.calls == 2, "空/占位最多再试一次"
    assert "![2. 洗涤说明（图 2）](page10_img2.png)" in out
    assert "![Page 10 Image 2]" not in out
    assert meta["fallback"] == 1
    assert meta["uncaptioned"] == 0
    assert_semantic_image_alts(out, DOC_NAME)


def test_exceptions_twice_fall_back(media, install_vlm):
    processor = install_vlm(
        _FakeProcessor([RuntimeError("vlm down"), RuntimeError("vlm down again")])
    )
    meta = {}
    out = _caption(_parser(), PAGE_MD, media, meta)
    assert processor.calls == 2
    assert "（图 2）" in out
    assert meta["fallback"] == 1


def test_timeout_falls_back_without_hanging(media, install_vlm):
    async def slow():
        await asyncio.sleep(0.2)
        return _FakeResult("迟到的结果")

    processor = install_vlm(_FakeProcessor([slow, slow]))
    parser = _parser(image_caption_timeout=0.01)
    meta = {}
    out = _caption(parser, PAGE_MD, media, meta)
    assert processor.calls == 2
    assert "（图 2）" in out
    assert meta["fallback"] == 1


def test_fallback_uses_first_sentence_when_no_heading(media, install_vlm):
    markdown = (
        "<!-- Page 10 -->\n最高洗涤30℃，不可漂白，悬挂滴干。\n\n"
        "<!-- Page 10 Image 2 -->\n![Page 10 Image 2](page10_img2.png)"
    )
    install_vlm(_FakeProcessor(["[Image]", ""]))
    meta = {}
    out = _caption(_parser(), markdown, media, meta)
    assert "![最高洗涤30℃，不可漂白，悬挂滴干。（图 2）](page10_img2.png)" in out
    assert meta["fallback"] == 1


def test_no_clue_keeps_placeholder_but_counts_it(media, install_vlm):
    markdown = (
        "<!-- Page 11 Image 1 -->\n![Page 11 Image 1](page11_img1.png)"
    )
    install_vlm(_FakeProcessor(["[Image]", ""]))
    parser = _parser()
    meta = {}
    out = _caption(parser, markdown, media, meta)

    assert out == markdown, "无线索时保留占位，绝不编造"
    assert meta["images_total"] == 1
    assert meta["uncaptioned"] == 1
    assert meta["fallback"] == 0
    assert meta["uncaptioned_samples"] == ["page11_img1.png"]
    assert len(audit_image_alts(out, DOC_NAME).invalid) == 1


def test_document_name_heading_is_not_used_as_fallback(media, install_vlm):
    markdown = (
        "# 常用产品面料\n\n<!-- Page 12 -->\n最高洗涤30℃，不可漂白。\n\n"
        "<!-- Page 12 Image 1 -->\n![Page 12 Image 1](page12_img1.png)"
    )
    install_vlm(_FakeProcessor(["[Image]", ""]))
    meta = {}
    out = _caption(_parser(), markdown, media, meta)
    assert "最高洗涤30℃" in out and "（图 1）" in out
    assert "![常用产品面料" not in out
    assert meta["fallback"] == 1


def test_missing_image_file_still_gets_traceable_alt(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["不会被调用"]))
    markdown = PAGE_MD.replace("page10_img2.png", "missing.png")
    meta = {}
    out = _caption(_parser(), markdown, media, meta)
    assert processor.calls == 0
    assert "2. 洗涤说明（图 2）" in out
    assert meta["fallback"] == 1
    assert meta["errors"] == 1


# ── 复用 ──────────────────────────────────────────────────────────────────


def test_existing_semantic_alt_is_reused_without_vlm(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["新结果"]))
    markdown = PAGE_MD.replace("![Page 10 Image 2]", "![一排七个罐体，标注 500FP 到 1000FP]")
    meta = {}
    out = _caption(_parser(), markdown, media, meta)
    assert out == markdown
    assert processor.calls == 0
    assert meta["reused"] == 1
    assert meta["captioned"] == 0


def test_reuse_disabled_recaptions_existing_alt(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["新的名称"]))
    markdown = PAGE_MD.replace("![Page 10 Image 2]", "![旧名称]")
    meta = {}
    out = _caption(_parser(image_caption_reuse_summary=False), markdown, media, meta)
    assert out.endswith("![新的名称](page10_img2.png)")
    assert processor.calls == 1
    assert meta["captioned"] == 1


def test_reuse_disabled_never_downgrades_existing_semantic_alt(media, install_vlm):
    install_vlm(_FakeProcessor(["[Image]", ""]))
    markdown = PAGE_MD.replace("![Page 10 Image 2]", "![旧名称]")
    meta = {}
    out = _caption(_parser(image_caption_reuse_summary=False), markdown, media, meta)
    assert out.endswith("![旧名称](page10_img2.png)")
    assert meta["reused"] == 1
    assert meta["fallback"] == 0


def test_same_image_bytes_share_one_vlm_call(media, install_vlm):
    shared = media.media_dir / "shared.png"
    shared.write_bytes(b"identical-image-bytes")
    markdown = (
        "# 洗涤说明\n\n<!-- Page 10 -->\n最高洗涤30℃。\n\n"
        "<!-- Page 10 Image 1 -->\n![Page 10 Image 1](shared.png)\n\n"
        "# 干燥说明\n\n<!-- Page 11 -->\n悬挂滴干。\n\n"
        "<!-- Page 11 Image 1 -->\n![Page 11 Image 1](shared.png)"
    )
    processor = install_vlm(_FakeProcessor(["圆形洗涤图标"]))
    meta = {}
    out = _caption(_parser(), markdown, media, meta)
    assert processor.calls == 1, "相同图片内容只问一次 VLM"
    assert out.count("![圆形洗涤图标](shared.png)") == 2
    assert meta["captioned"] == 1
    assert meta["reused"] == 1
    assert meta["images_total"] == 2


def test_second_pass_is_idempotent_and_calls_no_vlm(media, install_vlm):
    install_vlm(_FakeProcessor(["悬挂滴干"]))
    parser = _parser()
    first, first_meta = asyncio.run(
        parser._caption_images_with_report(PAGE_MD, media, document_name=DOC_NAME)
    )
    assert first_meta.captioned == 1

    import openviking.parse.vlm as vlm_module

    class _MustNotRun:
        async def understand_image(self, *args, **kwargs):
            raise AssertionError("second pass must reuse existing captions")

    original = vlm_module.VLMProcessor
    vlm_module.VLMProcessor = lambda *args, **kwargs: _MustNotRun()
    try:
        second, second_meta = asyncio.run(
            parser._caption_images_with_report(first, media, document_name=DOC_NAME)
        )
    finally:
        vlm_module.VLMProcessor = original

    assert second == first
    assert second_meta.captioned == 0
    assert second_meta.reused == 1
    assert second_meta.vlm_calls == 0


# ── 入口与 meta ───────────────────────────────────────────────────────────


def test_convert_to_markdown_writes_counts_into_meta(media, install_vlm, monkeypatch, tmp_path):
    import openviking_cli.utils.storage as storage_module

    install_vlm(_FakeProcessor(["悬挂滴干"]))
    monkeypatch.setattr(storage_module, "get_storage", lambda: media)
    monkeypatch.setattr(
        "openviking.parse.document_worker.should_isolate", lambda *args, **kwargs: False
    )

    parser = _parser()
    monkeypatch.setattr(parser, "_convert_local_sync", lambda *args: (PAGE_MD, {"pages_processed": 1}))
    out, meta = asyncio.run(parser._convert_to_markdown(tmp_path / "input.pdf"))

    assert out.endswith("![悬挂滴干](page10_img2.png)")
    assert meta["pages_processed"] == 1
    assert meta["images_total"] == 1
    assert meta["captioned"] == 1
    assert meta["uncaptioned"] == 0


def test_no_images_reports_zero_counts(media, install_vlm):
    processor = install_vlm(_FakeProcessor(["不该被调用"]))
    meta = {}
    out = _caption(_parser(), "# 只有文字\n\n没有图片。", media, meta)
    assert out == "# 只有文字\n\n没有图片。"
    assert processor.calls == 0
    assert meta["images_total"] == 0
    assert meta["captioned"] == meta["fallback"] == meta["uncaptioned"] == 0


def test_vlm_constructor_failure_degrades_to_page_fallback(media, monkeypatch):
    import openviking.parse.vlm as vlm_module

    class _BrokenVLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vlm unavailable")

    monkeypatch.setattr(vlm_module, "VLMProcessor", _BrokenVLM)
    meta = {}
    out = _caption(_parser(), PAGE_MD, media, meta)
    assert "![2. 洗涤说明（图 2）](page10_img2.png)" in out
    assert meta["fallback"] == 1
    assert meta["vlm_calls"] == 0


def test_caption_pass_exception_does_not_fail_parse(media, monkeypatch):
    parser = _parser()

    async def _explode(*args, **kwargs):
        raise RuntimeError("unexpected caption bug")

    monkeypatch.setattr(parser, "_caption_images_with_report", _explode)
    meta = {"pages_processed": 1}
    out = _caption(parser, PAGE_MD, media, meta)
    assert out == PAGE_MD, "图注异常时保留原 markdown"
    assert meta["caption_error"].startswith("unexpected caption bug")
    assert meta["pages_processed"] == 1


def test_identical_image_failure_is_not_retried_twice(media, install_vlm):
    shared = media.media_dir / "shared_fail.png"
    shared.write_bytes(b"identical-failing-image")
    markdown = (
        "# 洗涤说明\n\n<!-- Page 10 -->\n最高洗涤30℃。\n\n"
        "<!-- Page 10 Image 1 -->\n![Page 10 Image 1](shared_fail.png)\n\n"
        "# 干燥说明\n\n<!-- Page 11 -->\n悬挂滴干。\n\n"
        "<!-- Page 11 Image 1 -->\n![Page 11 Image 1](shared_fail.png)"
    )
    processor = install_vlm(_FakeProcessor(["[Image]", "", "[Image]", ""]))
    meta = {}
    out = _caption(_parser(), markdown, media, meta)

    assert processor.calls == 2, "同一内容图失败后，重复图不再重试 VLM"
    assert meta["vlm_calls"] == 2
    assert meta["fallback"] == 2
    assert meta["uncaptioned"] == 0
    assert "（图 1）" in out and out.count("（图 1）") == 2


def test_leading_image_without_page_uses_next_heading():
    """EXEPD-230261：文档/章节开头的首图（page=None）没有前置标题，用它后面那一节标题兜底。"""
    parser = PDFParser(PDFConfig(image_caption=True))
    markdown = "![Page 1 Image 1](page1_img1.png)\n\n# 店长销售管理办法\n\n正文：店长月度考核与奖惩标准。\n"
    caption = parser._fallback_alt(
        markdown,
        markdown.index("!["),
        "",
        "1",
        "店长销售管理办法-OBC-2026-59.pdf",
        60,
    )
    assert caption == "店长销售管理办法（图 1）"
