import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.parse.parsers.media import utils as media_utils
from openviking.parse.parsers.media.pdf_image_context import (
    PdfImageContextCache,
    extract_pdf_image_contexts,
)
from openviking_cli.utils.config.parser_config import ImageConfig


ROOT = "viking://resources/account/document/chapter"


def page(number=8, ref="page8_img1.png", text=None):
    body = text or "LATTICE晶格科技由无数3D打印小晶格组成，压缩结构形变带来能量聚集和释放。" * 3
    return f'<!-- Page {number} -->\n{body}\n<!-- Page {number} Image 1 -->\n<img src="{ref}" />'


def test_context_is_associated_with_exact_page_and_actual_sibling_reference():
    body = page() + page(80, "page80_img1.png", "泳镜1826332737价格79元，泳帽1826332716价格29元。" * 4)
    contexts = extract_pdf_image_contexts(body, ROOT, {"page8_img1.png", "page80_img1.png"})
    assert "LATTICE" in contexts["page8_img1.png"]
    assert "1826332737" not in contexts["page8_img1.png"]
    assert "79元" in contexts["page80_img1.png"]
    assert "LATTICE" not in contexts["page80_img1.png"]
    assert "not image OCR" in contexts["page8_img1.png"]


@pytest.mark.parametrize("ref", [
    "../page8_img1.png", "https://other/page8_img1.png",
    "viking://resources/other/page8_img1.png", "%2e%2e/page8_img1.png",
    "page9_img1.png", "unreferenced.png",
])
def test_foreign_traversal_wrong_page_and_unreferenced_images_do_not_get_context(ref):
    assert not extract_pdf_image_contexts(page(ref=ref), ROOT, {"page8_img1.png", "page9_img1.png"})


def test_absolute_local_reference_is_supported_and_noise_is_removed():
    context = extract_pdf_image_contexts(page(ref=ROOT + "/page8_img1.png"), ROOT, {"page8_img1.png"})["page8_img1.png"]
    assert "<!--" not in context and "<img" not in context and "viking://" not in context


def test_image_only_or_short_pages_keep_vision_fallback():
    assert not extract_pdf_image_contexts(page(text="只有标题"), ROOT, {"page8_img1.png"})
    assert not extract_pdf_image_contexts('<img src="page8_img1.png" />', ROOT, {"page8_img1.png"})


def test_long_context_is_bounded_and_does_not_duplicate_tables():
    body = page(text="产品说明。" * 1000).replace("<!-- Page 8 Image", "<!-- Page 8 Table 1 -->\nUNRELATED_TABLE\n<!-- Page 8 Image")
    context = extract_pdf_image_contexts(body, ROOT, {"page8_img1.png"})["page8_img1.png"]
    assert len(context) < 1300
    assert "UNRELATED_TABLE" not in context


@pytest.mark.asyncio
async def test_concurrent_images_read_sibling_once_and_pass_original_acl_context():
    fs = SimpleNamespace(read=AsyncMock(return_value=page().encode()))
    ctx = object()
    cache = PdfImageContextCache()
    paths = [ROOT + "/source.md", ROOT + "/page8_img1.png"]
    values = await asyncio.gather(*(cache.get(ROOT, "page8_img1.png", paths, fs, ctx) for _ in range(16)))
    assert len(set(values)) == 1 and "LATTICE" in values[0]
    fs.read.assert_awaited_once_with(paths[0], offset=0, size=cache.MAX_READ_BYTES, ctx=ctx)


@pytest.mark.asyncio
async def test_caches_are_task_local_and_do_not_cross_accounts():
    fs = SimpleNamespace(read=AsyncMock(side_effect=[page(text="甲租户产品。" * 30).encode(), page(text="乙租户产品。" * 30).encode()]))
    paths = [ROOT + "/source.md", ROOT + "/page8_img1.png"]
    first = await PdfImageContextCache().get(ROOT, "page8_img1.png", paths, fs, object())
    second = await PdfImageContextCache().get(ROOT, "page8_img1.png", paths, fs, object())
    assert "甲租户" in first and "乙租户" not in first
    assert "乙租户" in second and "甲租户" not in second


@pytest.mark.asyncio
async def test_failed_context_reads_consume_bounded_budget_and_return_fallback():
    fs = SimpleNamespace(read=AsyncMock(side_effect=RuntimeError("access denied")))
    paths = [ROOT + f"/{i:02}.md" for i in range(32)] + [ROOT + "/page8_img1.png"]
    cache = PdfImageContextCache()
    assert await cache.get(ROOT, "page8_img1.png", paths, fs, object()) == ""
    assert fs.read.await_count == cache.MAX_TOTAL_READ_BYTES // cache.MAX_READ_BYTES
    assert await cache.get(ROOT, "page8_img1.png", paths, fs, object()) == ""
    assert fs.read.await_count == 8


@pytest.mark.asyncio
async def test_small_markdown_reads_do_not_spend_full_requested_byte_budget():
    fs = SimpleNamespace(read=AsyncMock(return_value=page().encode()))
    paths = [ROOT + f"/{i:02}.md" for i in range(16)] + [ROOT + "/page8_img1.png"]
    cache = PdfImageContextCache()
    assert await cache.get(ROOT, "page8_img1.png", paths, fs, object())
    assert fs.read.await_count == 16
    assert cache._read_bytes == len(page().encode()) * 16


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_page_evidence_is_opt_in_and_original_image_is_still_read(monkeypatch, enabled):
    original = b"original-image-content"
    fs = SimpleNamespace(read_file_bytes=AsyncMock(return_value=original))
    client = SimpleNamespace(get_vision_completion_async=AsyncMock(return_value="visual summary"))
    config = SimpleNamespace(vlm=client, image=ImageConfig(reuse_pdf_page_text_for_summary=enabled))
    monkeypatch.setattr(media_utils, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(media_utils, "get_openviking_config", lambda: config)
    monkeypatch.setattr(media_utils, "render_prompt", lambda *_args: "prompt")
    monkeypatch.setattr(media_utils, "prepare_image_bytes_for_model", lambda b, **_kwargs: b)
    context = "Source PDF page 8 text (not image OCR): LATTICE晶格科技"
    result = await media_utils.generate_image_summary(ROOT + "/page8_img1.png", "page8_img1.png", image_context=context)
    assert result["summary"] == (context if enabled else "visual summary")
    fs.read_file_bytes.assert_awaited_once()
    assert client.get_vision_completion_async.await_count == (0 if enabled else 1)


@pytest.mark.asyncio
async def test_enabled_option_without_verified_page_context_still_calls_vision(monkeypatch):
    fs = SimpleNamespace(read_file_bytes=AsyncMock(return_value=b"image"))
    client = SimpleNamespace(get_vision_completion_async=AsyncMock(return_value="visual summary"))
    monkeypatch.setattr(media_utils, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(media_utils, "get_openviking_config", lambda: SimpleNamespace(vlm=client, image=ImageConfig(reuse_pdf_page_text_for_summary=True)))
    monkeypatch.setattr(media_utils, "render_prompt", lambda *_args: "prompt")
    monkeypatch.setattr(media_utils, "prepare_image_bytes_for_model", lambda b, **_kwargs: b)
    result = await media_utils.generate_image_summary(ROOT + "/standalone.png", "standalone.png")
    assert result["summary"] == "visual summary"
    client.get_vision_completion_async.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,multimodal,expected", [(False, True, False), (True, False, False), (True, True, True)])
async def test_dag_requires_opt_in_and_actual_multimodal_embedder(monkeypatch, enabled, multimodal, expected):
    from tests.storage.test_semantic_dag_stats import _FakeProcessor, _FakeVikingFS
    from openviking.storage.queuefs.semantic_dag import SemanticDagExecutor
    from openviking.server.identity import RequestContext, Role
    from openviking_cli.session.user_id import UserIdentifier

    fs = _FakeVikingFS({ROOT: [{"name": "source.md", "isDir": False}, {"name": "page8_img1.png", "isDir": False}]})
    fs.read = AsyncMock(return_value=page().encode())
    processor = _FakeProcessor()
    original = processor._generate_single_file_summary
    received = {}

    async def summarize(file_path, llm_sem=None, ctx=None, **kwargs):
        received[file_path] = kwargs
        return await original(file_path, llm_sem, ctx)

    processor._generate_single_file_summary = summarize
    config = SimpleNamespace(
        semantic=SimpleNamespace(overview_sample_limit=32),
        image=ImageConfig(reuse_pdf_page_text_for_summary=enabled),
        embedding=SimpleNamespace(get_embedder=lambda: SimpleNamespace(supports_multimodal=multimodal)),
    )
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fs)
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_openviking_config", lambda: config)
    ctx = RequestContext(user=UserIdentifier("account", "user"), role=Role.USER)
    executor = SemanticDagExecutor(processor=processor, context_type="resource", max_concurrent_llm=2, ctx=ctx)
    await executor.run(ROOT)
    await asyncio.sleep(0)
    assert bool(received[ROOT + "/page8_img1.png"].get("image_context")) is expected
    assert received[ROOT + "/source.md"] == {}
    assert sorted(processor.vectorized_files) == sorted(received)
    assert fs.read.await_count == int(expected)


@pytest.mark.asyncio
async def test_source_page_summary_keeps_original_image_in_embedding_message(monkeypatch):
    import base64
    from tests.unit.test_vectorize_file_strategy import DummyFS, DummyQueue, DummyQueueManager, DummyReq
    from openviking.utils import embedding_utils

    original = b"\x89PNG\r\n\x1a\noriginal-crop"
    queue = DummyQueue()
    fs = DummyFS(original)
    monkeypatch.setattr(embedding_utils, "get_queue_manager", lambda: DummyQueueManager(queue))
    monkeypatch.setattr(embedding_utils, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(embedding_utils, "get_openviking_config", lambda: SimpleNamespace(
        image=ImageConfig(reuse_pdf_page_text_for_summary=True),
        embedding=SimpleNamespace(text_source="summary_first", max_input_tokens=1000),
    ))
    summary = "Source PDF page 8 text (not image OCR): LATTICE晶格科技"
    await embedding_utils.vectorize_file(ROOT + "/page8_img1.png", {"name": "page8_img1.png", "summary": summary}, ROOT, ctx=DummyReq())
    message = queue.items[0].message
    assert message[0] == {"type": "text", "text": summary}
    image_uri = message[1]["image_url"]["url"]
    assert base64.b64decode(image_uri.split(",", 1)[1]) == original
