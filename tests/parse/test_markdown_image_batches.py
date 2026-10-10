"""Bounded local image I/O preserves bytes, names, and persistence failures."""

import asyncio
import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

from openviking.parse.image_rewrite import IMAGE_MAPPINGS_FILENAME
from openviking.parse.parsers.base_parser import BaseParser
from openviking.parse.parsers.markdown import MarkdownParser
from openviking_cli.utils.config.parser_config import ParserConfig, PDFConfig

ROOT = "viking://temp/image-batches/document"


def image(path, color="red", size=(32, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


class Store:
    def __init__(self, content, fail=None, block=False):
        self.files = {ROOT + "/section.md": content}
        self.active = self.maximum = 0
        self.fail = fail
        self.block = block
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def glob(self, *args, **kwargs):
        return {"matches": [ROOT + "/section.md"]}

    async def read_file(self, uri):
        return self.files[uri]

    async def write_file_bytes(self, uri, data):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        if self.active >= 2:
            self.started.set()
        try:
            if self.block:
                await self.release.wait()
            else:
                await asyncio.sleep(0.01)
            if uri.endswith(self.fail or "/never-fail"):
                raise OSError("storage failed")
            self.files[uri] = data
        finally:
            self.active -= 1

    async def write_file(self, uri, text):
        assert self.active == 0
        self.files[uri] = text


async def ingest(store, folder, concurrency=1):
    parser = MarkdownParser(config=PDFConfig(local_image_write_concurrency=concurrency))
    with patch.object(BaseParser, "_get_viking_fs", return_value=store):
        await parser._ingest_local_images(ROOT, base_dir=folder)


async def test_two_writes_overlap_with_identical_serial_bytes_and_names(tmp_path):
    a = image(tmp_path / "a/item.png", "red")
    b = image(tmp_path / "b/item.png", "blue")
    c = image(tmp_path / "third.png", "green")
    content = "![a](a/item.png) ![a again](a/item.png) <img src='b/item.png'> ![c](third.png)"
    serial, parallel = Store(content), Store(content)
    await ingest(serial, tmp_path)
    await ingest(parallel, tmp_path, 2)
    assert serial.maximum == 1 and parallel.maximum == 2
    assert serial.files == parallel.files
    mappings = json.loads(parallel.files[ROOT + "/" + IMAGE_MAPPINGS_FILENAME])["section.md"]
    assert parallel.files[ROOT + "/" + mappings["a/item.png"]] == a
    assert parallel.files[ROOT + "/" + mappings["b/item.png"]] == b
    assert parallel.files[ROOT + "/" + mappings["third.png"]] == c
    assert mappings["a/item.png"] != mappings["b/item.png"]


async def test_rejected_and_outside_images_are_not_written_or_named(tmp_path):
    image(tmp_path / "small/item.png", size=(10, 32))
    blob = image(tmp_path / "good/item.png")
    image(tmp_path.parent / "outside.png")
    store = Store("![tiny](small/item.png) ![good](good/item.png) ![outside](../outside.png)")
    await ingest(store, tmp_path, 2)
    mappings = json.loads(store.files[ROOT + "/" + IMAGE_MAPPINGS_FILENAME])["section.md"]
    assert mappings == {"good/item.png": "item.png"}
    assert store.files[ROOT + "/item.png"] == blob


async def test_storage_failure_is_terminal_after_other_batch_writes_settle(tmp_path):
    image(tmp_path / "a.png")
    image(tmp_path / "b.png")
    store = Store("![a](a.png) ![b](b.png)", fail="a.png")
    with pytest.raises(RuntimeError, match="persist an embedded image"):
        await ingest(store, tmp_path, 2)
    assert store.active == 0 and ROOT + "/b.png" in store.files
    assert ROOT + "/" + IMAGE_MAPPINGS_FILENAME not in store.files


async def test_cancellation_does_not_leave_active_batch_writes(tmp_path):
    image(tmp_path / "a.png")
    image(tmp_path / "b.png")
    store = Store("![a](a.png) ![b](b.png)", block=True)
    task = asyncio.create_task(ingest(store, tmp_path, 2))
    await asyncio.wait_for(store.started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.active == 0
    assert ROOT + "/" + IMAGE_MAPPINGS_FILENAME not in store.files


async def test_failed_section_read_is_not_silently_successful(tmp_path):
    store = Store("![a](a.png)")
    store.files.clear()
    with pytest.raises(RuntimeError, match="read a section"):
        await ingest(store, tmp_path, 2)


async def test_oversized_file_read_is_bounded_before_validation(tmp_path):
    limit = MarkdownParser.IMAGE_MAX_FILE_BYTES
    (tmp_path / "large.png").write_bytes(b"x" * (limit + 4096))
    store = Store("![a](large.png)")
    seen = []
    with patch.object(MarkdownParser, "_is_valid_image", side_effect=lambda b, p: seen.append(len(b)) or False):
        await ingest(store, tmp_path, 2)
    assert seen == [limit + 1]
    assert store.maximum == 0


@pytest.mark.parametrize("value", [0, 5, -1, True, 1.5, "2"])
def test_invalid_concurrency_is_rejected(value):
    with pytest.raises(ValueError, match="local_image_write_concurrency"):
        PDFConfig(local_image_write_concurrency=value).validate()


def test_serial_default_and_pdf_inheritance():
    assert ParserConfig().local_image_write_concurrency == PDFConfig().local_image_write_concurrency == 1
    PDFConfig.from_dict({"local_image_write_concurrency": 2}).validate()
