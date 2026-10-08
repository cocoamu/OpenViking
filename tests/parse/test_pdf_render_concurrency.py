"""PDF rendering must serialize PDFium across independent parser instances."""

import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from openviking.parse.parsers.pdf import PDFParser


def test_pdf_rendering_is_serialized_across_parser_instances():
    first_entered, second_started, second_entered, release = (Event() for _ in range(4))

    class Image:
        def save(self, buffer, format):
            buffer.write(b"png")

    class Page:
        width = height = 100

        def __init__(self, entered):
            self.entered = entered

        def crop(self, bbox):
            return self

        def to_image(self, resolution):
            self.entered.set()
            assert release.wait(5)
            return Image()

    bounds = dict(x0=0, top=0, x1=100, bottom=100)

    def second():
        second_started.set()
        return PDFParser()._extract_image_from_page(Page(second_entered), bounds)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(PDFParser()._extract_image_from_page, Page(first_entered), bounds)
        try:
            assert first_entered.wait(5)
            other = pool.submit(second)
            assert second_started.wait(5)
            overlap = second_entered.wait(0.2)
        finally:
            release.set()
        assert first.result(timeout=5) == other.result(timeout=5) == b"png"
    assert not overlap, "independent PDF parsers entered PDFium concurrently"
    assert second_entered.is_set()


def test_parallel_real_pdf_images_remain_valid(tmp_path):
    pdfplumber = pytest.importorskip("pdfplumber")
    PIL = pytest.importorskip("PIL.Image")
    paths = []
    for name, color in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
        path = tmp_path / f"{name}.pdf"
        PIL.new("RGB", (48, 48), color).save(path, "PDF")
        paths.append((path, color))

    def render(item):
        path, color = item
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[0]
            data = PDFParser()._extract_image_from_page(page, page.images[0])
        assert data is not None
        with PIL.open(io.BytesIO(data)) as image:
            image.load()
            pixel = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
            assert max(abs(a - b) for a, b in zip(pixel, color)) < 10

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(render, paths * 12))
