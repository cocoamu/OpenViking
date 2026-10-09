"""Reuse page rasters without changing extracted image pixels or resolution."""
import io

import pdfplumber
import pdfplumber.display
import pytest
from PIL import Image

from openviking.parse.parsers.pdf import PDFParser
from openviking_cli.utils.config.parser_config import PDFConfig


def test_multiple_image_crops_reuse_one_render_and_match_original(tmp_path, monkeypatch):
    source = tmp_path / "colored.pdf"
    image = Image.new("RGB", (64, 64), "blue")
    image.paste("red", (0, 0, 32, 64))
    image.save(source, "PDF", resolution=72)
    bounds = [(0, 0, 32, 64), (32, 0, 64, 64)]
    parser = PDFParser(PDFConfig(image_resolution=144))
    calls = []
    render = pdfplumber.display.get_page_image

    def counted(*args, **kwargs):
        calls.append(1)
        return render(*args, **kwargs)

    with pdfplumber.open(source) as pdf:
        page = pdf.pages[0]
        expected = []
        for box in bounds:
            stream = io.BytesIO()
            page.crop(box).to_image(resolution=144).save(stream, format="PNG")
            expected.append(stream.getvalue())
        monkeypatch.setattr(pdfplumber.display, "get_page_image", counted)
        actual = [
            parser._extract_image_from_page(
                page, dict(zip(("x0", "top", "x1", "bottom"), box))
            )
            for box in bounds
        ]
        assert actual == expected
        assert len(calls) == 1
        rendered = page._openviking_rendered_page
        parser._release_page_cache(page)
        assert not hasattr(page, "_openviking_rendered_page")
        with pytest.raises(ValueError, match="closed image"):
            rendered.original.getbbox()


def test_cropbox_pages_keep_original_rendering(tmp_path, monkeypatch):
    source = tmp_path / "cropped.pdf"
    Image.new("RGB", (64, 64), "blue").save(source, "PDF", resolution=72)
    parser = PDFParser(PDFConfig(image_resolution=72))
    with pdfplumber.open(source) as pdf:
        page = pdf.pages[0].crop((0, 0, 32, 32))
        box = {"x0": 0, "top": 0, "x1": 16, "bottom": 16}
        stream = io.BytesIO()
        page.crop((0, 0, 16, 16)).to_image(resolution=72).save(stream, format="PNG")
        assert parser._extract_image_from_page(page, box) == stream.getvalue()
        assert not hasattr(page, "_openviking_rendered_page")
