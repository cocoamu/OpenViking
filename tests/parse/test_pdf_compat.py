from contextlib import nullcontext
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from openviking.parse.parsers import pdf_compat


def test_normal_pdf_uses_original_without_normalization(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"original")
    pdf = SimpleNamespace(pages=[object()])
    opened = []
    def open_pdf(path):
        opened.append(path)
        return nullcontext(pdf)
    metadata = {}
    with pdf_compat.open_pdf_compat(source, SimpleNamespace(open=open_pdf), metadata, threading.Lock()) as result:
        assert result is pdf
    assert opened == [str(source)]
    assert metadata == {}
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("normalized_pages", [2, 1])
def test_catalog_repair_is_locked_and_preserves_source(tmp_path, monkeypatch, normalized_pages):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"original")
    mutex = threading.Lock()
    closed = []
    opened = []
    class Document:
        def __init__(self, path):
            assert mutex.locked()
            assert path == str(source)
        def __len__(self):
            return 2
        def save(self, path):
            assert mutex.locked()
            Path(path).write_bytes(b"normalized")
        def close(self):
            assert mutex.locked()
            closed.append(True)
    def open_pdf(path):
        opened.append(path)
        if path == str(source):
            raise ValueError("No /Root object! - Is this really a PDF?")
        assert Path(path).read_bytes() == b"normalized"
        return nullcontext(SimpleNamespace(pages=[object()] * normalized_pages))
    monkeypatch.setattr(pdf_compat.importlib, "import_module", lambda name: SimpleNamespace(PdfDocument=Document) if name == "pypdfium2" else SimpleNamespace())
    metadata = {}
    if normalized_pages == 2:
        with pdf_compat.open_pdf_compat(source, SimpleNamespace(open=open_pdf), metadata, mutex) as result:
            assert len(result.pages) == 2
        assert metadata == {"compat_normalization": "pdfium-xref"}
    else:
        with pytest.raises(ValueError, match="changed page count"):
            with pdf_compat.open_pdf_compat(source, SimpleNamespace(open=open_pdf), metadata, mutex):
                pytest.fail("A changed page count must not be accepted")
        assert metadata == {}
    assert closed == [True]
    assert not Path(opened[-1]).exists()
    assert source.read_bytes() == b"original"


def test_unrelated_error_is_not_repaired(tmp_path):
    error = PermissionError("not authorized")
    def open_pdf(path):
        raise error
    with pytest.raises(PermissionError) as caught:
        with pdf_compat.open_pdf_compat(tmp_path / "source.pdf", SimpleNamespace(open=open_pdf), {}, threading.Lock()):
            pytest.fail("An unrelated error must propagate")
    assert caught.value is error
