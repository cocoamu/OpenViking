"""Repair only PDF catalog failures without changing the uploaded source."""

from contextlib import ExitStack, contextmanager
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def open_pdf_compat(pdf_path, pdfplumber, metadata, render_lock):
    with ExitStack() as stack:
        try:
            pdf = stack.enter_context(pdfplumber.open(str(pdf_path)))
        except Exception as error:
            if "No /Root object!" not in str(error):
                raise
            display = importlib.import_module("pdfplumber.display")
            mutex = getattr(display, "_PRE_PDFIUM_RENDER_MUTEX", render_lock)
            pdfium = importlib.import_module("pypdfium2")
            directory = stack.enter_context(TemporaryDirectory(prefix="ov-pdf-compat-"))
            normalized = Path(directory) / "normalized.pdf"
            with mutex:
                document = pdfium.PdfDocument(str(pdf_path))
                try:
                    page_count = len(document)
                    document.save(str(normalized))
                finally:
                    document.close()
            pdf = stack.enter_context(pdfplumber.open(str(normalized)))
            if len(pdf.pages) != page_count:
                raise ValueError("PDF compatibility normalization changed page count")
            metadata["compat_normalization"] = "pdfium-xref"
        yield pdf
