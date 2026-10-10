"""Exercise the PDF worker lifecycle without loading server/native bindings."""

import ast
import asyncio
import __future__
import ctypes
import gc
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def load_parser():
    source = Path(__file__).parents[2] / "openviking/parse/parsers/pdf.py"
    tree = ast.parse(source.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PDFParser")
    namespace = {"BaseParser": object, "Path": Path, "asyncio": asyncio, "logger": Mock()}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace["PDFParser"]


PDFParser = load_parser()


class PDFMemoryReclaimTests(unittest.TestCase):
    def parser(self):
        return object.__new__(PDFParser)

    def test_success_preserves_result_and_reclaims_in_worker(self):
        parser = self.parser()
        expected = ("unchanged markdown", {"pages_processed": 88})
        main = threading.get_ident()
        parser._convert_local_sync = Mock(return_value=expected)
        worker_threads = []
        parser._reclaim_local_parse_memory = lambda _: worker_threads.append(threading.get_ident())
        self.assertIs(asyncio.run(parser._convert_local(Path("original.pdf"))), expected)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], main)

    def test_parse_failure_preserves_original_exception_and_reclaims(self):
        parser = self.parser()
        error = ValueError("original parse error")
        parser._convert_local_sync = Mock(side_effect=error)
        parser._reclaim_local_parse_memory = Mock()
        with self.assertRaises(ValueError) as raised:
            asyncio.run(parser._convert_local(Path("original.pdf")))
        self.assertIs(raised.exception, error)
        parser._reclaim_local_parse_memory.assert_called_once()

    def test_cancelled_awaiter_still_reclaims_when_worker_finishes(self):
        parser = self.parser()
        entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()

        def convert(*_):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test worker was not released")
            return "markdown", {}

        parser._convert_local_sync = convert
        parser._reclaim_local_parse_memory = lambda _: cleaned.set()

        async def cancel():
            task = asyncio.create_task(parser._convert_local(Path("original.pdf")))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()

        asyncio.run(cancel())
        self.assertTrue(cleaned.is_set())

    def test_small_pdf_skips_process_collection_and_allocator(self):
        with patch.object(sys, "platform", "linux"), patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1024)), \
                patch.object(gc, "collect") as collect, patch.object(ctypes, "CDLL") as library:
            PDFParser._reclaim_local_parse_memory(Path("small.pdf"))
        collect.assert_not_called()
        library.assert_not_called()

    def test_non_linux_skips_process_collection_and_allocator(self):
        with patch.object(sys, "platform", "darwin"), patch.object(gc, "collect") as collect, \
                patch.object(ctypes, "CDLL") as library:
            PDFParser._reclaim_local_parse_memory(Path("original.pdf"))
        collect.assert_not_called()
        library.assert_not_called()

    def test_large_pdf_collects_before_native_trim(self):
        order = []
        trim = Mock(side_effect=lambda _: order.append("trim"))
        with patch.object(sys, "platform", "linux"), patch.object(Path, "stat", return_value=SimpleNamespace(st_size=16 * 1024 * 1024)), \
                patch.object(gc, "collect", side_effect=lambda: order.append("gc")), \
                patch.object(ctypes, "CDLL", return_value=SimpleNamespace(malloc_trim=trim)):
            PDFParser._reclaim_local_parse_memory(Path("original.pdf"))
        self.assertEqual(order, ["gc", "trim"])
        trim.assert_called_once_with(0)

    def test_missing_glibc_symbol_remains_supported(self):
        with patch.object(sys, "platform", "linux"), patch.object(Path, "stat", return_value=SimpleNamespace(st_size=20 * 1024 * 1024)), \
                patch.object(gc, "collect") as collect, patch.object(ctypes, "CDLL", return_value=SimpleNamespace()):
            PDFParser._reclaim_local_parse_memory(Path("original.pdf"))
        collect.assert_called_once()

    def test_allocator_failure_cannot_fail_the_parse(self):
        parser = self.parser()
        expected = ("unchanged markdown", {})
        parser._convert_local_sync = Mock(return_value=expected)
        with patch.object(sys, "platform", "linux"), patch.object(Path, "stat", return_value=SimpleNamespace(st_size=20 * 1024 * 1024)), \
                patch.object(gc, "collect"), patch.object(ctypes, "CDLL", side_effect=OSError("unavailable")):
            self.assertIs(parser._convert_local_with_reclaim(Path("original.pdf")), expected)


if __name__ == "__main__":
    unittest.main()
