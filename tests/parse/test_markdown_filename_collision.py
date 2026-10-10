# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for _generate_merged_filename uniqueness when headings collide."""

import asyncio

from openviking.parse.parsers.markdown import MarkdownParser, _LayoutOp
from openviking_cli.utils.config.parser_config import ParserConfig


class TestGenerateMergedFilenameCollision:
    def _make_parser(self) -> MarkdownParser:
        return MarkdownParser(ParserConfig())

    def test_duplicate_heading_produces_unique_filenames(self):
        """Merge groups with same first heading but different content must get unique filenames."""
        parser = self._make_parser()

        group1 = [
            ("Our Culture", "content about values", 1),
            ("Our Culture", "content about mission", 2),
        ]
        group2 = [
            ("Our Culture", "content about team", 3),
            ("Our Culture", "content about vision", 4),
        ]

        name1 = parser._generate_merged_filename(group1)
        name2 = parser._generate_merged_filename(group2)

        assert name1 != name2, f"Filenames must be unique but both are '{name1}'"
        assert "Our" in name1 or "Culture" in name1  # Still human-readable

    def test_single_section_filename_still_works(self):
        """Single section should still produce a readable filename."""
        parser = self._make_parser()

        sections = [("Introduction", "some content", 1)]
        name = parser._generate_merged_filename(sections)

        assert "Introduction" in name

    def test_empty_sections_returns_merged(self):
        """Empty sections list should return 'merged'."""
        parser = self._make_parser()
        assert parser._generate_merged_filename([]) == "merged"


class TestMarkdownSourceNameLayout:
    def _make_parser(self) -> MarkdownParser:
        return MarkdownParser(ParserConfig())

    async def test_non_code_source_name_uses_stemmed_root(self):
        parser = self._make_parser()

        layout = await parser._compute_layout(
            "hello world\n",
            "viking://temp/test",
            source_name="aa.txt",
        )

        assert layout.root_dir == "viking://temp/test/aa"
        assert any(op.uri == "viking://temp/test/aa/aa.md" for op in layout.ops)

    async def test_code_source_name_preserves_extension_in_root(self):
        parser = self._make_parser()

        layout = await parser._compute_layout(
            "def foo():\n    return 1\n",
            "viking://temp/test",
            source_name="foo.py",
        )

        assert layout.root_dir == "viking://temp/test/foo.py"
        assert any(op.uri == "viking://temp/test/foo.py/foo.md" for op in layout.ops)

    async def test_unsupported_code_source_name_uses_stemmed_root(self):
        parser = self._make_parser()

        layout = await parser._compute_layout(
            "echo hello\n",
            "viking://temp/test",
            source_name="script.css",
        )

        assert layout.root_dir == "viking://temp/test/script"
        assert any(op.uri == "viking://temp/test/script/script.md" for op in layout.ops)


class TestRepeatedSectionContent:
    async def test_leading_deeper_heading_is_preserved_before_shallower_root(self):
        parser = MarkdownParser(config=ParserConfig())
        content = "\n".join(
            f"{'#' * level} Section_{page}\n\n<!-- Page {page} -->\n\n"
            + "产品知识技术说明。" * 150
            for page, level in enumerate((3, 2, 4, 3, 2), 1)
        )
        layout = await parser._compute_layout(
            content, "viking://temp/test", source_name="sample.pdf"
        )
        stored = {op.uri: op.content for op in layout.ops if op.kind == "write"}
        for page in range(1, 6):
            assert any(f"<!-- Page {page} -->" in text for text in stored.values())

    async def test_repeated_pdf_headings_preserve_every_page(self):
        parser = MarkdownParser(config=ParserConfig())
        content = "\n".join(
            f"# FROM\n\n<!-- Page {page} -->\n\nUNIQUE_{page}\n\n"
            + "产品知识技术说明。" * 150
            for page in range(1, 4)
        )
        layout = await parser._compute_layout(
            content, "viking://temp/test", source_name="sample.pdf"
        )
        writes = [op for op in layout.ops if op.kind == "write"]
        stored = {op.uri: op.content for op in writes}
        assert len(stored) == len(writes) == 3
        for page in range(1, 4):
            assert any(f"UNIQUE_{page}" in text for text in stored.values())
        assert all(text.startswith("# FROM") for text in stored.values())

    async def test_repeated_large_sections_have_separate_directories(self):
        parser = MarkdownParser(config=ParserConfig(max_section_chars=1000))
        ops = []
        for marker in ("FIRST", "SECOND"):
            await parser._save_section(
                ops,
                "",
                [],
                "viking://temp/test",
                {
                    "name": "Speaker_Notes",
                    "tokens": 3000,
                    "content": f"# Speaker Notes\n\n{marker}\n\n" + "notes\n\n" * 500,
                    "has_children": False,
                },
                2048,
                512,
            )
        directories = [op.uri for op in ops if op.kind == "mkdir"]
        stored = {op.uri: op.content for op in ops if op.kind == "write"}
        assert len(set(directories)) == len(directories) == 2
        assert any("FIRST" in text for text in stored.values())
        assert any("SECOND" in text for text in stored.values())

    async def test_equal_token_merge_groups_keep_both_contents(self):
        parser = MarkdownParser(config=ParserConfig())
        ops = []
        await parser._save_merged(ops, "viking://temp/test", [("FROM", "first", 10)])
        await parser._save_merged(ops, "viking://temp/test", [("FROM", "other", 10)])
        stored = {op.uri: op.content for op in ops if op.kind == "write"}
        assert sorted(stored.values()) == ["first", "other"]

    async def test_split_merge_avoids_existing_part_filename(self):
        parser = MarkdownParser(config=ParserConfig(max_section_chars=30))
        sections = [("FROM", "first paragraph\n\nsecond paragraph\n\nthird paragraph", 20)]
        name = parser._generate_merged_filename(sections)
        assert len(parser._smart_split_content(sections[0][1], 2048)) >= 2
        ops = [_LayoutOp("write", f"viking://temp/test/{name}_2.md", "original")]
        await parser._save_merged(ops, "viking://temp/test", sections)
        stored = {op.uri: op.content for op in ops if op.kind == "write"}
        assert len(stored) == len(ops)
        assert stored[f"viking://temp/test/{name}_2.md"] == "original"
        assert "first paragraph" in "\n".join(stored.values())
        assert "third paragraph" in "\n".join(stored.values())

    async def test_generated_suffix_does_not_overwrite_source_heading(self):
        parser = MarkdownParser(config=ParserConfig())
        ops = []
        for name, text in (("FROM", "first"), ("FROM_1", "source suffix"), ("FROM", "last")):
            await parser._save_section(
                ops,
                "",
                [],
                "viking://temp/test",
                {"name": name, "tokens": 10, "content": text, "has_children": False},
                2048,
                512,
            )
        stored = {op.uri: op.content for op in ops if op.kind == "write"}
        assert sorted(stored.values()) == ["first", "last", "source suffix"]

    async def test_directory_and_file_share_no_section_stem(self):
        parser = MarkdownParser(config=ParserConfig())
        ops = [_LayoutOp("mkdir", "viking://temp/test/FROM", exist_ok=True)]
        await parser._save_section(
            ops,
            "",
            [],
            "viking://temp/test",
            {"name": "FROM", "tokens": 10, "content": "kept", "has_children": False},
            2048,
            512,
        )
        assert ops[-1].uri == "viking://temp/test/FROM_1.md"

    async def test_concurrent_layouts_do_not_share_reserved_names(self):
        parser = MarkdownParser(config=ParserConfig())
        content = "\n".join("# FROM\n\n" + "产品知识技术说明。" * 150 for _ in range(3))
        layouts = await asyncio.gather(
            *[
                parser._compute_layout(content, "viking://temp/test", source_name="sample.pdf")
                for _ in range(2)
            ]
        )
        paths = [[op.uri for op in layout.ops if op.kind == "write"] for layout in layouts]
        assert paths[0] == paths[1]
        assert len(set(paths[0])) == 3
