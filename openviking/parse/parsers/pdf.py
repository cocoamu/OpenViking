# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
PDF parser for OpenViking.

Unified parser that converts PDF to Markdown then parses the result.
Supports dual strategy:
- Local: pdfplumber for direct conversion
- Remote: MinerU API for advanced conversion

This design simplifies PDF handling by delegating structure analysis
to the MarkdownParser after conversion.
"""

import asyncio
import base64
import hashlib
import io
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openviking.parse.base import (
    NodeType,
    ParseResult,
    ResourceNode,
    create_parse_result,
    lazy_import,
)
from openviking.parse.parsers.base_parser import BaseParser
from openviking.utils.time_utils import parse_iso_datetime
from openviking_cli.utils import get_logger
from openviking_cli.utils.config.parser_config import PDFConfig

logger = get_logger(__name__)

# PDFium is process-global and cannot be called concurrently, even for different
# documents. Parsers run in worker threads, so all instances share this mutex.
_PDF_RENDER_LOCK = threading.Lock()


# EXEPD-230436：扫描件常见「整页底图 + 若干嵌套文字块图」的多层结构。
# 页面没有文本层时，整页底图已经覆盖该页全部可见像素；再按 XObject 逐块抽图，
# 等于把同一页切成一批互相重叠的碎片，而块边界不受排版约束——可能正好落在折行中间。
# 现场（2026-11-11 dev《终端员工奖惩管理办法--签批版.pdf》）：
#   2.2.2「一次嘉奖…一次小功…一次大功奖励人民币 / 300 元。」被块边界切成两段，
#   命中的 page1_img4 只到「大功奖励人民币」，模型如实回答「文字截断、读不到金额」。
# 判定必须**三条件同时成立**，避免误删正常 PDF 的插图：
#   ① 该页没有可提取文本（纯扫描/整页栅格化）；
#   ② 某张图覆盖 ≥95% 页面（整页底图）；
#   ③ 另一张图完全落在它的 bbox 内（只是它上面的重复图层）。
# 只有同时满足时，被包含的图层才是冗余的；整页底图抽取失败时一律不丢（见调用处）。
PAGE_COVERING_AREA_RATIO = 0.95
NESTED_CONTAINMENT_TOLERANCE_PT = 1.0


# ── EXEPD-230261 图片 alt 契约 ──────────────────────────────────────────────
# 凡是进入入库 markdown 的图片引用，alt 必须是可读语义：
#   非空、非 "Page N Image M"、非 "[Image]"、非纯文档名/文件名/哈希。
# 唯一的例外是「同页无任何文本线索且 VLM 也失败」的图：保留占位，但必须计入
# ImageCaptionReport.uncaptioned 并以聚合 WARNING 暴露，不能静默漏掉。
_IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
_IMAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s+Image\s+(\d+)\s*-->")
_HEADING_LINE_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
# 契约第 2 条：只认 "Page N Image M" 这一种机读占位；裸 "Page N" 同样是占位。
_PLACEHOLDER_CAPTION_RE = re.compile(r"^page\s*\d+(?:\s*image\s*\d+)?$", re.IGNORECASE)
# 契约第 3 条：占位族（大小写不敏感）。
_PLACEHOLDER_CAPTION_TOKENS = frozenset(
    {"[image]", "[图片]", "图片", "image", "undefined", "null", "n/a"}
)
# 契约第 4 条：图片文件名形态（pageN_imgM[.ext]）。
_IMAGE_FILENAME_RE = re.compile(r"^page\d+[_-]?img\d+(?:\.[a-z0-9]{1,5})?$", re.IGNORECASE)
_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".svg",
)
# 契约第 1 条：超过 120 字符先截断再判定（与交付侧一致）。
_CAPTION_MAX_CHARS = 120
# 兜底形状 "<线索>（图 N）" 至少需要 8 个字符，配置值在运行时收敛到这个区间。
_CAPTION_MIN_CHARS = 8


def _normalize_alt(value: Any) -> str:
    # 契约只要求 strip；内部空白由正则的 \s* 处理，避免与交付侧长度口径漂移。
    return str(value or "").strip()


def _normalize_name(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return re.sub(r"[\s_\-—–·,，.。()（）\[\]【】<>《》'\"`]+", "", Path(raw).stem).lower()


def is_semantic_caption(value: Any, document_name: str = "", image_name: str = "") -> bool:
    """EXEPD-230261 语义 alt 契约（引擎侧唯一实现，交付侧按同一套规则核对）。

    判定为真当且仅当同时满足：
    1. 字符串 strip 后长度 2..120（超长先截断再判定）；
    2. 不匹配 `^page\\s*\\d+\\s*image\\s*\\d+$`（大小写不敏感；裸 Page N 也视作占位）；
    3. 不属于占位族 `[Image]` / `[图片]` / `图片` / `image` / `undefined` / `null` / `N/A`；
    4. 不等于文档名（含去扩展名形式）、图片文件名（pageN_imgM[.ext]）或哈希；
    5. 不是纯数字/纯符号（`40`、`→`、单字符 `W` 都不算语义）。
    """
    if not isinstance(value, str):
        return False
    text = _normalize_alt(value)
    if len(text) > _CAPTION_MAX_CHARS:
        text = text[:_CAPTION_MAX_CHARS].strip()
    if not 2 <= len(text) <= _CAPTION_MAX_CHARS:
        return False

    lowered = text.lower()
    if _PLACEHOLDER_CAPTION_RE.match(text) or lowered in _PLACEHOLDER_CAPTION_TOKENS:
        return False
    if document_name and _normalize_name(text) == _normalize_name(document_name):
        return False
    if image_name and _normalize_name(text) == _normalize_name(image_name):
        return False
    if _IMAGE_FILENAME_RE.match(text) or lowered.endswith(_IMAGE_SUFFIXES):
        return False
    if re.fullmatch(r"[0-9a-f]{16,}", lowered):
        return False

    # 纯数字（含 40、40.5、40%）或纯符号不算语义；"30℃" 这类数字+符号仍可接受。
    if re.fullmatch(r"[\d\s.,%‰+\-]+", text):
        return False
    if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text):
        return False
    return True


# 兼容旧调用名；实现保持唯一，避免两套判定漂移。
is_semantic_image_alt = is_semantic_caption


def _clean_caption(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace("[", " ").replace("]", " ")
    return " ".join(text.split())[:limit].strip()


@dataclass
class ImageAltAudit:
    """对一段 markdown 的图片 alt 做契约审计。"""

    total: int = 0
    semantic: int = 0
    invalid: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.invalid

    @property
    def uncaptioned(self) -> int:
        return len(self.invalid)


def audit_image_alts(markdown: str, document_name: str = "") -> ImageAltAudit:
    audit = ImageAltAudit()
    for match in _IMAGE_REF_RE.finditer(markdown or ""):
        audit.total += 1
        image_name = match.group(2).split(None, 1)[0].strip()
        if is_semantic_caption(match.group(1), document_name, image_name):
            audit.semantic += 1
        else:
            audit.invalid.append(match.group(0))
    return audit


class ImageCaptionInvariantError(AssertionError):
    """图片 alt 契约或计数不变量被破坏。"""


def assert_semantic_image_alts(markdown: str, document_name: str = "") -> None:
    """测试/验收用断言：所有图片 alt 都必须是可读语义。"""
    audit = audit_image_alts(markdown, document_name)
    if audit.invalid:
        preview = ", ".join(audit.invalid[:3])
        raise ImageCaptionInvariantError(
            f"{len(audit.invalid)}/{audit.total} image alt(s) are not semantic: {preview}"
        )


@dataclass
class ImageCaptionReport:
    """一次图注补齐的计数与不变量。"""

    images_total: int = 0
    captioned: int = 0
    fallback: int = 0
    reused: int = 0
    uncaptioned: int = 0
    vlm_calls: int = 0
    vlm_retries: int = 0
    errors: int = 0
    uncaptioned_samples: List[str] = field(default_factory=list)

    def problems(self) -> List[str]:
        accounted = self.captioned + self.fallback + self.reused + self.uncaptioned
        problems = []
        if accounted != self.images_total:
            problems.append(
                f"images_total({self.images_total}) != captioned({self.captioned}) + "
                f"fallback({self.fallback}) + reused({self.reused}) + "
                f"uncaptioned({self.uncaptioned})"
            )
        if self.vlm_retries > self.vlm_calls:
            problems.append("vlm_retries cannot exceed vlm_calls")
        return problems

    def assert_valid(self) -> None:
        problems = self.problems()
        if problems:
            raise ImageCaptionInvariantError("; ".join(problems))

    def to_meta(self) -> Dict[str, Any]:
        return {
            "images_total": self.images_total,
            "captioned": self.captioned,
            "fallback": self.fallback,
            "reused": self.reused,
            "uncaptioned": self.uncaptioned,
            "vlm_calls": self.vlm_calls,
            "vlm_retries": self.vlm_retries,
            "errors": self.errors,
            "uncaptioned_samples": list(self.uncaptioned_samples),
        }


class PDFParser(BaseParser):
    """
    PDF parser with dual conversion strategy.

    Converts PDF → Markdown → ParseResult using MarkdownParser.
    When available, extracts PDF bookmarks/outlines and injects them as
    markdown headings so MarkdownParser can build a hierarchical directory
    structure instead of flat numbered files.

    Strategies:
    - "local": Use pdfplumber for text and table extraction
    - "mineru": Use MinerU API for advanced PDF processing
    - "auto": Try local first, fallback to MinerU if configured

    Examples:
        >>> # Local parsing
        >>> parser = PDFParser(PDFConfig(strategy="local"))
        >>> result = await parser.parse("document.pdf")

        >>> # Remote API parsing
        >>> config = PDFConfig(
        ...     strategy="mineru",
        ...     mineru_endpoint="http://127.0.0.1:8000"
        ... )
        >>> parser = PDFParser(config)
        >>> result = await parser.parse("document.pdf")
    """

    def __init__(self, config: Optional[PDFConfig] = None):
        """
        Initialize PDF parser.

        Args:
            config: PDFConfig instance (defaults to auto strategy)
        """
        self.config = config or PDFConfig()
        self.config.validate()

        # Lazy import MarkdownParser to avoid circular imports
        self._markdown_parser = None

    def _get_markdown_parser(self):
        """Lazy import and create MarkdownParser."""
        if self._markdown_parser is None:
            from openviking.parse.parsers.markdown import MarkdownParser

            self._markdown_parser = MarkdownParser(config=self.config)
        return self._markdown_parser

    @property
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions."""
        return [".pdf"]

    async def parse(self, source: Union[str, Path], instruction: str = "", **kwargs) -> ParseResult:
        """
        Parse PDF file.

        Args:
            source: Path to PDF file
            **kwargs: Additional options (resource_name/source_name for original filename)

        Returns:
            ParseResult with document tree

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If conversion fails with all strategies
        """
        start_time = time.time()
        pdf_path = Path(source)

        # Get resource name from kwargs, prefer original filename from upload
        resource_name = kwargs.get("resource_name") or kwargs.get("source_name")

        if not pdf_path.exists():
            return create_parse_result(
                root=ResourceNode(type=NodeType.ROOT),
                source_path=str(pdf_path),
                source_format="pdf",
                parser_name="PDFParser",
                parse_time=time.time() - start_time,
                warnings=[f"File not found: {pdf_path}"],
            )

        try:
            # Step 1: Convert PDF to Markdown
            markdown_content, conversion_meta = await self._convert_to_markdown(
                pdf_path,
                resource_name=resource_name,
            )

            # Step 2: Parse Markdown using MarkdownParser, pass through resource name
            md_parser = self._get_markdown_parser()
            from openviking_cli.utils.storage import get_storage

            storage = get_storage()
            result = await md_parser.parse_content(
                markdown_content,
                source_path=str(pdf_path),
                resource_name=resource_name,
                source_name=resource_name,
                base_dir=pdf_path.parent,
                allowed_media_dirs=[storage.media_dir],
                split_content=kwargs.get("split_content", True),
            )

            # Step 3: Update metadata for PDF origin
            result.source_format = "pdf"  # Override markdown format
            result.parser_name = "PDFParser"
            result.parser_version = "2.0"
            result.parse_time = time.time() - start_time
            result.meta.update(conversion_meta)
            result.meta["pdf_strategy"] = self.config.strategy
            result.meta["intermediate_markdown_length"] = len(markdown_content)
            result.meta["intermediate_markdown_preview"] = markdown_content[:500]

            logger.info(
                f"PDF parsed successfully: {pdf_path.name} "
                f"({len(markdown_content)} chars markdown, "
                f"{result.parse_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            return create_parse_result(
                root=ResourceNode(type=NodeType.ROOT),
                source_path=str(pdf_path),
                source_format="pdf",
                parser_name="PDFParser",
                parse_time=time.time() - start_time,
                warnings=[f"Failed to parse PDF: {e}"],
            )

    async def _convert_to_markdown(
        self,
        pdf_path: Path,
        resource_name: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Convert PDF to Markdown using configured strategy.

        Args:
            pdf_path: Path to PDF file
            resource_name: Optional resource name for organizing saved images

        Returns:
            Tuple of (markdown_content, metadata_dict)

        Raises:
            ValueError: If all conversion strategies fail
        """
        if self.config.strategy == "local":
            return await self._convert_local(pdf_path, resource_name=resource_name)

        elif self.config.strategy == "mineru":
            return await self._convert_mineru(pdf_path, resource_name=resource_name)

        elif self.config.strategy == "auto":
            # Try local first
            try:
                return await self._convert_local(pdf_path, resource_name=resource_name)
            except Exception as e:
                logger.warning(f"Local conversion failed: {e}")

                # Fallback to MinerU if configured
                if self.config.mineru_endpoint:
                    logger.info("Falling back to MinerU API")
                    return await self._convert_mineru(pdf_path, resource_name=resource_name)
                else:
                    raise ValueError(
                        f"Local conversion failed and no MinerU endpoint configured: {e}"
                    )

        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    async def _convert_local(
        self, pdf_path: Path, storage=None, resource_name: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        # Import the lightweight worker only once the package is fully loaded.
        from openviking.parse.document_worker import should_isolate

        if should_isolate(pdf_path):
            from openviking.parse.document_worker import convert_document_in_worker
            from openviking_cli.utils.storage import StoragePath, get_storage

            worker_storage = get_storage() if storage is None else storage
            # Custom storage implementations retain their existing in-process contract.
            if type(worker_storage) is StoragePath:
                result = await convert_document_in_worker(
                    "pdf", pdf_path, storage=worker_storage, resource_name=resource_name,
                    pdf_config={name: getattr(self.config, name) for name in (
                        "heading_detection", "font_heading_min_delta",
                        "max_heading_levels", "image_resolution",
                    )},
                )
                markdown = result["markdown"]
                meta = result["meta"]
                # 开关只在 _caption_images 内部判断，隔离路径也必须走同一入口。
                markdown = await self._caption_images(
                    markdown,
                    worker_storage,
                    meta=meta,
                    document_name=resource_name or pdf_path.stem,
                )
                return markdown, meta
        # pdfplumber / pdfminer 的解析与图片/表格提取通常是 CPU/IO 密集且为同步实现，
        # 放到线程池中执行，避免阻塞事件循环。
        markdown, meta = await asyncio.to_thread(
            self._convert_local_with_reclaim, pdf_path, storage, resource_name
        )
        # EXEPD-230092/230261：抽出来的图只有 "Page N Image M" 这种占位描述，下游（知识库检索侧）
        # 拿不到"这张图画的是什么"，于是模型看不见没有文字说明的实物图。
        # 这里在解析末尾给每张图补一句内容描述写进 alt；开关关着时逐字不变。
        markdown = await self._caption_images(
            markdown,
            storage,
            meta=meta,
            document_name=resource_name or pdf_path.stem,
        )
        return markdown, meta

    def _convert_local_with_reclaim(
        self, pdf_path: Path, storage=None, resource_name: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        # Run cleanup in the same worker after the converter's frame has exited.
        # Cancellation of the awaiting coroutine must not skip worker cleanup.
        try:
            return self._convert_local_sync(pdf_path, storage, resource_name)
        finally:
            self._reclaim_local_parse_memory(pdf_path)

    @staticmethod
    def _reclaim_local_parse_memory(pdf_path: Path) -> None:
        """Return unused native buffers after a large local PDF on Linux."""
        import sys

        try:
            # Small PDFs and platforms without glibc need no allocator operation.
            if sys.platform != "linux" or Path(pdf_path).stat().st_size < 16 * 1024 * 1024:
                return
            import ctypes
            import gc

            gc.collect()
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)
        except Exception:
            # Allocator availability or cleanup must not change a parse result.
            logger.debug("Large PDF memory reclaim unavailable", exc_info=True)

    # EXEPD-230261：图注要能直接回答「这张图是什么」——名称优先用同页正文里的说法，
    # 否则模型只能看到 [图片:hash]，符号类小图会被张冠李戴。
    _CAPTION_INSTRUCTION = (
        "这是知识文档中的一张小图。请用不超过 20 个字给出它的名称与含义，"
        "并写在 abstract 的最开头；名称优先采用 Surrounding Text Context 里出现的说法"
        "（例如「不可水洗」「最高洗涤30℃」「悬挂滴干」）。"
        "图里若标有数字或字母（如 30、40、110℃），必须原样写进 abstract。"
    )
    # 只在首轮返回空/[Image] 时使用一次；不做无界重试。
    _CAPTION_RETRY_INSTRUCTION = (
        "再仔细看这张图。只输出图中可直接看到的名称或文字，不超过 20 个字；"
        "不要输出 [Image]、空内容或解释；确实无法识别时只输出一个空行。"
    )

    async def _caption_images(
        self,
        markdown: str,
        storage,
        *,
        meta: Optional[Dict[str, Any]] = None,
        document_name: str = "",
    ) -> str:
        """给 markdown 里的每张图补可读 alt，并把计数写进解析 meta。

        开关只在这里判断，`_convert_local` 的隔离/非隔离两条路径都走同一个入口，
        避免某条路径绕过 `image_caption` / `image_caption_reuse_summary`。
        补描述是增强：任何失败都不能让入库失败，最差也保留原引用并计入
        `uncaptioned`，由调用方把问题暴露出去。
        """
        if not getattr(getattr(self, "config", None), "image_caption", False):
            return markdown
        try:
            markdown, report = await self._caption_images_with_report(
                markdown, storage, document_name=document_name
            )
        except Exception as exc:
            # 最后一道保险：图注流程自身的意外不能把整篇解析拖失败。
            logger.warning("Image caption pass failed; keeping original markdown: %s", exc)
            if meta is not None:
                meta.setdefault("images_total", 0)
                meta["caption_error"] = str(exc)[:200]
            return markdown
        if meta is not None:
            meta.update(report.to_meta())
        return markdown

    async def _caption_images_with_report(
        self, markdown: str, storage, *, document_name: str = ""
    ) -> tuple[str, ImageCaptionReport]:
        """分层生成图注：已有摘要复用 → VLM（最多两轮）→ 同页线索兜底 → 计数保留。

        计数口径（不变量）：
            images_total == captioned + fallback + reused + uncaptioned
        - captioned：本次 VLM 新产出的语义 alt（首轮或一次重试成功）
        - fallback：VLM 未产出时，用最近章节标题/同页首句生成的可追溯 alt
        - reused：已有语义 alt 或相同图片内容已产出的摘要，未重复调用 VLM
        - uncaptioned：既无同页线索、VLM 也失败，保留占位（必须可见）
        """
        report = ImageCaptionReport()
        if not markdown:
            return markdown, report

        if storage is None:
            try:
                from openviking_cli.utils.storage import get_storage

                storage = get_storage()
            except Exception as exc:  # pragma: no cover - 默认存储不可用时仍可兜底
                logger.warning("Image caption storage unavailable: %s", exc)
                storage = None
        media_dir = getattr(storage, "media_dir", None)

        processor = None
        try:
            from openviking.parse.vlm import VLMProcessor

            processor = VLMProcessor()
        except Exception as exc:  # pragma: no cover - 依赖缺失时降级到文本兜底
            logger.warning("Image caption VLM unavailable; using page-text fallback: %s", exc)

        limit = int(getattr(self.config, "image_caption_max_chars", 60))
        limit = min(max(limit, _CAPTION_MIN_CHARS), _CAPTION_MAX_CHARS)
        reuse = bool(getattr(self.config, "image_caption_reuse_summary", True))
        timeout = float(getattr(self.config, "image_caption_timeout", 30.0))

        matches = list(_IMAGE_REF_RE.finditer(markdown))
        report.images_total = len(matches)
        if not matches:
            return markdown, report

        # 同一次解析内的复用：相同图片字节只问一次 VLM；同页上下文只算一次。
        caption_cache: Dict[str, str] = {}
        failed_digests: set[str] = set()
        page_contexts: Dict[str, str] = {}
        page_bodies: Dict[str, str] = {}
        parts: List[str] = []
        cursor = 0

        for index, match in enumerate(matches, start=1):
            parts.append(markdown[cursor : match.start()])
            cursor = match.end()
            raw_alt = match.group(1)
            raw_target = match.group(2)
            target = raw_target.split(None, 1)[0].strip()
            page_no, image_no = self._locate_image(markdown, match, index)
            existing_semantic = is_semantic_caption(raw_alt, document_name, target)

            # 第 1 层：已有语义 alt 直接复用，不重复调 VLM。
            if existing_semantic and reuse:
                report.reused += 1
                parts.append(match.group(0))
                continue

            replacement = ""
            image_bytes = None
            if media_dir is not None:
                try:
                    image_file = Path(media_dir) / target
                    if image_file.is_file():
                        image_bytes = image_file.read_bytes()
                    else:
                        report.errors += 1
                except Exception as exc:
                    report.errors += 1
                    logger.debug("Image caption read failed for %s: %s", target, exc)

            if processor is not None and image_bytes:
                context = page_contexts.get(page_no)
                if context is None:
                    context = self._page_text_context(markdown, page_no)
                    page_contexts[page_no] = context
                digest = hashlib.sha256(image_bytes).hexdigest()
                if reuse and digest in failed_digests:
                    # 同内容图已经确认 VLM 失败过，不再为重复图浪费重试。
                    pass
                else:
                    cached = caption_cache.get(digest) if reuse else ""
                    if cached:
                        replacement = cached
                        report.reused += 1
                    else:
                        # 第 2 层：空/占位/异常最多重试一次，用更严格的提示词。
                        for attempt, instruction in enumerate(
                            (self._CAPTION_INSTRUCTION, self._CAPTION_RETRY_INSTRUCTION)
                        ):
                            if attempt:
                                report.vlm_retries += 1
                            report.vlm_calls += 1
                            try:
                                result = await asyncio.wait_for(
                                    processor.understand_image(
                                        image_bytes, context=context, instruction=instruction
                                    ),
                                    timeout=timeout,
                                )
                                caption = _clean_caption(getattr(result, "abstract", ""), limit)
                            except Exception as exc:
                                logger.debug(
                                    "Image caption VLM attempt %d failed for %s: %s",
                                    attempt + 1,
                                    target,
                                    exc,
                                )
                                caption = ""
                            if is_semantic_caption(caption, document_name, target):
                                replacement = caption
                                report.captioned += 1
                                if reuse:
                                    caption_cache[digest] = caption
                                break
                        if not replacement and reuse:
                            failed_digests.add(digest)

            # 复用关闭但 VLM 没给出更好结果时，保留已有语义 alt，绝不回退占位。
            if not replacement and existing_semantic:
                replacement = _clean_caption(raw_alt, limit)
                report.reused += 1

            # 第 3 层：同页可定位文本兜底，形状「<小节标题>（图 N）」。
            if not replacement:
                body = page_bodies.get(page_no)
                if body is None:
                    body = self._page_body(markdown, page_no)
                    page_bodies[page_no] = body
                fallback = self._fallback_alt(
                    markdown,
                    match.start(),
                    page_no,
                    image_no,
                    document_name,
                    limit,
                    page_body=body,
                    image_name=target,
                )
                if fallback:
                    replacement = fallback
                    report.fallback += 1

            # 第 4 层：无任何线索，保留占位并计数，由末尾聚合 WARNING 暴露。
            if not replacement:
                report.uncaptioned += 1
                if len(report.uncaptioned_samples) < 5:
                    report.uncaptioned_samples.append(target)
                parts.append(match.group(0))
            else:
                safe_alt = _clean_caption(replacement, limit)
                parts.append(f"![{safe_alt}]({raw_target})")

        parts.append(markdown[cursor:])
        output = "".join(parts)

        problems = report.problems()
        if problems:
            logger.error("Image caption accounting invariant violated: %s", "; ".join(problems))
        audit = audit_image_alts(output, document_name)
        expected_semantic = report.captioned + report.fallback + report.reused
        if (
            audit.total != report.images_total
            or audit.semantic != expected_semantic
            or len(audit.invalid) != report.uncaptioned
        ):
            logger.warning(
                "Image caption output audit drift: total=%d/%d semantic=%d/%d uncaptioned=%d/%d",
                audit.total,
                report.images_total,
                audit.semantic,
                expected_semantic,
                len(audit.invalid),
                report.uncaptioned,
            )

        # 只打聚合日志：单图失败不再逐条 WARNING，避免 244 图刷屏。
        logger.info(
            "PDF image captions: total=%d captioned=%d fallback=%d reused=%d "
            "uncaptioned=%d vlm_calls=%d retries=%d errors=%d",
            report.images_total,
            report.captioned,
            report.fallback,
            report.reused,
            report.uncaptioned,
            report.vlm_calls,
            report.vlm_retries,
            report.errors,
        )
        if report.uncaptioned:
            logger.warning(
                "PDF image captions: %d/%d image(s) left without semantic alt "
                "(no same-page clue and VLM failed): %s",
                report.uncaptioned,
                report.images_total,
                ", ".join(report.uncaptioned_samples),
            )
        return output, report

    @staticmethod
    def _locate_image(markdown: str, match: re.Match, fallback_index: int) -> tuple[str, str]:
        """定位图片所属页/序号：优先占位 alt，其次最近的图片/页面注释。"""
        placeholder = re.match(
            r"^Page\s+(\d+)\s+Image\s+(\d+)$",
            _normalize_alt(match.group(1)),
            re.IGNORECASE,
        )
        if placeholder:
            return placeholder.group(1), placeholder.group(2)

        prefix = markdown[: match.start()]
        image_markers = list(_IMAGE_MARKER_RE.finditer(prefix))
        if image_markers:
            last = image_markers[-1]
            # 中间不能再有别的图片引用，否则这个注释属于更早的图。
            if not _IMAGE_REF_RE.search(markdown[last.end() : match.start()]):
                return last.group(1), last.group(2)

        page_markers = list(_PAGE_MARKER_RE.finditer(prefix))
        if page_markers:
            last_page = page_markers[-1]
            segment = markdown[last_page.end() : match.start()]
            return last_page.group(1), str(len(_IMAGE_REF_RE.findall(segment)) + 1)
        return "", str(fallback_index)

    def _fallback_alt(
        self,
        markdown: str,
        match_start: int,
        page_no: str,
        image_no: str,
        document_name: str,
        limit: int,
        page_body: Optional[str] = None,
        image_name: str = "",
    ) -> str:
        """用同页可定位文本生成可追溯 alt；无线索时返回空串（由调用方计数）。"""
        heading = self._nearest_heading_text(markdown, match_start)
        # 文档/章节开头的图前面没有任何标题（page=None 的首图就是这样）：它属于**紧随其后**
        # 的那一节，用后面的标题做线索仍然是可追溯的，比留占位或回落到文档名更准确。
        if not heading:
            heading = self._next_heading_text(markdown, match_start)
        clue = (
            heading
            if heading and is_semantic_caption(heading, document_name, image_name)
            else ""
        )
        if not clue and page_no:
            body = page_body if page_body is not None else self._page_body(markdown, page_no)
            sentence = self._first_sentence(body)
            if sentence and is_semantic_caption(sentence, document_name, image_name):
                clue = sentence
        if not clue:
            return ""
        suffix = f"（图 {image_no}）"
        clue = self._clip_clue(clue, max(1, limit - len(suffix)))
        return f"{clue}{suffix}" if clue else ""

    @staticmethod
    def _clip_clue(clue: str, limit: int) -> str:
        """把标题/首句裁到指定长度，优先在标点处断开，避免 alt 停在半个词上。"""
        clue = " ".join((clue or "").split())
        if len(clue) <= limit:
            return clue
        window = clue[:limit]
        cut = max(window.rfind(ch) for ch in "，,、；;：: ")
        if cut >= 2:
            window = window[:cut]
        return window.rstrip("，,、；;：: ")

    @staticmethod
    def _nearest_heading_text(markdown: str, position: int, scan_chars: int = 20000) -> str:
        """取图片之前最近的 markdown 标题；只回看有限窗口，避免整篇扫描。"""
        window = markdown[max(0, position - scan_chars) : position]
        matches = list(_HEADING_LINE_RE.finditer(window))
        if not matches:
            return ""
        title = re.sub(r"[*_`]+", "", matches[-1].group(2))
        title = re.sub(r"<[^>]+>", " ", title)
        return " ".join(title.split()).strip(" #")

    @staticmethod
    def _next_heading_text(markdown: str, position: int, scan_chars: int = 4000) -> str:
        """取图片之后最近的 markdown 标题（只回看有限窗口，避免跨章节串味）。"""
        window = markdown[position : position + scan_chars]
        matches = list(_HEADING_LINE_RE.finditer(window))
        if not matches:
            return ""
        title = re.sub(r"[*_`]+", "", matches[0].group(2))
        title = re.sub(r"<[^>]+>", " ", title)
        return " ".join(title.split()).strip(" #")

    @staticmethod
    def _first_sentence(text: str, limit: int = 40) -> str:
        text = " ".join((text or "").split()).strip()
        if not text:
            return ""
        match = re.search(r"[。！？!?；;]", text)
        sentence = text[: match.end()] if match else text
        sentence = " ".join(sentence.strip(" #*-•·|　").split())
        return PDFParser._clip_clue(sentence, limit)

    @staticmethod
    def _page_body(markdown: str, page_no: str) -> str:
        """取 `<!-- Page N -->` 到下一页标记之间的正文（去图片/注释/HTML/标题）。

        解析器把下一页的标题插在下一页 page marker 之前，因此这里剔除标题行，
        避免把下一节的标题当成当前页正文（标题由 `_nearest_heading_text` 单独取）。
        """
        blocks = list(_PAGE_MARKER_RE.finditer(markdown))
        for index, block in enumerate(blocks):
            if block.group(1) != str(page_no):
                continue
            end = blocks[index + 1].start() if index + 1 < len(blocks) else len(markdown)
            text = markdown[block.end() : end]
            text = _IMAGE_REF_RE.sub(" ", text)
            text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = _HEADING_LINE_RE.sub(" ", text)
            return " ".join(text.split())
        return ""

    def _page_text_context(self, markdown: str, page_no: str, limit: int = 500) -> str:
        """取该页正文 + 最近章节标题作为图注上下文。"""
        body = self._page_body(markdown, page_no)
        if not body:
            return ""
        marker = next(
            (m for m in _PAGE_MARKER_RE.finditer(markdown) if m.group(1) == str(page_no)),
            None,
        )
        heading = self._nearest_heading_text(markdown, marker.start()) if marker else ""
        text = f"{heading} {body}" if heading and heading not in body else body
        return " ".join(text.split())[:limit]

    def _convert_local_sync(
        self, pdf_path: Path, storage=None, resource_name: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """同步版：用 pdfplumber 将 PDF 转 Markdown。

        该方法会在 :meth:`_convert_local` 中通过 asyncio.to_thread 调用。
        """
        pdfplumber = lazy_import("pdfplumber")

        # Import storage utilities
        if storage is None:
            from openviking_cli.utils.storage import get_storage

            storage = get_storage()

        if resource_name is None:
            resource_name = pdf_path.stem

        parts = []
        meta = {
            "strategy": "local",
            "library": "pdfplumber",
            "pages_processed": 0,
            "images_extracted": 0,
            "images_deduplicated": 0,
            "images_nested_dropped": 0,
            "tables_extracted": 0,
            "bookmarks_found": 0,
            "bookmarks_resolved": 0,
            "bookmarks_unresolved": 0,
            "headings_found": 0,
            "heading_source": "none",
        }

        try:
            from openviking.parse.parsers.pdf_compat import open_pdf_compat

            with open_pdf_compat(pdf_path, pdfplumber, meta, _PDF_RENDER_LOCK) as pdf:
                meta["total_pages"] = len(pdf.pages)

                # Extract structure (bookmarks → font fallback)
                detection_mode = self.config.heading_detection
                bookmarks = []
                raw_bookmarks = []
                heading_source = "none"

                if detection_mode in ("bookmarks", "auto"):
                    raw_bookmarks = self._extract_bookmarks(pdf)
                    meta["bookmarks_found"] = len(raw_bookmarks)
                    bookmarks = [bm for bm in raw_bookmarks if bm["page_num"] is not None]
                    meta["bookmarks_resolved"] = len(bookmarks)
                    meta["bookmarks_unresolved"] = len(raw_bookmarks) - len(bookmarks)

                    if bookmarks:
                        heading_source = "bookmarks"
                    elif raw_bookmarks:
                        logger.info(
                            "Bookmark detection found %d entries but none resolved to pages; "
                            "ignoring bookmark headings",
                            len(raw_bookmarks),
                        )

                if not bookmarks and detection_mode in ("font", "auto"):
                    bookmarks = self._detect_headings_by_font(pdf)
                    if bookmarks:
                        heading_source = "font_analysis"

                meta["headings_found"] = len(bookmarks)
                meta["heading_source"] = heading_source
                logger.info(
                    "Heading detection: source=%s, headings=%d, bookmarks=%d, resolved=%d, "
                    "unresolved=%d",
                    heading_source,
                    len(bookmarks),
                    meta["bookmarks_found"],
                    meta["bookmarks_resolved"],
                    meta["bookmarks_unresolved"],
                )

                # Group bookmarks by page_num
                bookmarks_by_page = defaultdict(list)
                for bm in bookmarks:
                    page = bm["page_num"]
                    if page is None:
                        continue
                    bookmarks_by_page[page].append(bm)

                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Inject headings before page text
                        page_bookmarks = bookmarks_by_page.get(page_num, [])
                        for bm in page_bookmarks:
                            heading_prefix = "#" * bm["level"]
                            parts.append(f"\n{heading_prefix} {bm['title']}\n")

                        # Extract text
                        text = page.extract_text()
                        if text and text.strip():
                            # Add page marker as HTML comment
                            parts.append(f"<!-- Page {page_num} -->\n{text.strip()}")
                            meta["pages_processed"] += 1

                        # Extract tables
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables or []):
                            if table and len(table) > 0:
                                md_table = self._format_table_markdown(table)
                                if md_table:
                                    parts.append(
                                        f"<!-- Page {page_num} Table {table_idx + 1} -->\n{md_table}"
                                    )
                                    meta["tables_extracted"] += 1

                        # Extract images.
                        #
                        # A page can stack several image XObjects on the exact same
                        # spot — print-to-PDF producers routinely emit a background
                        # layer plus a content layer. Since extraction rasterises the
                        # page *region* rather than the XObject itself, every one of
                        # them renders to identical bytes. Skip the repeats: bbox
                        # first, which avoids the (expensive) render entirely, then a
                        # content hash as a backstop. Both sets are per-page, so a
                        # header logo repeated across pages is still kept once per
                        # page.
                        images = page.images or []
                        # EXEPD-230436：纯扫描页的多层图先归一为「整页底图」一张。
                        # 只在这一页没有文本层时启用；底图必须先成功抽出来，否则宁可
                        # 退回旧行为（逐块抽）也不能把整页内容丢掉。
                        cover_index = None
                        cover_image = None
                        cover_bytes = None
                        if not (text and text.strip()):
                            cover_index = self._page_covering_image_index(
                                images,
                                float(getattr(page, "width", 0) or 0),
                                float(getattr(page, "height", 0) or 0),
                            )
                            if cover_index is not None:
                                cover_image = images[cover_index]
                                try:
                                    cover_bytes = self._extract_image_from_page(page, cover_image)
                                except Exception as cover_err:
                                    cover_bytes = None
                                    logger.warning(
                                        f"Failed to extract page-covering image {cover_index + 1} "
                                        f"on page {page_num}: {cover_err}"
                                    )
                                if not cover_bytes:
                                    # 底图抽不出来 ⇒ 不启用嵌套剔除，逐块照旧
                                    cover_index, cover_image = None, None
                        else:
                            cover_bytes = None
                        seen_boxes = set()
                        seen_digests = set()
                        for img_idx, img in enumerate(images):
                            try:
                                # 被整页底图完全包含的图层（同一页的重复栅格）不再单独出图：
                                # 底图已包含它的全部像素，单独抽只会多出一条「半句话」证据。
                                if (
                                    cover_image is not None
                                    and img_idx != cover_index
                                    and self._is_nested_image(img, cover_image)
                                ):
                                    meta["images_nested_dropped"] += 1
                                    continue
                                bbox_key = (
                                    round(img["x0"], 1),
                                    round(img["top"], 1),
                                    round(img["x1"], 1),
                                    round(img["bottom"], 1),
                                )
                                if bbox_key in seen_boxes:
                                    meta["images_deduplicated"] += 1
                                    continue
                                seen_boxes.add(bbox_key)

                                # Extract image using underlying PDF object
                                image_obj = (
                                    cover_bytes
                                    if img_idx == cover_index and cover_bytes
                                    else self._extract_image_from_page(page, img)
                                )
                                if image_obj:
                                    # Dedup only — md5 keeps this cheap, and the
                                    # flag keeps it working on FIPS-locked hosts.
                                    digest = hashlib.md5(image_obj, usedforsecurity=False).digest()
                                    if digest in seen_digests:
                                        meta["images_deduplicated"] += 1
                                        continue
                                    seen_digests.add(digest)

                                    # Save image
                                    filename = f"page{page_num}_img{img_idx + 1}"
                                    image_path = storage.save_image(
                                        resource_name, image_obj, filename=filename
                                    )

                                    # Generate path relative to the media root.
                                    rel_path = image_path.relative_to(storage.media_dir)
                                    parts.append(
                                        f"<!-- Page {page_num} Image {img_idx + 1} -->\n"
                                        f"![Page {page_num} Image {img_idx + 1}]({rel_path})"
                                    )
                                    meta["images_extracted"] += 1
                            except Exception as img_err:
                                logger.warning(
                                    f"Failed to extract image {img_idx + 1} on page {page_num}: {img_err}"
                                )
                    finally:
                        self._release_page_cache(page)

            if not parts:
                logger.warning(f"No content extracted from {pdf_path}")
                return "", meta

            markdown_content = "\n\n".join(parts)
            logger.info(
                f"Local conversion: {meta['pages_processed']}/{meta['total_pages']} pages, "
                f"{meta['headings_found']} headings ({meta['heading_source']}, "
                f"bookmarks={meta['bookmarks_found']}, "
                f"resolved={meta['bookmarks_resolved']}), "
                f"{meta['images_extracted']} images "
                f"({meta['images_deduplicated']} duplicates skipped, "
                f"{meta['images_nested_dropped']} nested layers skipped), "
                f"{meta['tables_extracted']} tables → "
                f"{len(markdown_content)} chars"
            )

            return markdown_content, meta

        except Exception as e:
            logger.error(f"pdfplumber conversion failed: {e}")
            raise

    @staticmethod
    def _release_page_cache(page: Any) -> None:
        """Release pdfplumber/pdfminer per-page caches when available."""
        rendered = getattr(page, "_openviking_rendered_page", None)
        if rendered is not None:
            delattr(page, "_openviking_rendered_page")
            for image in (rendered.original, rendered.annotated):
                image.close()
        close = getattr(page, "close", None)
        if callable(close):
            try:
                close()
                return
            except Exception:
                pass

        flush_cache = getattr(page, "flush_cache", None)
        if callable(flush_cache):
            try:
                flush_cache()
            except Exception:
                pass

    def _extract_bookmarks(self, pdf) -> List[Dict[str, Any]]:
        """Extract bookmark structure from PDF outlines.

        Returns: [{level: int, title: str, page_num: int(1-based)}]
        """
        try:
            if not hasattr(pdf, "doc") or not hasattr(pdf.doc, "get_outlines"):
                return []

            outlines = list(pdf.doc.get_outlines())
            if not outlines:
                return []

            page_ref_to_num = self._build_page_number_map(pdf)

            bookmarks = []
            for level, title, dest, _action, _se in outlines:
                if not title or not title.strip():
                    continue

                page_num = None
                try:
                    if dest and len(dest) > 0:
                        page_num = self._resolve_bookmark_page(
                            dest[0], page_ref_to_num, len(pdf.pages)
                        )
                except Exception:
                    pass

                bookmarks.append(
                    {
                        "level": min(max(level, 1), 6),
                        "title": title.strip(),
                        "page_num": page_num,
                    }
                )

            return bookmarks

        except Exception as e:
            logger.warning(f"Failed to extract bookmarks: {e}")
            return []

    def _build_page_number_map(self, pdf) -> Dict[int, int]:
        """Build a lookup from PDF page object ids to 1-based page numbers.

        pdfminer outlines and link annotations reference page objects by object id.
        In pdfplumber these ids are exposed as ``page.page_obj.pageid``; some mocks
        or alternate inputs may still expose ``objid``, so we keep both.
        """
        page_ref_to_num: Dict[int, int] = {}
        for page_num, page in enumerate(pdf.pages, 1):
            page_obj = getattr(page, "page_obj", None)
            if page_obj is None:
                continue

            for attr_name in ("pageid", "objid"):
                ref_id = getattr(page_obj, attr_name, None)
                if isinstance(ref_id, int):
                    page_ref_to_num.setdefault(ref_id, page_num)

        return page_ref_to_num

    def _resolve_bookmark_page(
        self, page_ref: Any, page_ref_to_num: Dict[int, int], total_pages: int
    ) -> Optional[int]:
        """Resolve a bookmark destination to a 1-based page number."""
        ref_id = getattr(page_ref, "objid", None)
        if isinstance(ref_id, int):
            return page_ref_to_num.get(ref_id)

        if isinstance(page_ref, int):
            # 0-based integer page index (common in many PDF producers)
            candidate = page_ref + 1
            if 1 <= candidate <= total_pages:
                return candidate
            return None

        if hasattr(page_ref, "resolve"):
            resolved = page_ref.resolve()
            for attr_name in ("pageid", "objid"):
                resolved_id = getattr(resolved, attr_name, None)
                if isinstance(resolved_id, int):
                    return page_ref_to_num.get(resolved_id)

        return None

    def _detect_headings_by_font(self, pdf) -> List[Dict[str, Any]]:
        """Detect headings by font size analysis.

        Returns: [{level: int, title: str, page_num: int(1-based)}]
        """
        try:
            # Step 1: Sample font size distribution (every 5th page)
            size_counter: Counter = Counter()
            sample_pages = pdf.pages[::5]
            for page in sample_pages:
                try:
                    for char in page.chars:
                        if char["text"].strip():
                            rounded = round(char["size"] * 2) / 2
                            size_counter[rounded] += 1
                finally:
                    self._release_page_cache(page)

            if not size_counter:
                return []

            # Step 2: Determine body font size and heading font sizes
            body_size = size_counter.most_common(1)[0][0]
            min_delta = self.config.font_heading_min_delta

            heading_sizes = sorted(
                [
                    s
                    for s, count in size_counter.items()
                    if s >= body_size + min_delta and count < size_counter[body_size] * 0.5
                ],
                reverse=True,
            )

            max_levels = self.config.max_heading_levels
            heading_sizes = heading_sizes[:max_levels]

            if not heading_sizes:
                logger.debug(f"Font analysis: body_size={body_size}pt, no heading sizes found")
                return []

            size_to_level = {s: i + 1 for i, s in enumerate(heading_sizes)}
            logger.debug(
                f"Font analysis: body_size={body_size}pt, "
                f"heading_sizes={heading_sizes}, size_to_level={size_to_level}"
            )

            # Step 3: Extract heading text page by page
            headings: List[Dict[str, Any]] = []

            def flush_line(chars_to_flush: list, page_num: int) -> None:
                if not chars_to_flush:
                    return
                title = "".join(c["text"] for c in chars_to_flush).strip()
                size = round(chars_to_flush[0]["size"] * 2) / 2

                if len(title) < 2:
                    return
                if len(title) > 100:
                    return
                if title.isdigit():
                    return
                if re.match(r"^[\d\s.·…]+$", title):
                    return

                headings.append(
                    {
                        "level": size_to_level[size],
                        "title": title,
                        "page_num": page_num,
                    }
                )

            for page in pdf.pages:
                try:
                    page_num = page.page_number + 1
                    chars = sorted(page.chars, key=lambda c: (c["top"], c["x0"]))

                    current_line_chars: list = []
                    current_top = None

                    for char in chars:
                        # Performance: headings won't appear in bottom 70% of page
                        if char["top"] > page.height * 0.3:
                            flush_line(current_line_chars, page_num)
                            current_line_chars = []
                            break

                        rounded_size = round(char["size"] * 2) / 2
                        if rounded_size not in size_to_level:
                            flush_line(current_line_chars, page_num)
                            current_line_chars = []
                            current_top = None
                            continue

                        # Same line check (top offset < 2pt)
                        if current_top is not None and abs(char["top"] - current_top) > 2:
                            flush_line(current_line_chars, page_num)
                            current_line_chars = []

                        current_line_chars.append(char)
                        current_top = char["top"]

                    flush_line(current_line_chars, page_num)
                finally:
                    self._release_page_cache(page)

            # Step 4: Deduplicate - filter headers appearing on >30% of pages
            title_page_count: Counter = Counter(h["title"] for h in headings)
            total_pages = len(pdf.pages)
            header_titles = {t for t, c in title_page_count.items() if c > total_pages * 0.3}
            headings = [h for h in headings if h["title"] not in header_titles]

            logger.debug(
                f"Font heading detection: {len(headings)} headings found "
                f"(filtered {len(header_titles)} header titles)"
            )
            return headings

        except Exception as e:
            logger.warning(f"Failed to detect headings by font: {e}")
            return []

    @staticmethod
    def _page_covering_image_index(
        images: List[Dict[str, Any]], page_width: float, page_height: float
    ) -> Optional[int]:
        """返回覆盖整页的底图下标；没有则 None。

        「整页」按面积比判定（≥ ``PAGE_COVERING_AREA_RATIO``），不要求 bbox 完全等于页面：
        扫描件底图常带 1pt 以内的越界（现场 bbox top=-0.7）。多张候选时取面积最大的一张。
        """
        if not images or page_width <= 0 or page_height <= 0:
            return None
        page_area = page_width * page_height
        best_index = None
        best_area = 0.0
        for index, img in enumerate(images):
            width = img["x1"] - img["x0"]
            height = img["bottom"] - img["top"]
            if width <= 0 or height <= 0:
                continue
            area = width * height
            if area / page_area >= PAGE_COVERING_AREA_RATIO and area > best_area:
                best_index, best_area = index, area
        return best_index

    @staticmethod
    def _is_nested_image(img: Dict[str, Any], cover: Dict[str, Any]) -> bool:
        """img 的 bbox 是否完全落在 cover 内（容差 ``NESTED_CONTAINMENT_TOLERANCE_PT``）。"""
        tolerance = NESTED_CONTAINMENT_TOLERANCE_PT
        return (
            img["x0"] >= cover["x0"] - tolerance
            and img["top"] >= cover["top"] - tolerance
            and img["x1"] <= cover["x1"] + tolerance
            and img["bottom"] <= cover["bottom"] + tolerance
        )

    def _extract_image_from_page(self, page, img_info: dict) -> Optional[bytes]:
        """
        Extract a PDF image as valid PNG bytes.

        Renders the image's bounding box on the page to a raster PNG via
        pdfplumber's ``crop().to_image()`` instead of returning the raw decoded
        XObject stream (which is not a valid image file and cannot be opened).

        Args:
            page: pdfplumber page object
            img_info: Image metadata from page.images

        Returns:
            PNG-encoded image bytes or None if extraction fails
        """
        try:
            # pdfplumber coordinates: ``top`` is measured from the top of the page.
            bbox = (
                max(0, img_info["x0"]),
                max(0, img_info["top"]),
                min(page.width, img_info["x1"]),
                min(page.height, img_info["bottom"]),
            )

            # Skip degenerate / zero-area boxes that cannot be cropped.
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return None

            cropped = page.crop(bbox)
            with _PDF_RENDER_LOCK:
                # Rendering a crop still rasterizes the entire PDF page. Reuse
                # that raster for all images on an ordinary, uncropped page.
                # Non-standard crop boxes retain pdfplumber's original path.
                if getattr(page, "is_original", False) and page.bbox == page.cropbox:
                    from pdfplumber.display import PageImage

                    rendered = getattr(page, "_openviking_rendered_page", None)
                    if rendered is None:
                        rendered = page.to_image(resolution=self.config.image_resolution)
                        page._openviking_rendered_page = rendered
                    page_image = PageImage(
                        cropped, original=rendered.original, resolution=rendered.resolution
                    )
                else:
                    page_image = cropped.to_image(resolution=self.config.image_resolution)
                buffer = io.BytesIO()
                page_image.save(buffer, format="PNG")
                return buffer.getvalue()

        except Exception as e:
            logger.debug(f"Image extraction error: {e}")
            return None

    async def _convert_mineru(
        self,
        pdf_path: Path,
        storage=None,
        resource_name: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Convert PDF to Markdown using the MinerU /file_parse API.

        Args:
            pdf_path: Path to PDF file
            storage: Media storage used to persist extracted images; defaults to
                the configured storage if None
            resource_name: Resource name under which extracted images are saved;
                defaults to the PDF stem

        Returns:
            Tuple of (markdown_content, metadata) where metadata includes
            strategy, endpoint, api_version, backend, task_id, processing_time
            (seconds) and images_saved

        Raises:
            ValueError: If MinerU endpoint is not configured, or the task does
                not complete
            Exception: If the API call fails
        """
        httpx = lazy_import("httpx")

        if not self.config.mineru_endpoint:
            raise ValueError("MinerU endpoint not configured")

        # mineru_endpoint is a base URL; append /file_parse unless it already ends with it
        base = self.config.mineru_endpoint.rstrip("/")
        url = base if base.endswith("/file_parse") else f"{base}/file_parse"

        meta = {
            "strategy": "mineru",
            "endpoint": self.config.mineru_endpoint,
            "api_version": None,
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.mineru_timeout) as client:
                # Prepare file upload
                with open(pdf_path, "rb") as f:
                    files = {"files": (pdf_path.name, f, "application/pdf")}

                    # Prepare Form fields
                    data: Dict[str, Any] = dict(self.config.mineru_bodys or {})

                    # MinerU must return extracted images for the markdown refs.
                    data["return_images"] = True

                    # Make API request
                    logger.info(f"Calling MinerU API: {url}")
                    response = await client.post(
                        url,
                        files=files,
                        data=data,
                    )
                    response.raise_for_status()

            # Parse response
            result = response.json()
            if result.get("status") != "completed":
                raise ValueError(f"MinerU task not completed: {result.get('status')}")

            results = result.get("results") or {}
            file_result = results.get(pdf_path.name) or next(iter(results.values()), {})
            markdown_content = file_result.get("md_content") or ""

            # Extract metadata from response
            meta["api_version"] = result.get("version")
            meta["backend"] = result.get("backend")
            meta["task_id"] = result.get("task_id")
            started_at, completed_at = result.get("started_at"), result.get("completed_at")
            if started_at and completed_at:
                meta["processing_time"] = (
                    parse_iso_datetime(completed_at) - parse_iso_datetime(started_at)
                ).total_seconds()

            if not markdown_content:
                logger.warning(f"MinerU returned empty content for {pdf_path}")
                return markdown_content, meta

            if storage is None:
                from openviking_cli.utils.storage import get_storage

                storage = get_storage()

            if resource_name is None:
                resource_name = pdf_path.stem

            # MinerU embeds images as base64 data-URLs, referenced from markdown
            # as `images/<filename>`; save them into the media store and rewrite
            # the references to the stored relative paths.
            repl: Dict[str, str] = {}
            media_dir = storage.media_dir
            for img_name, data_url in (file_result.get("images") or {}).items():
                try:
                    # data URL form: "data:image/jpeg;base64,<b64>"
                    image_bytes = base64.b64decode(data_url.split(",", 1)[-1])
                    img_path = Path(img_name)
                    image_path = storage.save_image(
                        resource_name,
                        image_bytes,
                        filename=img_path.stem,
                        extension=img_path.suffix or ".png",
                    )
                    repl[f"images/{img_name}"] = image_path.relative_to(media_dir).as_posix()
                except Exception as img_err:
                    logger.warning(f"Failed to save MinerU image {img_name}: {img_err}")

            if repl:
                # Single pass over the markdown replaces every known reference;
                # unknown ``images/...`` text is left untouched.
                ref_pattern = re.compile(
                    "|".join(re.escape(ref) for ref in sorted(repl, key=len, reverse=True))
                )
                markdown_content = ref_pattern.sub(lambda m: repl[m.group(0)], markdown_content)
                meta["images_saved"] = len(repl)

            logger.info(f"MinerU conversion: {len(markdown_content)} chars")

            return markdown_content, meta

        except Exception as e:
            logger.error(f"MinerU API call failed: {e}")
            raise

    def _format_table_markdown(self, table: List[List[Optional[str]]]) -> str:
        """
        Convert table data to Markdown table format.

        Args:
            table: 2D array of table cells

        Returns:
            Markdown table string

        Examples:
            >>> table = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
            >>> print(parser._format_table_markdown(table))
            | Name | Age |
            | --- | --- |
            | Alice | 30 |
            | Bob | 25 |
        """
        if not table or not table[0]:
            return ""

        # Clean cells and handle None values
        def clean_cell(cell):
            if cell is None:
                return ""
            return str(cell).strip().replace("|", "\\|")  # Escape pipe characters

        lines = []

        # Header row
        header = table[0]
        header_cells = [clean_cell(cell) for cell in header]
        lines.append("| " + " | ".join(header_cells) + " |")

        # Separator row
        separator = ["---"] * len(header)
        lines.append("| " + " | ".join(separator) + " |")

        # Data rows
        for row in table[1:]:
            # Pad row to match header length
            padded_row = row + [None] * (len(header) - len(row))
            cells = [clean_cell(cell) for cell in padded_row[: len(header)]]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    async def parse_content(
        self, content: str, source_path: Optional[str] = None, instruction: str = "", **kwargs
    ) -> ParseResult:
        """
        Parse PDF content string.

        Note: This method is not recommended for PDFParser as it requires
        file path for conversion tools. Use parse() with file path instead.

        Args:
            content: PDF content (not supported)
            source_path: Optional source path
            **kwargs: Additional options

        Raises:
            NotImplementedError: PDFParser requires file path
        """
        raise NotImplementedError(
            "PDFParser does not support parsing content strings. "
            "Use parse() with a file path instead."
        )
