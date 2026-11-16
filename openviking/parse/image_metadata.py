# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One image-metadata contract for every parser that extracts pictures.

Why this module exists
----------------------
Knowledge service only learns an image's page/size/content hash from the HTML
comment that sits **immediately before** the Markdown image link:

    <!-- Page 7 Image 2 w=1750 h=1254 sha256=<64 hex> -->

Parsers used to write different (or no) comments: the PDF parser wrote only
``<!-- Page n Image m -->``, the AnyDoc renderer wrote no comment at all, so
``contentHash``/``width``/``height`` were permanently null for Office documents
and cross-URI image dedupe could not work. This module is the single place that
formats the comment so the two parsers cannot drift apart again.

Compatibility rules (enforced by knowledge-service ``SnippetText``):
  * the comment must contain the standalone word ``Image``;
  * ``w=`` / ``h=`` are at most 5 digits, ``sha256=`` is exactly 64 hex chars;
  * the comment must be glued to the image link -- no visible text between them.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Optional

_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class ImageMetadata:
    """Content identity of one extracted image byte string."""

    sha256: str
    width: Optional[int] = None
    height: Optional[int] = None


def describe_image(data: bytes) -> ImageMetadata:
    """Compute the content hash (and pixel size when the image can be decoded).

    The hash is taken over the **bytes that are saved to disk**, so downstream
    consumers can verify it after downloading the resource. Size is best effort:
    a decode failure still yields a usable content hash.
    """
    digest = hashlib.sha256(data).hexdigest()
    width: Optional[int] = None
    height: Optional[int] = None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - metadata is optional, never fail parsing
        width = height = None
    if width is not None and height is not None and (width <= 0 or height <= 0):
        width = height = None
    return ImageMetadata(sha256=digest, width=width, height=height)


def format_comment(
    metadata: ImageMetadata,
    *,
    page: Optional[int] = None,
    index: Optional[int] = None,
) -> str:
    """Return the HTML comment that must precede the image link.

    ``page``/``index`` are kept for the PDF parser (its pages are meaningful);
    Office conversions have no page concept and omit both.
    """
    if page is not None and index is not None:
        label = f"Page {int(page)} Image {int(index)}"
    else:
        label = "Image"
    parts = [label]
    if metadata.width is not None and metadata.height is not None:
        parts.append(f"w={int(metadata.width)} h={int(metadata.height)}")
    sha = (metadata.sha256 or "").strip().lower()
    if len(sha) == _SHA256_HEX_LENGTH:
        parts.append(f"sha256={sha}")
    return "<!-- " + " ".join(parts) + " -->"


def markdown_with_comment(comment: str, markdown_image: str) -> str:
    """Glue the comment to the image link; knowledge service pairs them by adjacency."""
    if not comment:
        return markdown_image
    return f"{comment}\n{markdown_image}"
