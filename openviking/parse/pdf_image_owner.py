import math
import re
from typing import Any, Optional


PRODUCT_CODE = re.compile(r"(?:货号|款号)\s*[:：]\s*([A-Za-z0-9][A-Za-z0-9_-]{3,39})(?![A-Za-z0-9_-])")


def image_product_owner(page: Any, image: dict) -> Optional[str]:
    image_box = _box(image)
    if image_box is None:
        return None
    image_area = _area(image_box)
    owners = set()
    for rectangle in page.rects:
        region = _box(rectangle)
        if region is None or _intersection(image_box, region) / image_area < 0.8:
            continue
        codes = set()
        for word in page.extract_words():
            word_box = _box(word)
            if word_box is None or _intersection(word_box, region) / _area(word_box) < 0.95:
                continue
            codes.update(PRODUCT_CODE.findall(word.get("text", "")))
        if len(codes) == 1:
            owners.update(codes)
        elif codes:
            return None
    return next(iter(owners)) if len(owners) == 1 else None


def _box(value: dict) -> Optional[tuple[float, float, float, float]]:
    try:
        coordinates = tuple(float(value[key]) for key in ("x0", "top", "x1", "bottom"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in coordinates):
        return None
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        return None
    return coordinates


def _area(box: tuple) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(first: tuple, second: tuple) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
