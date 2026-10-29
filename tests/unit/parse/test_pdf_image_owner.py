from types import SimpleNamespace

from openviking.parse.pdf_image_owner import image_product_owner


def box(left, top, right, bottom, **extra):
    return dict(x0=left, top=top, x1=right, bottom=bottom, **extra)


def page(rectangles, words):
    return SimpleNamespace(rects=rectangles, extract_words=lambda: words)


def test_distinguishes_adjacent_product_regions():
    source = page(
        [box(660.6, 94.8, 899.5, 262.8), box(903.5, 94.8, 1142.4, 262.8)],
        [box(750.6, 199.6, 809.4, 206.6, text="货号：192626164"),
         box(993.5, 199.6, 1052.3, 206.6, text="货号：192626155")],
    )
    assert image_product_owner(source, box(981.2, 117.7, 1027.9, 172.8)) == "192626155"
    assert image_product_owner(source, box(833.6, 122.3, 908.1, 182.2)) == "192626164"
    assert image_product_owner(source, box(818.1, 122.1, 882.7, 182.5)) == "192626164"


def test_ambiguous_region_has_no_owner():
    source = page([box(0, 0, 100, 100)],
                  [box(10, 60, 50, 65, text="款号：112621131"),
                   box(10, 70, 50, 75, text="款号：112621111")])
    assert image_product_owner(source, box(10, 10, 50, 50)) is None


def test_cannot_guess_without_enclosing_region():
    source = page([], [box(10, 60, 50, 65, text="货号：192626164")])
    assert image_product_owner(source, box(10, 10, 50, 50)) is None


def test_border_crossing_does_not_select_a_product():
    source = page([box(0, 0, 50, 100), box(50, 0, 100, 100)],
                  [box(5, 70, 45, 75, text="货号：192626164"),
                   box(55, 70, 95, 75, text="货号：192626155")])
    assert image_product_owner(source, box(40, 10, 60, 50)) is None


def test_invalid_coordinates_have_no_owner():
    assert image_product_owner(page([], []), box(0, 0, float("nan"), 1)) is None
    assert image_product_owner(page([], []), box(0, 0, 0, 1)) is None
