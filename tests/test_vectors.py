import pytest

from bottle3d.pdf.vectors import extract, find_vertical_axes
from tests.synthetic.make_pdf import make


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_rotation_does_not_move_geometry(tmp_path, rotation):
    s = make(tmp_path / "b.pdf", rotation=rotation)
    v = extract(s.path)
    assert v.width_mm == pytest.approx(594, abs=0.1)
    assert v.height_mm == pytest.approx(420, abs=0.1)
    axes = find_vertical_axes(v)
    assert axes[0][0] == pytest.approx(117.2, abs=0.05)


def test_width_histogram_separates_contour_and_dimensions(synthetic):
    widths = dict(extract(synthetic.path).width_histogram())
    assert 0.84 in widths and 0.72 in widths
    assert widths[0.84] > widths[0.72]
