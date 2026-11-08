import numpy as np
import pytest
from shapely.geometry import LineString, Point

from bottle3d.geometry.calibrate import calibrate
from bottle3d.geometry.contour import extract_contour, select_segments
from bottle3d.pdf.vectors import extract
from tests.synthetic.make_pdf import profile_polyline

BBOX = (117.2 - 80, 40, 117.2 + 79, 360)


@pytest.fixture(scope="module")
def contour(synthetic):
    v = extract(synthetic.path)
    return extract_contour(select_segments(v, BBOX, 0.84), synthetic.axis_x_mm)


def test_contour_matches_ground_truth(contour, synthetic):
    truth = LineString(profile_polyline())
    dev = max(truth.distance(Point(p / synthetic.scale)) for p in contour.rz)
    assert dev < 0.05
    assert contour.max_gap_mm < 0.05
    assert contour.symmetry_mm < 0.05


def test_contour_starts_on_axis_and_ignores_inner_lines(contour):
    assert tuple(contour.rz[0]) == (0.0, 0.0)
    # Neck rings and the fill line are drawn inside the outline; none may stick out.
    assert np.all(contour.rz[:, 0] >= 0)


def test_calibration_recovers_scale(contour, synthetic):
    cal = calibrate(contour, height=246.0, body_d=88.3, base_d=73.0)
    assert cal.scale == pytest.approx(1 / synthetic.scale, rel=1e-4)
    assert not cal.problems


def test_calibration_flags_inconsistent_dimensions(contour):
    cal = calibrate(contour, height=246.0, body_d=92.0, base_d=None)
    assert cal.problems
