import math

import numpy as np
import pytest

from bottle3d.geometry.wall import outer_section, revolve_volume_cm3, solve_wall
from tests.synthetic.make_pdf import profile_polyline


def test_cylinder_volume_is_exact():
    rz = np.array([[0, 0], [40, 0], [40, 100], [0, 100]], float)
    assert revolve_volume_cm3(outer_section(rz)) == pytest.approx(
        math.pi * 40**2 * 100 / 1000, rel=1e-9
    )


def test_cone_volume_is_exact():
    rz = np.array([[0, 0], [30, 0], [0, 90]], float)
    assert revolve_volume_cm3(outer_section(rz)) == pytest.approx(
        math.pi * 30**2 * 90 / 3 / 1000, rel=1e-9
    )


def test_wall_matches_glass_weight():
    w = solve_wall(profile_polyline(), glass_weight_g=500, base_t=10, bore_d=18.7)
    assert w.v_glass == pytest.approx(200, rel=1e-3)
    assert 1.0 < w.wall_t < 8.0


def test_section_is_open_with_ends_on_axis():
    w = solve_wall(profile_polyline(), glass_weight_g=500, base_t=10, bore_d=18.7)
    s = w.section
    assert s[0, 0] == 0 and s[-1, 0] == 0
    assert s[0, 1] == pytest.approx(10)  # inner bottom, base thickness above z=0
    assert s[-1, 1] == pytest.approx(0)
    assert np.all(s[1:-1, 0] > 0)
    # Neck bore keeps the drawing's inner diameter.
    top = s[s[:, 1] > 240]
    assert top[:, 0].min() == pytest.approx(18.7 / 2, abs=1e-6)


def test_outer_volume_check_flags_wrong_contour():
    w = solve_wall(profile_polyline(), glass_weight_g=500, brimful_ml=500)
    assert any("внешний объём" in p for p in w.problems)
