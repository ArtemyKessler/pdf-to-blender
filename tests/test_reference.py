"""The real reference drawing: AB "Panevėžio stiklas" B70-0089, whisky 700 ml Cork."""

from pathlib import Path

import pytest

from bottle3d import pipeline
from bottle3d.pdf.vectors import extract, find_vertical_axes
from bottle3d.spec.schema import BottleSpec, Job

PDF = Path(__file__).parent / "fixtures" / "BUTELIS_WISKEY_700ml_Cork.pdf"
SPEC = BottleSpec(
    overall_height_mm=246.0,
    body_diameter_mm=88.3,
    base_diameter_mm=73.0,
    glass_weight_g=500,
    brimful_capacity_ml=720,
    nominal_capacity_ml=700,
    fill_height_mm=53,
    neck_inner_diameter_mm=18.7,
    glass_colour="Flint",
)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return pipeline.run(PDF, Job(spec=SPEC), tmp_path_factory.mktemp("ref"))


def test_axis_is_found_and_frame_ticks_are_ignored():
    axes = find_vertical_axes(extract(PDF))
    assert axes[0][0] == pytest.approx(117.2, abs=0.1)


def test_auto_detection_picks_front_view(result):
    j = result.job
    assert j.axis_x_mm == pytest.approx(117.2, abs=0.1)
    assert j.contour_width_pt == pytest.approx(0.84, abs=0.01)
    # FRONT VIEW, not BOTTOM VIEW that shares the same axis below it.
    x0, y0, x1, y1 = j.view_bbox_mm
    assert y0 > 140 and y1 > 395


def test_contour_is_closed_and_symmetric(result):
    assert result.contour.max_gap_mm <= 0.1
    assert result.contour.symmetry_mm < 0.05


def test_dimensions_after_calibration(result):
    k = result.calibration.scale
    assert 0.975 < k < 0.99
    assert result.contour.height * k == pytest.approx(246.0, abs=0.5)
    assert result.contour.max_diameter * k == pytest.approx(88.3, abs=0.5)
    assert result.contour.base_diameter * k == pytest.approx(73.0, abs=0.5)
    assert not result.calibration.problems


def test_volumes_match_the_stamp(result):
    w = result.wall
    assert w.v_outer == pytest.approx(920, rel=0.03)  # brimful 720 + glass 200
    assert w.v_below_fill == pytest.approx(700, rel=0.02)
    assert 2.0 < w.wall_t < 4.5
