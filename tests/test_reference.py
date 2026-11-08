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
    # brimful 720 + glass 200; the punt takes ~20 ml off the outer volume.
    assert w.v_outer == pytest.approx(920, rel=0.01)
    assert w.v_below_fill == pytest.approx(700, rel=0.02)
    assert 2.0 < w.wall_t < 4.5


def test_punt_is_found_in_the_side_view(result):
    j = result.job
    # Dimensioned "20.0" on the drawing; the dashed dome sits in the LEFT VIEW.
    assert j.punt_depth_mm == pytest.approx(20.0, abs=0.6)
    assert j.punt_diameter_mm == pytest.approx(46, abs=2)
    assert "punt" in result.auto
    assert any("пунт найден" in n for n in result.notes)


def test_inner_base_follows_the_punt(result):
    s = result.wall.section
    # Inner bottom on the axis: punt depth plus the base thickness.
    assert s[0, 0] == 0
    assert s[0, 1] == pytest.approx(result.job.punt_depth_mm + 10, abs=0.05)


def test_flat_base_when_punt_is_switched_off(tmp_path):
    r = pipeline.run(PDF, Job(spec=SPEC, punt_depth_mm=0), tmp_path)
    assert "punt" not in r.auto
    assert r.wall.section[0, 1] == pytest.approx(10)
    assert r.wall.v_outer > 935  # the ~20 ml of the dome is back


def test_engravings_are_found_and_placed(result):
    by_side = {e.angle_deg: e for e in result.engravings}
    assert set(by_side) == {-90.0, 180.0}
    mono, text = by_side[-90.0], by_side[180.0]  # front view, left view
    # Developed outlines: monogram ~20 × 20 mm (5:1), "RISE & SHINE" ~60 × 9 mm (2:1).
    assert mono.size == pytest.approx((20.4, 20.5), abs=0.5)
    assert text.size == pytest.approx((61.6, 9.3), abs=0.5)
    assert len(text.shape.geoms) == 10  # letters
    # Both on the shoulder, 0.6 mm high.
    for e in (mono, text):
        assert 150 < e.z_center < 170
        assert e.relief_mm == 0.6
    assert sum("гравировка" in n and "мм" in n for n in result.notes) == 2


def test_engraving_can_be_switched_off(tmp_path):
    r = pipeline.run(PDF, Job(spec=SPEC, engraving=False), tmp_path)
    assert r.engravings == []
    assert any("не моделируется: гравировка" in n for n in r.notes)
