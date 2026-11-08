from bottle3d import pipeline
from bottle3d.spec.schema import BottleSpec, Job
from tests.test_reference import PDF


def test_reference_stamp_and_dimensions_are_read():
    job, auto = pipeline.prefill(PDF)
    sp = job.spec
    assert sp.name == 'BUTELIS "WISKEY" 700ml -Cork'
    assert sp.drawing_no == "B70-0089"
    assert sp.nominal_capacity_ml == 700
    assert sp.brimful_capacity_ml == 720
    assert sp.glass_weight_g == 500
    assert sp.glass_colour == "Flint"
    assert sp.fill_height_mm == 53
    assert sp.neck_inner_diameter_mm == 18.7
    assert sp.overall_height_mm == 246.0
    assert sp.body_diameter_mm == 88.3
    assert sp.base_diameter_mm == 73.0
    assert "spec.glass_weight_g" in auto


def test_typed_values_win_over_the_drawing():
    job, auto = pipeline.prefill(PDF, Job(spec=BottleSpec(glass_weight_g=520)))
    assert job.spec.glass_weight_g == 520
    assert "spec.glass_weight_g" not in auto


def test_run_without_any_typed_data_reaches_ok(tmp_path):
    r = pipeline.run(PDF, Job(), tmp_path)
    assert r.status == "ok", r.problems
    assert "прочитано с чертежа" in (tmp_path / "report.md").read_text(encoding="utf-8")
