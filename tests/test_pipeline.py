import json

import pytest

from bottle3d import pipeline
from bottle3d.spec.schema import BottleSpec, Job

SPEC = BottleSpec(
    overall_height_mm=246.0,
    body_diameter_mm=88.3,
    base_diameter_mm=73.0,
    glass_weight_g=500,
    neck_inner_diameter_mm=18.7,
)


def test_run_writes_outputs_and_job_roundtrip(synthetic, tmp_path):
    r1 = pipeline.run(synthetic.path, Job(spec=SPEC), tmp_path / "a")
    for name in ("profile.json", "report.json", "report.md", "debug.png", "job.yaml"):
        assert (tmp_path / "a" / name).exists()
    assert set(r1.auto) == {"axis_x_mm", "view_bbox_mm", "contour_width_pt"}
    assert r1.job.contour_width_pt == 0.84

    # Re-running from the saved job reproduces the same profile and guesses nothing.
    r2 = pipeline.run(synthetic.path, Job.load(tmp_path / "a" / "job.yaml"), tmp_path / "b")
    assert r2.auto == []
    a = json.loads((tmp_path / "a" / "profile.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "b" / "profile.json").read_text(encoding="utf-8"))
    assert a["section"] == b["section"]
    assert a["units"] == "m"
    assert max(z for _, z in a["section"]) == pytest.approx(0.246, abs=1e-4)


def test_cyrillic_paths(synthetic, tmp_path):
    out = tmp_path / "Результат с пробелом"
    pipeline.run(synthetic.path, Job(spec=SPEC), out)
    assert (out / "report.md").read_text(encoding="utf-8").startswith("#")


def test_job_overrides_profile_defaults():
    profile = Job(contour_width_pt=0.84, base_thickness_mm=12)
    job = Job.model_validate({"base_thickness_mm": 8, "spec": {"glass_weight_g": 450}})
    m = job.merged_with(profile)
    assert m.contour_width_pt == 0.84
    assert m.base_thickness_mm == 8
    assert m.spec.glass_weight_g == 450
