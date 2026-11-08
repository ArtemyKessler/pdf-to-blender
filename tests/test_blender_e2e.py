"""Builds the mesh in Blender. Runs with `blender` on PATH / $BLENDER, or with the bpy module."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bottle3d import blender_runner, pipeline
from bottle3d.spec.schema import BottleSpec, Job
from tests.conftest import has_blender

pytestmark = [
    pytest.mark.blender,
    pytest.mark.skipif(not has_blender(), reason="Blender not available"),
]
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "blender"


def _build(profile: Path, out: Path) -> dict:
    if importlib.util.find_spec("bpy"):
        code = (
            "import bpy, sys; sys.path.insert(0, sys.argv[1]); import build_bottle as b; "
            "b.build(sys.argv[2], sys.argv[3])"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(SCRIPT_DIR), str(profile), str(out)], check=True
        )
    else:
        blender_runner.build(profile, out, preview=False)
    return json.loads((out / "blender_stats.json").read_text())


def test_mesh_dimensions_manifold_and_volume(synthetic, tmp_path):
    spec = BottleSpec(
        overall_height_mm=246.0,
        body_diameter_mm=88.3,
        base_diameter_mm=73.0,
        glass_weight_g=500,
        neck_inner_diameter_mm=18.7,
    )
    r = pipeline.run(synthetic.path, Job(spec=spec), tmp_path)
    stats = _build(tmp_path / "profile.json", tmp_path)
    assert stats["height_mm"] == pytest.approx(246.0, abs=0.5)
    assert stats["diameter_mm"] == pytest.approx(88.3, abs=0.5)
    assert stats["non_manifold_edges"] == 0
    # Positive and equal to the glass volume: normals point outwards, cavity is hollow.
    assert stats["volume_cm3"] == pytest.approx(r.wall.v_glass, rel=0.01)
    assert (tmp_path / "model.blend").exists()


def test_mesh_with_punt_is_manifold(synthetic, tmp_path):
    spec = BottleSpec(
        overall_height_mm=246.0,
        body_diameter_mm=88.3,
        glass_weight_g=500,
        neck_inner_diameter_mm=18.7,
    )
    job = Job(spec=spec, punt_depth_mm=15, punt_diameter_mm=40)
    r = pipeline.run(synthetic.path, job, tmp_path)
    stats = _build(tmp_path / "profile.json", tmp_path)
    assert stats["non_manifold_edges"] == 0
    assert stats["volume_cm3"] == pytest.approx(r.wall.v_glass, rel=0.01)
