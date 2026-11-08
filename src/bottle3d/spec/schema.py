"""Data shapes shared by the pipeline, the CLI, the future GUI and the Blender script."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BottleSpec(BaseModel):
    """What the drawing says about the bottle. Every field may be unknown."""

    name: str | None = None
    drawing_no: str | None = None
    manufacturer: str | None = None
    nominal_capacity_ml: float | None = None
    brimful_capacity_ml: float | None = None
    glass_weight_g: float | None = None
    glass_colour: str | None = None
    finish: str | None = None
    fill_height_mm: float | None = Field(None, description="fill level, measured from the top")
    overall_height_mm: float | None = None
    body_diameter_mm: float | None = None
    base_diameter_mm: float | None = None
    neck_inner_diameter_mm: float | None = None


class Job(BaseModel):
    """Everything a person decided about one drawing. Re-running with the same job gives the
    same result. Fields left empty are detected automatically (or fail with a clear message)."""

    page: int = 0
    axis_x_mm: float | None = Field(None, description="axis of the main view, sheet mm from left")
    view_bbox_mm: tuple[float, float, float, float] | None = Field(
        None, description="x0, y0, x1, y1 of the main view, sheet mm, Y up from the bottom edge"
    )
    contour_width_pt: float | None = None
    base_thickness_mm: float = Field(
        10.0, description="glass under the cavity; the drawing's stamp does not determine it"
    )
    glass_density_g_cm3: float = 2.5
    spec: BottleSpec = Field(default_factory=BottleSpec)

    @classmethod
    def load(cls, path: str | Path) -> Job:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)

    def merged_with(self, defaults: Job) -> Job:
        """Fields set in this job win over `defaults` (e.g. a manufacturer profile)."""
        base = defaults.model_dump(exclude_unset=True)
        mine = self.model_dump(exclude_unset=True)
        spec = {**base.get("spec", {}), **mine.get("spec", {})}
        return Job.model_validate({**base, **mine, "spec": spec})
