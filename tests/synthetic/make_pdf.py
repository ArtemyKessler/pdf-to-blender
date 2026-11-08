"""Synthetic bottle drawings with known geometry — the ground truth for contour and calibration.

A drawing has: an outline of contour line width (both halves), a dash-dot axis, thinner
dimension lines with numbers, a few "noise" lines inside the outline (neck rings, fill line),
and optional page rotation and drawing scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

MM = 72.0 / 25.4

# Half profile of a 246 mm whisky-like bottle in real mm, (r, z) from the base centre upward.
# Each entry is a line ("l", end) or cubic Bezier ("c", c1, c2, end), starting at (0, 0).
DEFAULT_PROFILE: list[tuple] = [
    ("l", (36.5, 0.0)),
    ("c", (41.0, 0.0), (44.15, 3.0), (44.15, 8.0)),  # heel
    ("l", (44.15, 150.0)),
    ("c", (44.15, 185.0), (13.75, 175.0), (13.75, 205.0)),  # shoulder
    ("l", (13.75, 228.0)),
    ("l", (15.5, 230.0)),  # bead
    ("l", (15.5, 240.0)),
    ("l", (14.5, 246.0)),
    ("l", (0.0, 246.0)),  # rim top up to the axis
]


@dataclass
class Synthetic:
    path: Path
    axis_x_mm: float  # on the displayed sheet, Y-up frame
    base_y_mm: float
    scale: float  # sheet mm per real mm
    height: float = 246.0
    body_d: float = 88.3
    base_d: float = 73.0
    profile: list[tuple] = field(default_factory=lambda: DEFAULT_PROFILE)


def profile_polyline(profile: list[tuple] = DEFAULT_PROFILE, n: int = 200) -> np.ndarray:
    """Dense (r, z) polyline of the ground-truth profile."""
    pts = [np.array([0.0, 0.0])]
    for seg in profile:
        p0 = pts[-1]
        if seg[0] == "l":
            pts.append(np.array(seg[1], float))
        else:
            c1, c2, p3 = (np.array(p, float) for p in seg[1:])
            t = np.linspace(0, 1, n + 1)[1:, None]
            mt = 1 - t
            curve = mt**3 * p0 + 3 * mt**2 * t * c1 + 3 * mt * t**2 * c2 + t**3 * p3
            pts.extend(curve)
    return np.array(pts)


def make(
    out: str | Path,
    *,
    scale: float = 1.0187,
    rotation: int = 0,
    axis_x_mm: float = 117.2,
    base_y_mm: float = 80.0,
    contour_w: float = 0.84,
    dim_w: float = 0.72,
    page_mm: tuple[float, float] = (594.0, 420.0),
    profile: list[tuple] = DEFAULT_PROFILE,
    second_view_dx_mm: float | None = 160.0,
) -> Synthetic:
    """Write a synthetic drawing. Coordinates are given for the page *as displayed*."""
    out = Path(out)
    w_mm, h_mm = page_mm
    doc = pymupdf.open()
    # For rotation 90/270 the unrotated page has swapped sides; we draw in displayed coordinates
    # and map them back through the inverse of the rotation matrix.
    if rotation in (90, 270):
        page = doc.new_page(width=h_mm * MM, height=w_mm * MM)
    else:
        page = doc.new_page(width=w_mm * MM, height=h_mm * MM)
    page.set_rotation(rotation)
    derot = page.derotation_matrix  # displayed -> unrotated

    def P(x_mm: float, y_up_mm: float) -> pymupdf.Point:
        return pymupdf.Point(x_mm * MM, (h_mm - y_up_mm) * MM) * derot

    def to_sheet(ax: float, r: float, z: float, side: int) -> pymupdf.Point:
        return P(ax + side * r * scale, base_y_mm + z * scale)

    shape = page.new_shape()

    def outline(ax: float) -> None:
        for side in (1, -1):
            cur = (0.0, 0.0)
            for seg in profile:
                if seg[0] == "l":
                    shape.draw_line(to_sheet(ax, *cur, side), to_sheet(ax, *seg[1], side))
                    cur = seg[1]
                else:
                    c1, c2, p3 = seg[1:]
                    shape.draw_bezier(
                        to_sheet(ax, *cur, side),
                        to_sheet(ax, *c1, side),
                        to_sheet(ax, *c2, side),
                        to_sheet(ax, *p3, side),
                    )
                    cur = p3
        # Noise inside the outline with contour width: neck rings and a fill line.
        for z, r in ((230.0, 15.5), (240.0, 15.5), (198.0, 14.5)):
            shape.draw_line(to_sheet(ax, r, z, -1), to_sheet(ax, r, z, 1))
        shape.finish(width=contour_w, color=(0, 0, 0))

        # Dash-dot axis, longer than the bottle.
        shape.draw_line(to_sheet(ax, 0, -8, 1), to_sheet(ax, 0, 256, 1))
        shape.finish(width=0.35, color=(0, 0, 0), dashes="[12 2 2 2] 0")

    outline(axis_x_mm)
    if second_view_dx_mm:
        outline(axis_x_mm + second_view_dx_mm)

    # Dimension lines with numbers (thin) — height on the left, body diameter below.
    dx = axis_x_mm - 60 * scale
    shape.draw_line(P(dx, base_y_mm), P(dx, base_y_mm + 246 * scale))
    shape.draw_line(
        P(axis_x_mm - 44.15 * scale, base_y_mm - 15),
        P(axis_x_mm + 44.15 * scale, base_y_mm - 15),
    )
    shape.finish(width=dim_w, color=(0, 0, 0))
    page.insert_text(P(dx - 6, base_y_mm + 120 * scale), "246.0±1.6", fontsize=9)
    page.insert_text(P(axis_x_mm - 10, base_y_mm - 18), "Ø88.3±1.6", fontsize=9)

    # A frame around the sheet, as real drawings have.
    shape.draw_rect(pymupdf.Rect(P(10, 10), P(w_mm - 10, h_mm - 10)))
    shape.finish(width=0.7, color=(0, 0, 0))
    shape.commit()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()
    return Synthetic(out, axis_x_mm, base_y_mm, scale, profile=profile)
