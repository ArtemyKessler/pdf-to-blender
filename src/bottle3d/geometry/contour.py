"""Outer half-profile of a body of revolution, taken from one view of the drawing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bottle3d.pdf.vectors import PageVectors, Segment


@dataclass
class Contour:
    """Outer half profile in sheet mm, relative to the axis: r = x - axis_x, z = y - z_base.

    `rz` runs from (0, 0) on the axis at the base, up the outside, to (r_top, height).
    The flat rim top (from r_top to the axis) is implied.
    """

    rz: np.ndarray
    axis_x: float
    base_y: float
    max_gap_mm: float
    symmetry_mm: float  # mean |r_right(z) - r_left(z)|; ~0 for a body of revolution

    @property
    def height(self) -> float:
        return float(self.rz[:, 1].max())

    @property
    def max_diameter(self) -> float:
        return float(2 * self.rz[:, 0].max())

    @property
    def base_diameter(self) -> float:
        """Diameter of the flat standing surface (the bottom line of the outline)."""
        bottom = self.rz[self.rz[:, 1] <= 1e-9]
        return float(2 * bottom[:, 0].max())


def select_segments(
    vectors: PageVectors,
    view_bbox: tuple[float, float, float, float],
    width_pt: float,
    width_tol_pt: float = 0.03,
) -> list[Segment]:
    x0, y0, x1, y1 = view_bbox
    out = []
    for s in vectors.segments:
        if s.dashed or abs(s.width_pt - width_pt) > width_tol_pt:
            continue
        bx0, by0, bx1, by1 = s.bbox
        if bx0 >= x0 and bx1 <= x1 and by0 >= y0 and by1 <= y1:
            out.append(s)
    return out


def _edges(segments: list[Segment]) -> np.ndarray:
    """All polyline edges as an (n, 4) array of x1, y1, x2, y2."""
    parts = []
    for s in segments:
        p = s.polyline()
        parts.append(np.hstack([p[:-1], p[1:]]))
    return np.vstack(parts) if parts else np.zeros((0, 4))


def _envelope(edges: np.ndarray, zs: np.ndarray, side: int, axis_x: float) -> np.ndarray:
    """For each height in `zs`, the largest distance from the axis on one side (NaN if none)."""
    x1, y1, x2, y2 = edges.T
    lo = np.minimum(y1, y2)
    hi = np.maximum(y1, y2)
    out = np.full(zs.shape, np.nan)
    dy = y2 - y1
    for i, z in enumerate(zs):
        m = (lo <= z) & (hi >= z)
        if not m.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dy[m]) > 1e-12, (z - y1[m]) / dy[m], 0.0)
        xa = x1[m] + t * (x2[m] - x1[m])
        # A horizontal edge at exactly this height contributes both of its ends.
        flat = np.abs(dy[m]) <= 1e-12
        xs = np.concatenate([xa, x1[m][flat], x2[m][flat]])
        r = side * (xs - axis_x)
        r = r[r >= -1e-6]
        if r.size:
            out[i] = r.max()
    return out


def extract_contour(segments: list[Segment], axis_x: float, step_mm: float = 0.05) -> Contour:
    """Outer envelope r(z) of the contour lines on the right of the axis.

    For every height the outermost line wins, so everything drawn inside the outline (neck
    rings, fill line, hidden lines, engraving) is ignored.
    """
    edges = _edges(segments)
    if not len(edges):
        raise ValueError("no contour lines in the selected view")
    right = edges[(np.maximum(edges[:, 0], edges[:, 2]) >= axis_x - 0.05)]
    y_min = float(right[:, [1, 3]].min())
    y_max = float(right[:, [1, 3]].max())

    # Sample heights: a regular grid plus every vertex height, so corners are kept exactly.
    grid = np.arange(y_min, y_max, step_mm)
    verts = np.concatenate([right[:, 1], right[:, 3]])
    zs = np.unique(np.round(np.concatenate([grid, verts, [y_max]]), 6))

    r_right = _envelope(right, zs, +1, axis_x)
    left = edges[(np.minimum(edges[:, 0], edges[:, 2]) <= axis_x + 0.05)]
    r_left = _envelope(left, zs, -1, axis_x) if len(left) else np.full(zs.shape, np.nan)

    ok = ~np.isnan(r_right)
    gaps = np.diff(zs[ok])
    max_gap = float(gaps.max() - step_mm) if gaps.size else 0.0
    both = ok & ~np.isnan(r_left)
    sym = float(np.mean(np.abs(r_right[both] - r_left[both]))) if both.any() else float("nan")

    z = zs[ok] - y_min
    r = r_right[ok]
    rz = np.column_stack([r, z])
    # Start on the axis at the base; the bottom line itself is the first sample.
    rz = np.vstack([[0.0, 0.0], rz])
    rz = _simplify(rz, 0.005)
    return Contour(rz, axis_x, y_min, max(max_gap, 0.0), sym)


def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Drop points that lie on a straight line between their neighbours (within `tol`)."""
    from shapely.geometry import LineString

    return np.asarray(LineString(pts).simplify(tol, preserve_topology=False).coords)
