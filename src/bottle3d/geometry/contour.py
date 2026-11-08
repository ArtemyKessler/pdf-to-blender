"""Outer half-profile of a body of revolution, taken from one view of the drawing."""

from __future__ import annotations

import heapq
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
        """Diameter at the bottom of the body, where the heel rounding ends.

        That is how bottle drawings dimension "base diameter" (e.g. Ø73.0 on the reference
        sheet sits 5.7 mm above the bottom, at the heel's tangent point), not the contact ring.
        Walking up from the bottom, it is the first point where the outline is almost vertical.
        """
        rz = self.rz[1:]  # skip the axis point
        d = np.diff(rz, axis=0)
        steep = (np.abs(d[:, 0]) <= 0.15 * np.abs(d[:, 1]) + 1e-9) & (d[:, 1] > 0)
        # A straight body wall is one long edge: its lower end is exactly the tangent point.
        long_ = steep & (np.hypot(d[:, 0], d[:, 1]) >= 5.0)
        if long_.any():
            i = int(np.argmax(long_))
        else:  # curved body: the first almost vertical step is close enough
            i = int(np.argmax(steep)) if steep.any() else 0
        return float(2 * rz[i, 0])


def in_bbox(s: Segment, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = s.bbox
    return bx0 >= x0 and bx1 <= x1 and by0 >= y0 and by1 <= y1


def select_segments(
    vectors: PageVectors,
    view_bbox: tuple[float, float, float, float],
    width_pt: float,
    width_tol_pt: float = 0.03,
    bridge: bool = True,
) -> list[Segment]:
    """Contour-width lines of the view, plus thinner lines that close gaps between them.

    Drafters do not always keep one line style: on the reference sheet the flat bottom and one
    concave neck arc are drawn thin. Such pieces are found as the shortest chain of other
    lines between two loose ends of the contour.
    """
    main, other = [], []
    for s in vectors.segments:
        if s.dashed or not in_bbox(s, view_bbox):
            continue
        (main if abs(s.width_pt - width_pt) <= width_tol_pt else other).append(s)
    if bridge and main and other:
        main = main + bridge_gaps(main, other)
    return main


def _ends(s: Segment) -> tuple[np.ndarray, np.ndarray]:
    return s.pts[0], s.pts[-1]


class _Snap:
    """Merges points closer than `tol` into one node id."""

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.cells: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = {}
        self.n = 0

    def id(self, p: np.ndarray) -> int:
        cx, cy = int(np.floor(p[0] / self.tol)), int(np.floor(p[1] / self.tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q, i in self.cells.get((cx + dx, cy + dy), []):
                    if np.hypot(*(p - q)) <= self.tol:
                        return i
        self.cells.setdefault((cx, cy), []).append((p, self.n))
        self.n += 1
        return self.n - 1


def _length(s: Segment) -> float:
    p = s.polyline()
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def bridge_gaps(
    main: list[Segment], other: list[Segment], tol: float = 0.1, max_detour: float = 1.5
) -> list[Segment]:
    snap = _Snap(tol)
    degree: dict[int, int] = {}
    pos: dict[int, np.ndarray] = {}
    for s in main:
        for p in _ends(s):
            i = snap.id(p)
            degree[i] = degree.get(i, 0) + 1
            pos[i] = p
    loose = {i for i, d in degree.items() if d == 1}
    if len(loose) < 2:
        return []

    adj: dict[int, list[tuple[int, float, int]]] = {}
    for k, s in enumerate(other):
        a, b = (snap.id(p) for p in _ends(s))
        if a == b:
            continue
        w = _length(s)
        adj.setdefault(a, []).append((b, w, k))
        adj.setdefault(b, []).append((a, w, k))

    added: set[int] = set()
    used: set[int] = set()
    for start in sorted(loose):
        if start in used or start not in adj:
            continue
        # Dijkstra over the thin lines until another loose end of the contour is reached.
        dist = {start: 0.0}
        prev: dict[int, tuple[int, int]] = {}
        heap = [(0.0, start)]
        goal = None
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            if u != start and u in loose and u not in used:
                goal = u
                break
            for v, w, k in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, np.inf) and nd <= 200.0:
                    dist[v] = nd
                    prev[v] = (u, k)
                    heapq.heappush(heap, (nd, v))
        if goal is None:
            continue
        straight = float(np.hypot(*(pos[goal] - pos[start])))
        if dist[goal] > max_detour * straight + 1.0:
            continue
        u = goal
        while u != start:
            u, k = prev[u]
            added.add(k)
        used.update((start, goal))
    return [other[k] for k in sorted(added)]


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
