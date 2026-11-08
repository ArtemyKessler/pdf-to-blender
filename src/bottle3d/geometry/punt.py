"""Punt (the dome pushed up into the base) — found as a dashed hidden line on the drawing.

Bottle drawings show the punt in a side view as a dashed dome above the base line. The dashes
are drawn as many short separate strokes, so the dome is recognised as a set of short pieces of
one line width that are mirror images of each other across the view's axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bottle3d.pdf.vectors import PageVectors, find_vertical_axes

FLAT_TOL = 1e-3  # mm: the traced base line is flat only to rounding


@dataclass
class PuntFound:
    depth: float  # sheet mm
    diameter: float  # sheet mm, where the dome meets the base line
    points: np.ndarray  # (r, z) samples of the dome, sheet mm
    axis_x: float


def punt_profile(depth: float, diameter: float, n: int = 48) -> np.ndarray:
    """(r, z) of the dome from the base edge (a, 0) to the top (0, depth).

    A quarter ellipse: steep at the base like real punts, round at the top.
    """
    a = diameter / 2
    t = np.linspace(0, np.pi / 2, n)
    return np.column_stack([a * np.cos(t), depth * np.sin(t)])


def detect_punt(
    vectors: PageVectors, contour_width: float, *, min_pairs: int = 6
) -> PuntFound | None:
    best: PuntFound | None = None
    best_pairs = 0
    for ax, ay0, ay1 in find_vertical_axes(vectors):
        # Base of this view: the lowest outline line that crosses the axis area.
        outline = [
            s
            for s in vectors.segments
            if abs(s.width_pt - contour_width) <= 0.03
            and s.bbox[0] < ax + 60
            and s.bbox[2] > ax - 60
            and ay0 - 5 <= s.bbox[1] <= ay1 + 5
        ]
        if not outline:
            continue
        base = min(s.bbox[1] for s in outline)
        width_r = max(abs(x - ax) for s in outline for x in (s.bbox[0], s.bbox[2]))

        pieces: dict[float, list[tuple[float, float]]] = {}
        for s in vectors.segments:
            if s.kind != "l" or abs(s.width_pt - contour_width) <= 0.03:
                continue
            (x1, y1), (x2, y2) = s.pts
            length = float(np.hypot(x2 - x1, y2 - y1))
            mx, my = (x1 + x2) / 2 - ax, (y1 + y2) / 2 - base
            if 0.6 <= length <= 2.5 and 0.3 < my < 40 and 0.5 < abs(mx) < 0.8 * width_r:
                pieces.setdefault(round(s.width_pt, 2), []).append((mx, my))

        for _w, pts in pieces.items():
            arr = np.array(pts)
            left, right = arr[arr[:, 0] < 0], arr[arr[:, 0] > 0]
            if not len(left) or not len(right):
                continue
            pairs = []
            for r, z in right:
                d = np.hypot(left[:, 0] + r, left[:, 1] - z)
                if d.min() < 0.3:
                    pairs.append((r, z))
            if len(pairs) < min_pairs or len(pairs) <= best_pairs:
                continue
            p = np.array(sorted(pairs, key=lambda t: -t[0]))
            # A dome: height grows as the radius shrinks.
            if np.corrcoef(p[:, 0], p[:, 1])[0, 1] > -0.8:
                continue
            depth = float(p[:, 1].max() + 0.3)
            # Extrapolate the base edge from the two lowest samples.
            low = p[np.argsort(p[:, 1])[:2]]
            if abs(low[1, 1] - low[0, 1]) > 1e-6:
                slope = (low[1, 0] - low[0, 0]) / (low[1, 1] - low[0, 1])
                a = float(low[0, 0] - slope * low[0, 1])
            else:
                a = float(low[:, 0].max())
            a = float(np.clip(a, p[:, 0].max(), 0.95 * width_r))
            best = PuntFound(depth, 2 * a, p, ax)
            best_pairs = len(pairs)
    return best


def apply_punt(rz: np.ndarray, depth: float, diameter: float) -> np.ndarray:
    """Outer half profile (real mm, from (0, 0) up the outside) with the base pushed up into
    a dome of `depth` and base `diameter`. The dome is kept inside the flat standing ring."""
    flat = rz[np.abs(rz[:, 1]) < FLAT_TOL]
    r_contact = float(flat[:, 0].max()) if len(flat) else float(rz[1, 0])
    a = min(diameter / 2, 0.95 * r_contact)
    dome = punt_profile(depth, 2 * a)[::-1]  # (0, depth) … (a, 0)
    rest = rz[(rz[:, 1] >= FLAT_TOL) | (rz[:, 0] > a + 1e-6)]
    return np.vstack([dome, rest])
