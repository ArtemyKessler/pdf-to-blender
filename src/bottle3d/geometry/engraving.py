"""Raised engraving (logo, lettering) from the drawing.

Bottle drawings give an engraving twice:

* a *developed* (unrolled) outline, labelled "Developed engraving, Scale: 2:1" — the exact shape;
* its projection in a side view — where it sits on the bottle (height) and which side it faces.

The relief height comes from a note like "Engraving 0.6mm high".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import shapely
from shapely import STRtree
from shapely.affinity import scale as scale_geom
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union

from bottle3d.pdf.vectors import PT_TO_MM, PageVectors, Segment, find_vertical_axes
from bottle3d.spec.extract_rules import TextLine

NUM = r"(\d+(?:[.,]\d+)?)"


@dataclass
class Developed:
    shape: MultiPolygon  # real mm, centred on (0, 0); u around the bottle, v up the side
    scale: float  # drawing scale of the developed outline (2 for "2:1")
    width_pt: float
    label_xy: tuple[float, float]  # sheet mm


@dataclass
class Engraving:
    shape: MultiPolygon  # real mm, centred on (0, 0)
    z_center: float  # sheet mm above the base of the view it was found in
    view_axis_x: float
    angle_deg: float  # direction the engraving faces, 0 = +X, -90 = towards the camera (front)
    relief_mm: float

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.shape.bounds
        return x1 - x0, y1 - y0


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def relief_height(lines: list[TextLine], default: float = 0.6) -> float:
    for ln in lines:
        m = re.search(r"engra\w*\s+" + NUM + r"\s*mm", ln.text, flags=re.I)
        if m:
            return _f(m.group(1))
    return default


def _grow(
    segs: list[Segment],
    geoms: list[LineString],
    tree: STRtree,
    seed: list[int],
    width: float,
    gap: float,
    keep,
) -> set[int]:
    comp: set[int] = set()
    todo = list(seed)
    while todo:
        i = todo.pop()
        if i in comp:
            continue
        comp.add(i)
        for j in tree.query(geoms[i].buffer(gap)):
            j = int(j)
            if (
                j not in comp
                and round(segs[j].width_pt, 2) == width
                and keep(segs[j])
                and geoms[j].distance(geoms[i]) < gap
            ):
                todo.append(j)
    return comp


def _faces(lines: list[LineString]) -> MultiPolygon:
    """Filled regions of closed outlines, even-odd: a face inside another one is a hole."""
    faces = [f for f in polygonize(unary_union(lines)) if f.area > 1e-3]
    keep, holes = [], []
    filled = [Polygon(f.exterior) for f in faces]
    for i, f in enumerate(filled):
        p = faces[i].representative_point()
        depth = sum(1 for j, g in enumerate(filled) if j != i and g.contains(p))
        (keep if depth % 2 == 0 else holes).append(f)
    shape = unary_union(keep)
    if holes:
        shape = shape.difference(unary_union(holes))
    return shape if isinstance(shape, MultiPolygon) else MultiPolygon([shape])


def find_developed(vectors: PageVectors, lines: list[TextLine]) -> list[Developed]:
    segs = [s for s in vectors.segments if s.width_pt > 0 and not s.dashed]
    if not segs:
        return []
    geoms = [LineString(s.polyline()) for s in segs]
    tree = STRtree(geoms)
    out = []
    for lab in lines:
        if not re.search(r"developed\s+engrav", lab.text, flags=re.I):
            continue
        lx0, lx1 = lab.x0 * PT_TO_MM, lab.x1 * PT_TO_MM
        ly = vectors.height_mm - lab.y0 * PT_TO_MM  # top of the label, Y up
        seed = [int(i) for i in tree.query(box(lx0 - 30, ly, lx1 + 30, ly + 25))]
        if not seed:
            continue
        width = Counter(round(segs[i].width_pt, 2) for i in seed).most_common(1)[0][0]
        seed = [i for i in seed if round(segs[i].width_pt, 2) == width]

        def near(s: Segment, lx0=lx0, lx1=lx1, ly=ly) -> bool:
            b = s.bbox
            return b[1] >= ly - 1 and b[0] > lx0 - 80 and b[2] < lx1 + 80

        comp = _grow(segs, geoms, tree, seed, width, 5.0, near)
        shape = _faces([geoms[i] for i in comp])
        if shape.is_empty or shape.area < 5:
            continue
        # "Scale: 2:1" (or "~5:1") just under the label.
        scale = None
        for ln in lines:
            if 0 <= ln.y0 - lab.y0 < 3 * (lab.y1 - lab.y0) and abs(ln.x0 - lab.x0) < 40:
                m = re.search(r"scale\W*~?\s*" + NUM + r"\s*:\s*" + NUM, ln.text, flags=re.I)
                if m:
                    scale = _f(m.group(1)) / _f(m.group(2))
                    break
        x0, y0, x1, y1 = shape.bounds
        shape = translate(shape, -(x0 + x1) / 2, -(y0 + y1) / 2)
        if scale:
            shape = scale_geom(shape, 1 / scale, 1 / scale, origin=(0, 0))
        out.append(Developed(shape, scale or 0.0, width, (lx0, ly)))
    return out


def _view_base(vectors: PageVectors, ax: float, ay0: float, ay1: float, contour_width: float):
    outline = [
        s
        for s in vectors.segments
        if abs(s.width_pt - contour_width) <= 0.03
        and s.bbox[0] < ax + 60
        and s.bbox[2] > ax - 60
        and ay0 - 5 <= s.bbox[1] <= ay1 + 5
    ]
    if not outline:
        return None
    base = min(s.bbox[1] for s in outline)
    half = max(abs(x - ax) for s in outline for x in (s.bbox[0], s.bbox[2]))
    return base, half


def _bands(vectors: PageVectors, ax: float, ay0: float, ay1: float, width: float, half: float):
    """Bounding boxes (x0, y0, x1, y1, sheet mm) of groups of short lines of `width` near the
    axis, merged by height: candidate projections of an engraving."""
    cand = [
        s
        for s in vectors.segments
        if round(s.width_pt, 2) == width
        and not s.dashed
        and abs((s.bbox[0] + s.bbox[2]) / 2 - ax) < half
        and ay0 < s.bbox[1] < ay1
        and max(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]) < 15
    ]
    if not cand:
        return []
    geoms = [LineString(s.polyline()) for s in cand]
    tree = STRtree(geoms)
    seen: set[int] = set()
    boxes = []
    for k in range(len(cand)):
        if k in seen:
            continue
        comp = _grow(cand, geoms, tree, [k], width, 1.5, lambda s: True)
        seen |= comp
        if len(comp) >= 15:
            b = np.array([cand[i].bbox for i in comp])
            boxes.append([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])
    # Letters are separate groups: merge groups that share most of their height.
    boxes.sort(key=lambda b: b[1])
    merged: list[list[float]] = []
    for b in boxes:
        for m in merged:
            overlap = min(m[3], b[3]) - max(m[1], b[1])
            if overlap > 0.6 * min(m[3] - m[1], b[3] - b[1]):
                m[:] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
                break
        else:
            merged.append(list(b))
    return merged


def detect_engravings(
    vectors: PageVectors,
    lines: list[TextLine],
    contour_width: float,
    main_axis_x: float,
    main_base_y: float,
) -> list[Engraving]:
    """Engravings with their place on the bottle. The main view's base is passed in, because
    its axis is often shared with the bottom view drawn under it."""
    developed = find_developed(vectors, lines)
    if not developed:
        return []
    relief = relief_height(lines)
    views = []
    for ax, ay0, ay1 in find_vertical_axes(vectors):
        vb = _view_base(vectors, ax, ay0, ay1, contour_width)
        if vb is None:
            continue
        base, half = vb
        if abs(ax - main_axis_x) < 1:
            base = main_base_y
        views.append((ax, ay0, ay1, base, half))

    out = []
    for d in developed:
        dw, dh = d.shape.bounds[2] - d.shape.bounds[0], d.shape.bounds[3] - d.shape.bounds[1]
        best = None
        for ax, ay0, ay1, base, half in views:
            for x0, y0, x1, y1 in _bands(vectors, ax, ay0, ay1, d.width_pt, 0.9 * half):
                pw, ph = x1 - x0, y1 - y0
                if not d.scale:  # no scale text: match the shape's proportions only
                    ok = 0.5 < (pw / ph) / (dw / dh) < 1.5
                else:
                    ok = abs(ph / dh - 1) < 0.2 and 0.6 * dw < pw < 1.3 * dw
                centred = abs((x0 + x1) / 2 - ax) < 0.2 * pw + 1
                if not (ok and centred):
                    continue
                err = abs(ph / dh - 1) if d.scale else 0.0
                if best is None or err < best[0]:
                    best = (err, ax, (y0 + y1) / 2 - base, ph)
        if best is None:
            continue
        _, ax, zc, ph = best
        shape = d.shape
        if not d.scale:
            k = ph / dh
            shape = scale_geom(shape, k, k, origin=(0, 0))
        # First-angle projection (ISO, Europe): a view drawn right of the front view is seen
        # from the left (-X); the front view faces the camera (-Y).
        if abs(ax - main_axis_x) < 1:
            angle = -90.0
        elif ax > main_axis_x:
            angle = 180.0
        else:
            angle = 0.0
        out.append(Engraving(shape, zc, ax, angle, relief))
    return out


def flat_mesh(shape: MultiPolygon, cell: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Triangles covering `shape`, cut into `cell`-sized pieces so the mesh bends smoothly when
    wrapped onto the bottle. Returns (verts (n, 2), faces (m, 3)); shared vertices are merged."""
    x0, y0, x1, y1 = shape.bounds
    verts: dict[tuple[float, float], int] = {}
    faces = []

    def vid(p) -> int:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in verts:
            verts[key] = len(verts)
        return verts[key]

    for gx in np.arange(x0, x1, cell):
        for gy in np.arange(y0, y1, cell):
            piece = shape.intersection(box(gx, gy, gx + cell, gy + cell))
            if piece.is_empty or piece.area < 1e-8:
                continue
            for tri in shapely.constrained_delaunay_triangles(piece).geoms:
                if tri.area < 1e-10:
                    continue
                a, b, c = list(tri.exterior.coords)[:3]
                ids = [vid(a), vid(b), vid(c)]
                if len(set(ids)) < 3:
                    continue
                # Counter-clockwise, so the top faces point away from the bottle.
                area2 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
                faces.append(ids if area2 > 0 else ids[::-1])
    v = np.array(list(verts.keys()), dtype=float).reshape(-1, 2)
    return v, np.array(faces, dtype=int).reshape(-1, 3)
