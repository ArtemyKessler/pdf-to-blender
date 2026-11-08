"""Vector extraction from a PDF page.

All geometry leaves this module in *sheet millimetres*: origin at the bottom-left corner of the
page as it is displayed (page rotation applied), X to the right, Y up.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

PT_TO_MM = 25.4 / 72.0


@dataclass(frozen=True)
class Segment:
    """A straight line (`kind="l"`, 2 points) or cubic Bezier (`kind="c"`, 4 points)."""

    kind: str
    pts: np.ndarray  # (n, 2) in sheet mm, Y up
    width_pt: float
    dashed: bool
    color: tuple[float, ...] | None

    def polyline(self, chord_mm: float = 0.02) -> np.ndarray:
        """Points along the segment; Beziers are sampled finely enough for `chord_mm`."""
        if self.kind == "l":
            return self.pts
        p0, p1, p2, p3 = self.pts
        # Bound on second derivative gives the number of samples for a given chord error.
        dd = max(np.linalg.norm(p0 - 2 * p1 + p2), np.linalg.norm(p1 - 2 * p2 + p3)) * 6
        n = int(np.clip(np.ceil(np.sqrt(dd / (8 * chord_mm))), 2, 400))
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        mt = 1 - t
        return mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        lo = self.pts.min(axis=0)
        hi = self.pts.max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


@dataclass
class PageVectors:
    segments: list[Segment]
    width_mm: float
    height_mm: float
    to_page: pymupdf.Matrix  # sheet mm (Y up) -> unrotated page points, for drawing overlays

    def width_histogram(self) -> list[tuple[float, int]]:
        """(line width in pt rounded to 0.01, number of segments), most frequent first."""
        c = Counter(round(s.width_pt, 2) for s in self.segments)
        return c.most_common()


def _is_dashed(dashes: str | None) -> bool:
    if not dashes:
        return False
    # PyMuPDF gives e.g. "[] 0" for solid lines and "[3 1 0.5 1] 0" for dashed ones.
    inner = dashes.split("]")[0].strip("[ ")
    return bool(inner)


def extract(pdf_path: str | Path, page_no: int = 0) -> PageVectors:
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no]
        # get_drawings() reports coordinates of the unrotated page; rotation_matrix maps them to
        # the page as displayed, page.rect is the displayed size.
        rot = page.rotation_matrix
        h_pt = page.rect.height
        flip = pymupdf.Matrix(PT_TO_MM, 0, 0, -PT_TO_MM, 0, h_pt * PT_TO_MM)
        to_mm = rot * flip

        def conv(*points: pymupdf.Point) -> np.ndarray:
            return np.array([[(p * to_mm).x, (p * to_mm).y] for p in points])

        segments: list[Segment] = []
        for path in page.get_drawings():
            width = path.get("width")
            if width is None:  # fill-only path (hatching, logos): no outline
                continue
            dashed = _is_dashed(path.get("dashes"))
            color = path.get("color")
            for item in path["items"]:
                op = item[0]
                if op == "l":
                    pts = [conv(item[1], item[2])]
                    kinds = ["l"]
                elif op == "c":
                    pts = [conv(item[1], item[2], item[3], item[4])]
                    kinds = ["c"]
                elif op == "re":
                    r = item[1]
                    q = [r.tl, r.tr, r.br, r.bl, r.tl]
                    pts = [conv(q[i], q[i + 1]) for i in range(4)]
                    kinds = ["l"] * 4
                elif op == "qu":
                    qd = item[1]
                    q = [qd.ul, qd.ur, qd.lr, qd.ll, qd.ul]
                    pts = [conv(q[i], q[i + 1]) for i in range(4)]
                    kinds = ["l"] * 4
                else:
                    continue
                for k, p in zip(kinds, pts, strict=True):
                    segments.append(Segment(k, p, float(width), dashed, color))

        return PageVectors(
            segments=segments,
            width_mm=page.rect.width * PT_TO_MM,
            height_mm=page.rect.height * PT_TO_MM,
            to_page=~to_mm,
        )


def find_vertical_axes(
    vectors: PageVectors, min_length_mm: float = 30.0, x_tol_mm: float = 0.2
) -> list[tuple[float, float, float]]:
    """Dashed vertical lines (or chains of short collinear dashes) — axis-of-symmetry candidates.

    Returns (x, y_min, y_max) sorted by length, longest first.
    """
    xs: list[tuple[float, float, float, bool]] = []
    for s in vectors.segments:
        if s.kind != "l":
            continue
        (x1, y1), (x2, y2) = s.pts
        if abs(x1 - x2) > x_tol_mm or abs(y1 - y2) < 1e-6:
            continue
        # A long solid line is a contour edge, not an axis; short pieces may be drawn dashes.
        if s.dashed or abs(y1 - y2) < 15.0:
            xs.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2), s.dashed))
    xs.sort()
    groups: list[list[tuple[float, float, float, bool]]] = []
    for item in xs:
        if groups and item[0] - groups[-1][-1][0] <= x_tol_mm:
            groups[-1].append(item)
        else:
            groups.append([item])
    out = []
    for g in groups:
        y0 = min(i[1] for i in g)
        y1 = max(i[2] for i in g)
        covered = sum(i[2] - i[1] for i in g)
        # Either a path with a dash pattern, or a chain of separate dashes that covers most of
        # its span (sheet-frame zone ticks at the top and bottom edges cover almost nothing).
        dashed = any(i[3] for i in g)
        chain = len(g) >= 3 and covered >= 0.5 * (y1 - y0)
        if y1 - y0 >= min_length_mm and (dashed or chain):
            out.append((float(np.mean([i[0] for i in g])), y0, y1))
    out.sort(key=lambda t: t[2] - t[1], reverse=True)
    return out
