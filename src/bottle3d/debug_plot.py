"""PNG of the drawing with the detected axis, view box and contour drawn over it."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from bottle3d.geometry.contour import Contour
from bottle3d.pdf.vectors import PageVectors


def overlay_png(
    pdf_path: str | Path,
    page_no: int,
    vectors: PageVectors,
    contour: Contour,
    view_bbox: tuple[float, float, float, float],
    out: str | Path,
    dpi: int = 200,
) -> None:
    m = vectors.to_page

    def pt(x_mm: float, y_mm: float) -> pymupdf.Point:
        return pymupdf.Point(x_mm, y_mm) * m

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no]
        shape = page.new_shape()
        x0, y0, x1, y1 = view_bbox
        shape.draw_polyline([pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1), pt(x0, y0)])
        shape.finish(color=(0, 0.4, 1), width=0.8, dashes="[4 2] 0")
        shape.draw_line(
            pt(contour.axis_x, contour.base_y - 5),
            pt(contour.axis_x, contour.base_y + contour.height + 5),
        )
        shape.finish(color=(0, 0.7, 0), width=0.8, closePath=False)
        xs = contour.axis_x + contour.rz[:, 0]
        ys = contour.base_y + contour.rz[:, 1]
        shape.draw_polyline([pt(x, y) for x, y in zip(xs, ys, strict=True)])
        shape.finish(color=(1, 0, 0), width=1.2, closePath=False)
        shape.commit()
        # Crop to the view with a margin; the whole A2 sheet would be too small to read.
        # get_pixmap() wants the clip in displayed (rotated) page coordinates.
        rot = page.rotation_matrix
        a, b = pt(x0 - 10, y0 - 10) * rot, pt(x1 + 10, y1 + 10) * rot
        clip = pymupdf.Rect(a, b).normalize() & page.rect
        pix = page.get_pixmap(dpi=dpi, clip=clip)
        pix.save(str(out))
