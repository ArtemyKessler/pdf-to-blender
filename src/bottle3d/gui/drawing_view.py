"""The drawing with the detected axis, view box and contour, where a person can correct them.

Coordinates: the scene is the rendered page in pixels; the pipeline works in sheet mm with Y up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from bottle3d.geometry.contour import Contour
from bottle3d.geometry.punt import punt_profile

DPI = 110
PX_PER_MM = DPI / 25.4


class DrawingView(QGraphicsView):
    """Zoom with the wheel, pan by dragging. Modes: "axis" — click sets the axis;
    "view" — drag a rectangle around the main view."""

    axis_picked = Signal(float)  # x in sheet mm
    view_picked = Signal(tuple)  # (x0, y0, x1, y1) in sheet mm, Y up

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#3a3a3a"))
        self.page_h_mm = 0.0
        self.mode: str | None = None
        self._page: QGraphicsPixmapItem | None = None
        self._axis = self._line(QColor(0, 170, 0), 2.0)
        self._box = self._rect(QColor(0, 110, 255))
        self._contour = QGraphicsPathItem()
        self._contour.setPen(self._pen(QColor(230, 0, 0), 2.5))
        self._contour.setZValue(3)
        self.scene().addItem(self._contour)
        self._punt = QGraphicsPathItem()
        self._punt.setPen(self._pen(QColor(255, 140, 0), 2.0, dashed=True))
        self._punt.setZValue(3)
        self.scene().addItem(self._punt)
        self._rubber: QGraphicsRectItem | None = None
        self._drag_start: QPointF | None = None
        self._fitted = False

    # ---- items ---------------------------------------------------------------------------

    @staticmethod
    def _pen(color: QColor, width: float, dashed: bool = False) -> QPen:
        pen = QPen(color, width)
        pen.setCosmetic(True)  # same thickness at any zoom
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _line(self, color: QColor, width: float) -> QGraphicsLineItem:
        item = QGraphicsLineItem()
        item.setPen(self._pen(color, width))
        item.setZValue(2)
        self.scene().addItem(item)
        return item

    def _rect(self, color: QColor) -> QGraphicsRectItem:
        item = QGraphicsRectItem()
        item.setPen(self._pen(color, 1.5, dashed=True))
        item.setZValue(2)
        self.scene().addItem(item)
        return item

    # ---- coordinates ---------------------------------------------------------------------

    def to_scene(self, x_mm: float, y_mm: float) -> QPointF:
        return QPointF(x_mm * PX_PER_MM, (self.page_h_mm - y_mm) * PX_PER_MM)

    def to_mm(self, p: QPointF) -> tuple[float, float]:
        return p.x() / PX_PER_MM, self.page_h_mm - p.y() / PX_PER_MM

    # ---- content -------------------------------------------------------------------------

    def load_page(self, pdf: Path, page_no: int = 0) -> None:
        with pymupdf.open(pdf) as doc:
            page = doc[page_no]
            self.page_h_mm = page.rect.height * 25.4 / 72
            pix = page.get_pixmap(dpi=DPI, alpha=False)
            img = QImage(
                pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(img.copy())
        if self._page is None:
            self._page = self.scene().addPixmap(pixmap)
            self._page.setZValue(0)
        else:
            self._page.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self._fitted = False

    def show_analysis(
        self,
        axis_x: float | None,
        bbox: tuple | None,
        contour: Contour | None,
        punt: tuple[float, float] | None = None,
    ) -> None:
        """`punt`: (depth, diameter) in sheet mm, drawn as a dashed dome over the contour base."""
        if bbox:
            x0, y0, x1, y1 = bbox
            self._box.setRect(QRectF(self.to_scene(x0, y1), self.to_scene(x1, y0)))
            self._box.show()
        else:
            self._box.hide()
        if axis_x is not None:
            top = bbox[3] + 5 if bbox else self.page_h_mm
            bottom = bbox[1] - 5 if bbox else 0
            a, b = self.to_scene(axis_x, bottom), self.to_scene(axis_x, top)
            self._axis.setLine(a.x(), a.y(), b.x(), b.y())
            self._axis.show()
        else:
            self._axis.hide()
        path = QPainterPath()
        if contour is not None and len(contour.rz):
            pts = contour.rz + np.array([contour.axis_x, contour.base_y])
            path.moveTo(self.to_scene(*pts[0]))
            for p in pts[1:]:
                path.lineTo(self.to_scene(*p))
        self._contour.setPath(path)
        punt_path = QPainterPath()
        if contour is not None and punt and punt[0] > 0 and punt[1] > 0:
            half = punt_profile(*punt)
            pts = np.vstack([half * [-1, 1], half[::-1]]) + [contour.axis_x, contour.base_y]
            punt_path.moveTo(self.to_scene(*pts[0]))
            for p in pts[1:]:
                punt_path.lineTo(self.to_scene(*p))
        self._punt.setPath(punt_path)
        if bbox and not self._fitted:
            self.zoom_to(bbox)
            self._fitted = True

    def zoom_to(self, bbox: tuple) -> None:
        x0, y0, x1, y1 = bbox
        r = QRectF(self.to_scene(x0 - 10, y1 + 10), self.to_scene(x1 + 10, y0 - 10))
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)

    def fit_page(self) -> None:
        if self._page is not None:
            self.fitInView(self._page, Qt.AspectRatioMode.KeepAspectRatio)

    # ---- interaction ---------------------------------------------------------------------

    def set_mode(self, mode: str | None) -> None:
        self.mode = mode
        if mode is None:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def wheelEvent(self, e) -> None:  # noqa: N802
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.mode == "axis":
            x, _ = self.to_mm(self.mapToScene(e.position().toPoint()))
            self.set_mode(None)
            self.axis_picked.emit(round(x, 3))
            return
        if e.button() == Qt.MouseButton.LeftButton and self.mode == "view":
            self._drag_start = self.mapToScene(e.position().toPoint())
            self._rubber = QGraphicsRectItem()
            self._rubber.setPen(self._pen(QColor(255, 140, 0), 1.5, dashed=True))
            self._rubber.setZValue(4)
            self.scene().addItem(self._rubber)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._rubber is not None and self._drag_start is not None:
            p = self.mapToScene(e.position().toPoint())
            self._rubber.setRect(QRectF(self._drag_start, p).normalized())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._rubber is not None and self._drag_start is not None:
            p = self.mapToScene(e.position().toPoint())
            self.scene().removeItem(self._rubber)
            self._rubber = None
            start, self._drag_start = self._drag_start, None
            self.set_mode(None)
            r = QRectF(start, p).normalized()
            if r.width() > 10 and r.height() > 10:
                x0, y1 = self.to_mm(r.topLeft())
                x1, y0 = self.to_mm(r.bottomRight())
                self.view_picked.emit(tuple(round(v, 2) for v in (x0, y0, x1, y1)))
            return
        super().mouseReleaseEvent(e)
