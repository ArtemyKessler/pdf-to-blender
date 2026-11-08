"""Inner cavity, wall thickness and volumes of a body of revolution.

Regions are kept as shapely polygons of the *full* axial section (both halves), so that the
inward offset never treats the axis as a wall. Volumes use Pappus: V = 2π · A · x̄ over the
right half.

What the stamp determines and what it does not: weight and brimful capacity only check the
outer volume (V_outer ≈ brimful + V_glass, whatever the wall looks like). How glass is split
between the side wall and the base is an assumption, so base thickness is an explicit input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

BIG = 1e4


def outer_section(rz: np.ndarray) -> Polygon:
    """Full section from the outer half profile (0,0) → … → (r_top, H); rim top is flat."""
    h = float(rz[:, 1].max())
    right = [(float(r), float(z)) for r, z in rz]
    if right[-1][0] > 1e-9:
        right.append((0.0, h))
    left = [(-r, z) for r, z in reversed(right[1:-1])]
    poly = Polygon(right + left)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def revolve_volume_cm3(region) -> float:
    half = region.intersection(box(0, -BIG, BIG, BIG))
    if half.is_empty:
        return 0.0
    return 2 * math.pi * half.area * half.centroid.x / 1000.0


def _half_width(region, z: float) -> float:
    cut = region.intersection(LineString([(-BIG, z), (BIG, z)]))
    if cut.is_empty:
        return 0.0
    return max(abs(x) for x in (cut.bounds[0], cut.bounds[2]))


def cavity(outer: Polygon, t: float, base_t: float, bore_r: float | None) -> Polygon:
    """Inside of the bottle for side wall `t`, base `base_t` and neck bore radius `bore_r`.

    The offset cavity is used from the base up to the height where it first narrows to the
    bore; above that the neck is a straight bore up through the rim.
    """
    _, _, _, h = outer.bounds
    off = outer.buffer(-t, join_style="mitre").intersection(box(-BIG, base_t, BIG, BIG))
    if off.is_empty:
        return Polygon()
    if bore_r is None:
        r_top = _half_width(outer, h - 1e-6)
        bore_r = max(r_top - t, 0.5)

    # Walk up from the widest part of the cavity to where it narrows to the bore.
    zs = np.linspace(base_t, h, 600)
    widths = np.array([_half_width(off, z) for z in zs])
    start = int(np.argmax(widths))
    z_cut = h
    for z, w in zip(zs[start:], widths[start:], strict=True):
        if w <= bore_r:
            z_cut = float(z)
            break
    lower = off.intersection(box(-BIG, -BIG, BIG, z_cut))
    bore = box(-bore_r, min(z_cut, h) - 0.5, bore_r, h + 1)
    cav = unary_union([lower, bore]).intersection(outer)
    if cav.geom_type != "Polygon":  # keep the piece connected to the bore
        cav = max(cav.geoms, key=lambda g: g.intersects(Point(0, h - 0.1)) * 1e9 + g.area)
    return cav


@dataclass
class WallResult:
    wall_t: float | None
    base_t: float
    bore_r: float | None
    v_outer: float
    v_glass_target: float | None
    v_glass: float
    v_cavity: float
    v_below_fill: float | None
    section: np.ndarray  # open polyline (r, z) mm, both ends on the axis, for revolving
    problems: list[str] = field(default_factory=list)


def section_polyline(glass) -> np.ndarray:
    """Right-half boundary of the glass, opened at the axis: inner bottom → … → outer bottom."""
    half = glass.intersection(box(0, -BIG, BIG, BIG))
    if half.geom_type != "Polygon":
        half = max(half.geoms, key=lambda g: g.area)
    ring = np.asarray(half.exterior.coords)[:-1]
    on_axis = np.abs(ring[:, 0]) < 1e-7
    n = len(ring)
    # Find the edge (i, i+1) that lies on the axis; the polyline starts right after the run.
    axis_edges = [i for i in range(n) if on_axis[i] and on_axis[(i + 1) % n]]
    if not axis_edges:
        raise ValueError("glass section does not touch the axis at the base")
    # The run's last edge: one whose successor edge is not on the axis.
    last = next(i for i in axis_edges if (i + 1) % n not in axis_edges)
    first = next(i for i in axis_edges if (i - 1) % n not in axis_edges)
    start = (last + 1) % n
    end = first
    idx = [(start + k) % n for k in range((end - start) % n + 1)]
    poly = ring[idx].copy()
    poly[np.abs(poly[:, 0]) < 1e-7, 0] = 0.0
    # Orientation: go from the inner bottom (higher z on the axis) to the outer bottom (z=0).
    if poly[0, 1] < poly[-1, 1]:
        poly = poly[::-1]
    return poly


def solve_wall(
    outer_rz: np.ndarray,
    *,
    glass_weight_g: float | None,
    density: float = 2.5,
    base_t: float = 10.0,
    bore_d: float | None = None,
    brimful_ml: float | None = None,
    nominal_ml: float | None = None,
    fill_height_mm: float | None = None,
    default_t: float = 3.0,
) -> WallResult:
    outer = outer_section(outer_rz)
    h = outer.bounds[3]
    bore_r = bore_d / 2 if bore_d else None
    v_outer = revolve_volume_cm3(outer)
    problems: list[str] = []
    target = glass_weight_g / density if glass_weight_g else None

    def glass_v(t: float) -> float:
        return v_outer - revolve_volume_cm3(cavity(outer, t, base_t, bore_r))

    if target is None:
        t = default_t
        problems.append(f"вес стекла неизвестен: толщина стенки взята по умолчанию {t} мм")
    else:
        lo, hi = 0.2, 20.0
        if glass_v(lo) > target:
            t = lo
            problems.append(
                f"даже при стенке {lo} мм стекла больше веса ({glass_v(lo):.0f} > "
                f"{target:.0f} см³): уменьшите толщину дна"
            )
        elif glass_v(hi) < target:
            t = hi
            problems.append("стекла по весу больше, чем помещается в контур")
        else:
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if glass_v(mid) < target else (lo, mid)
            t = (lo + hi) / 2
        if not 1.0 <= t <= 8.0:
            problems.append(f"толщина стенки {t:.2f} мм вне разумного диапазона 1–8 мм")

    cav = cavity(outer, t, base_t, bore_r)
    glass = outer.difference(cav)
    v_cav = revolve_volume_cm3(cav)
    v_glass = v_outer - v_cav

    if brimful_ml and target:
        expected = brimful_ml + target
        rel = v_outer / expected - 1
        if abs(rel) > 0.03:
            problems.append(
                f"внешний объём {v_outer:.0f} см³, а по штампу должно быть ≈ {expected:.0f} "
                f"(полная ёмкость + стекло), расхождение {rel * 100:+.1f} %: проверьте контур "
                "и масштаб"
            )
    v_fill = None
    if fill_height_mm and nominal_ml:
        v_fill = revolve_volume_cm3(cav.intersection(box(-BIG, -BIG, BIG, h - fill_height_mm)))
        rel = v_fill / nominal_ml - 1
        if abs(rel) > 0.02:
            problems.append(
                f"объём до уровня налива {v_fill:.0f} мл против номинала {nominal_ml:.0f} "
                f"({rel * 100:+.1f} %): возможно, у дна есть пунт (вогнутость), который пока не "
                "моделируется, или внутренний профиль не сходится с чертежом"
            )

    return WallResult(
        wall_t=t,
        base_t=base_t,
        bore_r=bore_r,
        v_outer=v_outer,
        v_glass_target=target,
        v_glass=v_glass,
        v_cavity=v_cav,
        v_below_fill=v_fill,
        section=section_polyline(glass),
        problems=problems,
    )
