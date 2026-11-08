"""PDF + Job → profile.json, report, debug overlay. The CLI and the GUI both call `run`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bottle3d import __version__
from bottle3d.debug_plot import overlay_png
from bottle3d.geometry.calibrate import Calibration, calibrate
from bottle3d.geometry.contour import Contour, extract_contour, in_bbox, select_segments
from bottle3d.geometry.engraving import Engraving, detect_engravings, flat_mesh
from bottle3d.geometry.punt import FLAT_TOL, apply_punt, detect_punt
from bottle3d.geometry.wall import WallResult, solve_wall
from bottle3d.pdf.vectors import PageVectors, extract, find_vertical_axes
from bottle3d.spec.extract_rules import read_dimensions, read_lines, read_stamp
from bottle3d.spec.schema import Job


@dataclass
class Result:
    job: Job  # with auto-detected values filled in
    contour: Contour
    calibration: Calibration
    wall: WallResult
    auto: list[str] = field(default_factory=list)  # which job fields were guessed
    problems: list[str] = field(default_factory=list)  # measurable mismatches → needs_review
    notes: list[str] = field(default_factory=list)  # always-true limitations and reminders
    engravings: list[Engraving] = field(default_factory=list)  # z_center in real mm

    @property
    def status(self) -> str:
        return "needs_review" if self.problems else "ok"


def _seg_len(s) -> float:
    p = s.polyline()
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _guess_contour_width(vectors: PageVectors, bbox) -> float:
    """The thickest line style that carries a good share of the drawn length in the view.

    Counting segments would pick hatching and engraving (thousands of tiny thin pieces);
    length favours long outline curves.
    """
    lengths: dict[float, float] = {}
    for s in vectors.segments:
        if s.dashed or s.width_pt <= 0 or not in_bbox(s, bbox):
            continue
        w = round(s.width_pt, 2)
        lengths[w] = lengths.get(w, 0.0) + _seg_len(s)
    if not lengths:
        raise ValueError("в выбранной области нет линий")
    top = max(lengths.values())
    return max(w for w, length in lengths.items() if length >= 0.2 * top)


def _view_bbox(vectors: PageVectors, axis, half_w: float) -> tuple[float, ...]:
    """Box of the tallest group of lines around the axis.

    Views often share an axis (FRONT above BOTTOM VIEW), so the axis span alone is not a view.
    Lines that cross the axis area are grouped by height; a gap of more than 8 mm splits views.
    """
    x, ay0, ay1 = axis
    box = (x - half_w, ay0 - 5, x + half_w, ay1 + 5)
    width = _guess_contour_width(vectors, box)
    spans = sorted(
        (s.bbox[1], s.bbox[3])
        for s in vectors.segments
        if not s.dashed and abs(s.width_pt - width) <= 0.03 and in_bbox(s, box)
    )
    groups: list[list[float]] = []
    for y0, y1 in spans:
        if groups and y0 <= groups[-1][1] + 8.0:
            groups[-1][1] = max(groups[-1][1], y1)
        else:
            groups.append([y0, y1])
    y0, y1 = max(groups, key=lambda g: g[1] - g[0])
    return tuple(round(v, 2) for v in (x - half_w, y0 - 3, x + half_w, y1 + 3))


def resolve_job(vectors: PageVectors, job: Job) -> tuple[Job, list[str]]:
    """Fill in axis, view box and contour width the job leaves empty. Guesses are suggestions:
    they are listed so that the report (and the GUI) can ask a person to confirm them."""
    auto: list[str] = []
    j = job.model_copy(deep=True)
    axes = find_vertical_axes(vectors)
    if j.axis_x_mm is None:
        if j.view_bbox_mm is not None:  # the person framed a view: its own axis
            x0, _, x1, _ = j.view_bbox_mm
            axes = [a for a in axes if x0 < a[0] < x1] or axes
            axes.sort(key=lambda a: abs(a[0] - (x0 + x1) / 2))
        if not axes:
            raise ValueError("не найдена ось симметрии: укажите ось вручную")
        j.axis_x_mm = round(axes[0][0], 3)
        auto.append("axis_x_mm")
    if j.view_bbox_mm is None:
        if axes:
            mine = min(axes, key=lambda a: abs(a[0] - j.axis_x_mm))
        else:
            mine = (j.axis_x_mm, 0.0, vectors.height_mm)
        others = [abs(a[0] - j.axis_x_mm) for a in axes if a is not mine]
        half_w = min([o for o in others if o > 20] + [300.0]) / 2
        j.view_bbox_mm = _view_bbox(vectors, mine, half_w)
        auto.append("view_bbox_mm")
    if j.contour_width_pt is None:
        j.contour_width_pt = _guess_contour_width(vectors, j.view_bbox_mm)
        auto.append("contour_width_pt")
    return j, auto


@dataclass
class Analysis:
    """Everything known about a drawing before building: the GUI shows it and lets a person
    correct axis, view and line width; `run` builds from it."""

    job: Job
    auto: list[str]
    vectors: PageVectors
    contour: Contour


def analyze(
    pdf: str | Path,
    job: Job,
    *,
    vectors: PageVectors | None = None,
    read_spec: bool = True,
) -> Analysis:
    """Vectors → axis/view/width (guessed where empty) → contour → spec read from the text.

    Pass `vectors` from a previous call to skip re-reading the PDF (interactive corrections).
    """
    pdf = Path(pdf)
    if vectors is None:
        vectors = extract(pdf, job.page)
    if not vectors.segments:
        raise ValueError("в PDF нет векторной графики: похоже на скан")
    job, auto = resolve_job(vectors, job)
    segs = select_segments(vectors, job.view_bbox_mm, job.contour_width_pt)
    contour = extract_contour(segs, job.axis_x_mm)

    if read_spec:
        lines = read_lines(pdf, job.page)
        found: dict[str, object] = {**read_stamp(lines)}
        found.update(read_dimensions(lines, contour, job.view_bbox_mm, vectors.height_mm))
        # What the person typed (set fields) always wins over what was read from the drawing.
        missing = {k: v for k, v in found.items() if getattr(job.spec, k) is None}
        if missing:
            job = job.model_copy(update={"spec": job.spec.model_copy(update=missing)})
            auto += [f"spec.{k}" for k in missing]
    if job.punt_depth_mm is None:
        found_punt = detect_punt(vectors, job.contour_width_pt)
        if found_punt is not None:
            # Real size: the scale is not calibrated yet, use the height ratio when known.
            h = job.spec.overall_height_mm
            k = h / contour.height if h else 1.0
            job = job.model_copy(
                update={
                    "punt_depth_mm": round(found_punt.depth * k, 1),
                    "punt_diameter_mm": round(found_punt.diameter * k, 1),
                }
            )
            auto.append("punt")

    return Analysis(job, auto, vectors, contour)


def line_widths(vectors: PageVectors, bbox) -> list[tuple[float, float]]:
    """(line width in pt, total drawn length in mm) inside the box, longest first."""
    lengths: dict[float, float] = {}
    for s in vectors.segments:
        if s.dashed or s.width_pt <= 0 or not in_bbox(s, bbox):
            continue
        w = round(s.width_pt, 2)
        lengths[w] = lengths.get(w, 0.0) + _seg_len(s)
    return sorted(lengths.items(), key=lambda t: -t[1])


def prefill(pdf: str | Path, job: Job | None = None) -> tuple[Job, list[str]]:
    """Job with everything that can be read from the drawing filled in, and what was filled."""
    a = analyze(pdf, job or Job())
    return a.job, a.auto


def run(pdf: str | Path, job: Job, out_dir: str | Path) -> Result:
    pdf = Path(pdf)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = analyze(pdf, job)
    job, auto, vectors, contour = a.job, a.auto, a.vectors, a.contour

    sp = job.spec
    cal = calibrate(
        contour,
        height=sp.overall_height_mm,
        body_d=sp.body_diameter_mm,
        base_d=sp.base_diameter_mm,
    )
    k = cal.scale
    outer_rz = contour.rz * k
    bottom_rz = None
    if job.punt_depth_mm and job.punt_diameter_mm:
        outer_rz = apply_punt(outer_rz, job.punt_depth_mm, job.punt_diameter_mm)
        last_flat = int(np.nonzero(np.abs(outer_rz[:, 1]) < FLAT_TOL)[0].max())
        bottom_rz = outer_rz[: last_flat + 1]
    wall = solve_wall(
        outer_rz,
        bottom_rz=bottom_rz,
        glass_weight_g=sp.glass_weight_g,
        density=job.glass_density_g_cm3,
        base_t=job.base_thickness_mm,
        bore_d=sp.neck_inner_diameter_mm,
        brimful_ml=sp.brimful_capacity_ml,
        nominal_ml=sp.nominal_capacity_ml,
        fill_height_mm=sp.fill_height_mm,
    )

    engravings = []
    if job.engraving:
        found = detect_engravings(
            vectors, read_lines(pdf, job.page), job.contour_width_pt, job.axis_x_mm, contour.base_y
        )
        for e in found:
            e.z_center *= k
        engravings = found

    problems = [*cal.problems, *wall.problems]
    if contour.max_gap_mm > 0.1:
        problems.append(f"разрыв в контуре {contour.max_gap_mm:.2f} мм")
    if contour.symmetry_mm == contour.symmetry_mm and contour.symmetry_mm > 0.3:
        problems.append(
            f"левая и правая половины вида расходятся в среднем на {contour.symmetry_mm:.2f} мм:"
            " тара может быть неосесимметричной"
        )
    notes = []
    if auto:
        geo = [a for a in auto if not a.startswith("spec.")]
        spec = [a.removeprefix("spec.") for a in auto if a.startswith("spec.")]
        if geo:
            notes.append(
                "определено автоматически, проверьте контур на картинке: " + ", ".join(geo)
            )
        if spec:
            notes.append("прочитано с чертежа, сверьте со штампом: " + ", ".join(spec))
    if job.punt_depth_mm and job.punt_diameter_mm:
        how = "найден на чертеже" if "punt" in auto else "задан"
        notes.append(
            f"пунт {how}: глубина {job.punt_depth_mm:.1f} мм, Ø {job.punt_diameter_mm:.1f} мм"
        )
    else:
        notes.append("дно плоское: пунт на чертеже не найден и не задан")
    side = {-90.0: "спереди", 180.0: "слева", 0.0: "справа"}
    for e in engravings:
        w, h = e.size
        notes.append(
            f"гравировка {w:.0f}×{h:.0f} мм, выпуклость {e.relief_mm:g} мм, "
            f"{side.get(e.angle_deg, '')} на высоте {e.z_center:.0f} мм"
        )
    notes.append(
        "не моделируется: " + ("" if engravings else "гравировка, ") + "пупырышки, dot code"
    )
    result = Result(job, contour, cal, wall, auto, problems, notes, engravings)

    _write_profile(result, pdf, out / "profile.json")
    _write_report(result, pdf, out)
    overlay_png(pdf, job.page, vectors, contour, job.view_bbox_mm, out / "debug.png")
    job.save(out / "job.yaml")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_profile(r: Result, pdf: Path, path: Path) -> None:
    data = {
        "version": 1,
        "units": "m",
        "section": [[round(x / 1000, 7), round(z / 1000, 7)] for x, z in r.wall.section],
        "material": {"glass_colour": r.job.spec.glass_colour or "Flint"},
        # Outer side of the bottle (r, z), for wrapping engravings onto it.
        "outer_side": [
            [round(x / 1000, 7), round(z / 1000, 7)]
            for x, z in r.contour.rz[1:] * r.calibration.scale
        ],
        "engravings": [_engraving_json(e) for e in r.engravings],
        "meta": {
            "source_pdf": pdf.name,
            "pdf_sha256": _sha256(pdf),
            "tool_version": __version__,
            "scale": r.calibration.scale,
            "name": r.job.spec.name,
        },
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def _engraving_json(e: Engraving) -> dict:
    verts, faces = flat_mesh(e.shape)
    return {
        "verts": np.round(verts / 1000, 8).tolist(),  # (u, v) m: around the bottle, up its side
        "faces": faces.tolist(),
        "angle_deg": e.angle_deg,
        "z_center": round(e.z_center / 1000, 7),
        "relief": e.relief_mm / 1000,
    }


def _rows(r: Result) -> list[tuple[str, float | None, float]]:
    k = r.calibration.scale
    sp = r.job.spec
    return [
        ("Общая высота, мм", sp.overall_height_mm, r.contour.height * k),
        ("Ø корпуса, мм", sp.body_diameter_mm, r.contour.max_diameter * k),
        ("Ø дна, мм", sp.base_diameter_mm, r.contour.base_diameter * k),
        ("Стекло, см³ (по весу)", r.wall.v_glass_target, r.wall.v_glass),
        (
            "Внешний объём, см³ (≈ ёмкость + стекло)",
            (sp.brimful_capacity_ml + r.wall.v_glass_target)
            if sp.brimful_capacity_ml and r.wall.v_glass_target
            else None,
            r.wall.v_outer,
        ),
        ("Объём до налива, мл", sp.nominal_capacity_ml, r.wall.v_below_fill or float("nan")),
    ]


def _write_report(r: Result, pdf: Path, out: Path) -> None:
    rows = _rows(r)
    data = {
        "status": r.status,
        "source_pdf": pdf.name,
        "scale": r.calibration.scale,
        "scale_x": r.calibration.scale_x,
        "scale_z": r.calibration.scale_z,
        "contour_gap_mm": r.contour.max_gap_mm,
        "symmetry_mm": r.contour.symmetry_mm,
        "wall_thickness_mm": r.wall.wall_t,
        "base_thickness_mm": r.wall.base_t,
        "volumes_cm3": {
            "outer": r.wall.v_outer,
            "glass": r.wall.v_glass,
            "cavity": r.wall.v_cavity,
            "below_fill": r.wall.v_below_fill,
        },
        "checks": [{"name": n, "drawing": d, "model": m} for n, d, m in rows],
        "auto_detected": r.auto,
        "problems": r.problems,
        "notes": r.notes,
    }
    (out / "report.json").write_text(
        json.dumps(_nan_to_none(data), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# {r.job.spec.name or pdf.stem}",
        "",
        f"**Статус: {r.status}**",
        "",
        "| Параметр | По чертежу | В модели |",
        "|---|---:|---:|",
    ]
    for name, d, m in rows:
        ds = "—" if d is None else f"{d:.1f}"
        ms = "—" if m != m else f"{m:.1f}"
        lines.append(f"| {name} | {ds} | {ms} |")
    lines += [
        "",
        f"- Масштаб: {r.calibration.scale:.4f} (X: {_f(r.calibration.scale_x)}, "
        f"Z: {_f(r.calibration.scale_z)})",
        f"- Толщина стенки: {r.wall.wall_t:.2f} мм, дна: {r.wall.base_t:.1f} мм (задано)",
        f"- Разрыв контура: {r.contour.max_gap_mm:.3f} мм, "
        f"асимметрия половин: {r.contour.symmetry_mm:.3f} мм",
        "",
        "## Требует внимания" if r.problems else "",
        *[f"- {p}" for p in r.problems],
        "",
        "## Заметки",
        *[f"- {n}" for n in r.notes],
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _f(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _nan_to_none(o):
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_nan_to_none(v) for v in o]
    return o
