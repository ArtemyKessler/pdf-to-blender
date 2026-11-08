"""PDF + Job → profile.json, report, debug overlay. The CLI and the GUI both call `run`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from bottle3d import __version__
from bottle3d.debug_plot import overlay_png
from bottle3d.geometry.calibrate import Calibration, calibrate
from bottle3d.geometry.contour import Contour, extract_contour, select_segments
from bottle3d.geometry.wall import WallResult, solve_wall
from bottle3d.pdf.vectors import PageVectors, extract, find_vertical_axes
from bottle3d.spec.schema import Job


@dataclass
class Result:
    job: Job  # with auto-detected values filled in
    contour: Contour
    calibration: Calibration
    wall: WallResult
    auto: list[str] = field(default_factory=list)  # which job fields were guessed
    problems: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "needs_review" if self.problems else "ok"


def _guess_contour_width(vectors: PageVectors, bbox) -> float:
    x0, y0, x1, y1 = bbox
    counts: dict[float, int] = {}
    for s in vectors.segments:
        bx0, by0, bx1, by1 = s.bbox
        if s.dashed or not (bx0 >= x0 and bx1 <= x1 and by0 >= y0 and by1 <= y1):
            continue
        w = round(s.width_pt, 2)
        counts[w] = counts.get(w, 0) + 1
    if not counts:
        raise ValueError("в выбранной области нет линий")
    top = max(counts.values())
    # The outline is the thickest line style that is used a lot.
    return max(w for w, c in counts.items() if c >= 0.1 * top)


def resolve_job(vectors: PageVectors, job: Job) -> tuple[Job, list[str]]:
    """Fill in axis, view box and contour width the job leaves empty. Guesses are suggestions:
    they are listed so that the report (and the GUI) can ask a person to confirm them."""
    auto: list[str] = []
    j = job.model_copy(deep=True)
    if j.axis_x_mm is None:
        axes = find_vertical_axes(vectors)
        if not axes:
            raise ValueError("не найдена ось симметрии: задайте axis_x_mm в job.yaml")
        j.axis_x_mm = round(axes[0][0], 3)
        auto.append("axis_x_mm")
    if j.view_bbox_mm is None:
        axes = find_vertical_axes(vectors)
        mine = min(axes, key=lambda a: abs(a[0] - j.axis_x_mm))
        others = [abs(a[0] - j.axis_x_mm) for a in axes if a is not mine]
        half_w = min([*others, 300.0]) / 2
        j.view_bbox_mm = tuple(
            round(v, 2)
            for v in (j.axis_x_mm - half_w, mine[1] - 5, j.axis_x_mm + half_w, mine[2] + 5)
        )
        auto.append("view_bbox_mm")
    if j.contour_width_pt is None:
        j.contour_width_pt = _guess_contour_width(vectors, j.view_bbox_mm)
        auto.append("contour_width_pt")
    return j, auto


def run(pdf: str | Path, job: Job, out_dir: str | Path) -> Result:
    pdf = Path(pdf)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vectors = extract(pdf, job.page)
    if not vectors.segments:
        raise ValueError("в PDF нет векторной графики: похоже на скан")
    job, auto = resolve_job(vectors, job)

    segs = select_segments(vectors, job.view_bbox_mm, job.contour_width_pt)
    contour = extract_contour(segs, job.axis_x_mm)

    sp = job.spec
    cal = calibrate(
        contour,
        height=sp.overall_height_mm,
        body_d=sp.body_diameter_mm,
        base_d=sp.base_diameter_mm,
    )
    k = cal.scale
    wall = solve_wall(
        contour.rz * k,
        glass_weight_g=sp.glass_weight_g,
        density=job.glass_density_g_cm3,
        base_t=job.base_thickness_mm,
        bore_d=sp.neck_inner_diameter_mm,
        brimful_ml=sp.brimful_capacity_ml,
        nominal_ml=sp.nominal_capacity_ml,
        fill_height_mm=sp.fill_height_mm,
    )

    problems = [*cal.problems, *wall.problems]
    if contour.max_gap_mm > 0.05:
        problems.append(f"разрыв в контуре {contour.max_gap_mm:.2f} мм")
    if contour.symmetry_mm == contour.symmetry_mm and contour.symmetry_mm > 0.3:
        problems.append(
            f"левая и правая половины вида расходятся в среднем на {contour.symmetry_mm:.2f} мм:"
            " тара может быть неосесимметричной"
        )
    if auto:
        problems.append("определено автоматически, проверьте по картинке: " + ", ".join(auto))
    problems.append("дно плоское (пунт не моделируется)")
    result = Result(job, contour, cal, wall, auto, problems)

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
        "meta": {
            "source_pdf": pdf.name,
            "pdf_sha256": _sha256(pdf),
            "tool_version": __version__,
            "scale": r.calibration.scale,
            "name": r.job.spec.name,
        },
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


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
