"""Drawing scale from known dimensions. The scale printed on the view is never trusted."""

from __future__ import annotations

from dataclasses import dataclass, field

from bottle3d.geometry.contour import Contour


@dataclass
class Calibration:
    scale_x: float | None  # real mm per sheet mm, from diameters
    scale_z: float | None  # from heights
    residuals: dict[str, float] = field(default_factory=dict)  # relative error per dimension
    problems: list[str] = field(default_factory=list)

    @property
    def scale(self) -> float:
        """Single isotropic scale used for the model."""
        vals = [s for s in (self.scale_x, self.scale_z) if s is not None]
        if not vals:
            raise ValueError("no reference dimensions: cannot calibrate")
        return sum(vals) / len(vals)


def calibrate(
    contour: Contour,
    *,
    height: float | None,
    body_d: float | None,
    base_d: float | None,
    tol: float = 0.005,
) -> Calibration:
    """Least-squares scale separately along X (diameters) and Z (height).

    `tol` is the allowed relative disagreement between the two scales and for each residual.
    """
    xs = [
        (name, contour_v, spec_v)
        for name, contour_v, spec_v in (
            ("body_diameter", contour.max_diameter, body_d),
            ("base_diameter", contour.base_diameter, base_d),
        )
        if spec_v
    ]
    scale_x = sum(m * s for _, m, s in xs) / sum(m * m for _, m, _ in xs) if xs else None
    scale_z = height / contour.height if height else None
    cal = Calibration(scale_x, scale_z)

    if scale_x and scale_z and abs(scale_x / scale_z - 1) > tol:
        cal.problems.append(
            f"масштабы по X ({scale_x:.4f}) и по Z ({scale_z:.4f}) расходятся "
            f"на {abs(scale_x / scale_z - 1) * 100:.2f} %"
        )
    k = cal.scale
    checks = [*xs]
    if height:
        checks.append(("overall_height", contour.height, height))
    for name, measured, spec in checks:
        rel = measured * k / spec - 1
        cal.residuals[name] = rel
        if abs(rel) > tol:
            cal.problems.append(
                f"{name}: по чертежу {spec:.1f}, в модели {measured * k:.1f} мм "
                f"({rel * 100:+.2f} %)"
            )
    return cal
