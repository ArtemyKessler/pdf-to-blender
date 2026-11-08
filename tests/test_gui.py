"""Smoke test of the desktop window: drop the reference PDF, fill the stamp, build."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from bottle3d import blender_runner  # noqa: E402
from bottle3d.gui.main import MainWindow, job_path_for  # noqa: E402
from tests.test_reference import PDF  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path):
    from PySide6.QtCore import QSettings

    w = MainWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    return w


def test_drop_fill_and_build_without_blender(window, qtbot, tmp_path, monkeypatch):
    def no_blender(*a, **k):
        raise FileNotFoundError("Blender 4.2+ не найден")

    monkeypatch.setattr(blender_runner, "build", no_blender)
    pdf = tmp_path / "Чертёж с пробелом.pdf"
    pdf.write_bytes(PDF.read_bytes())
    window.load_pdf(pdf)
    assert window.build_btn.isEnabled()

    for key, value in {
        "overall_height_mm": "246",
        "body_diameter_mm": "88,3",
        "base_diameter_mm": "73",
        "glass_weight_g": "500",
        "neck_inner_diameter_mm": "18.7",
    }.items():
        window.fields[key].setText(value)

    window.start()
    qtbot.waitUntil(
        lambda: window.save_btn.isEnabled() or "Ошибка" in window.status.text(), timeout=60_000
    )
    assert "Ошибка" not in window.status.text(), window.log.toPlainText()
    assert "Blender" in window.status.text()  # the missing Blender is reported, not fatal
    assert window.result is not None
    assert window.result.calibration.scale == pytest.approx(0.98, abs=0.01)
    assert window.job.axis_x_mm == pytest.approx(117.2, abs=0.1)  # auto value kept
    assert "Ø корпуса" in window.report.toPlainText()

    # Saving copies outputs and remembers the settings next to the drawing.
    out = tmp_path / "saved"
    out.mkdir()
    monkeypatch.setattr(
        "bottle3d.gui.main.QFileDialog.getExistingDirectory", lambda *a, **k: str(out)
    )
    window.save()
    assert (out / pdf.stem / "report.md").exists()
    assert job_path_for(pdf).exists()

    # Dropping the same drawing again restores what was typed.
    window.load_pdf(pdf)
    assert window.fields["body_diameter_mm"].text() == "88.3"


def test_bad_number_is_rejected(window, tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(PDF.read_bytes())
    window.load_pdf(pdf)
    window.fields["glass_weight_g"].setText("пятьсот")
    with pytest.raises(ValueError):
        window.read_form()
