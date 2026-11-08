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
    # The stamp and dimensions are read from the drawing as soon as it is dropped.
    assert window.fields["glass_weight_g"].text() == "500"
    assert window.fields["body_diameter_mm"].text() == "88.3"
    assert window.colour.currentText() == "Flint"
    assert window.fields["glass_weight_g"].styleSheet()  # framed as auto-filled

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


def test_contour_is_shown_and_can_be_corrected(window, tmp_path):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(PDF.read_bytes())
    window.load_pdf(pdf)
    a = window.analysis
    assert a is not None and a.contour.height == pytest.approx(250.6, abs=0.5)
    assert "найдено автоматически: ось, вид" in window.contour_info.text().lower()
    assert window.width_box.currentData() == pytest.approx(0.84)

    # Frame the BOTTOM VIEW instead: the axis inside it is taken, the contour is rebuilt.
    window.on_view_picked((60.0, 20.0, 175.0, 140.0))
    assert window.job.view_bbox_mm == (60.0, 20.0, 175.0, 140.0)
    assert window.analysis.contour.height < 120
    assert "задано вручную: вид" in window.contour_info.text().lower()

    # A wrong axis gives an asymmetric outline and a warning.
    window.reset_geometry()
    window.on_axis_picked(125.0)
    assert window.job.axis_x_mm == 125.0
    assert "подозрительно" in window.contour_info.text().lower()

    # Reset brings back the automatic front view, and the dimensions read for it.
    window.reset_geometry()
    assert window.job.axis_x_mm == pytest.approx(117.2, abs=0.1)
    assert window.analysis.contour.height == pytest.approx(250.6, abs=0.5)
    assert window.fields["body_diameter_mm"].text() == "88.3"


def test_choosing_a_line_width_rebuilds_the_contour(window, tmp_path):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(PDF.read_bytes())
    window.load_pdf(pdf)
    i = next(
        i
        for i in range(window.width_box.count())
        if abs(window.width_box.itemData(i) - 0.72) < 1e-3
    )
    window.on_width_chosen(i)
    assert window.job.contour_width_pt == pytest.approx(0.72)
    assert "contour_width_pt" in window.manual_geo


def test_corrections_are_saved_next_to_the_drawing(window, tmp_path, monkeypatch):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(PDF.read_bytes())
    window.load_pdf(pdf)
    window.on_axis_picked(117.18)
    window.out = tmp_path / "out"
    window.out.mkdir()
    monkeypatch.setattr(
        "bottle3d.gui.main.QFileDialog.getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    window.save()
    window.load_pdf(pdf)
    assert window.job.axis_x_mm == 117.18
    assert "axis_x_mm" in window.manual_geo


def test_punt_is_filled_and_can_be_switched_off(window, tmp_path):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(PDF.read_bytes())
    window.load_pdf(pdf)
    assert window.punt_depth.value() == pytest.approx(20, abs=0.6)
    assert window.punt_dia.value() == pytest.approx(46, abs=2)
    assert "d4a017" in window.punt_depth.styleSheet()  # framed: found automatically
    assert not window.drawing._punt.path().isEmpty()

    window.punt_depth.setValue(0)
    assert window.manual_punt and window.job.punt_depth_mm == 0
    assert window.punt_depth.styleSheet() == ""
    assert window.drawing._punt.path().isEmpty()
    # A re-analysis (another line width) keeps what the person typed...
    window.on_width_chosen(window.width_box.currentIndex())
    assert window.punt_depth.value() == 0
    # ...and "Reset" goes back to what the drawing shows.
    window.reset_geometry()
    assert window.punt_depth.value() == pytest.approx(20, abs=0.6)
