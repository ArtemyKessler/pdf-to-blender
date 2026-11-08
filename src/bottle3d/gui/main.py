"""Desktop window: drop a PDF, check the drawing data, get a contour overlay, report and render."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from bottle3d import __version__, blender_runner, pipeline
from bottle3d.spec.schema import BottleSpec, Job

# (field, label) for the numeric part of the stamp shown in the form.
SPEC_FIELDS: list[tuple[str, str]] = [
    ("overall_height_mm", "Общая высота, мм"),
    ("body_diameter_mm", "Ø корпуса, мм"),
    ("base_diameter_mm", "Ø дна (над пяткой), мм"),
    ("glass_weight_g", "Вес стекла, г"),
    ("brimful_capacity_ml", "Полная ёмкость, мл"),
    ("nominal_capacity_ml", "Номинал, мл"),
    ("fill_height_mm", "Налив от верха, мм"),
    ("neck_inner_diameter_mm", "Внутр. Ø горла, мм"),
]
COLOURS = [
    "Flint",
    "Extra Flint",
    "Half White",
    "Green",
    "Antique Green",
    "UVAG",
    "Amber",
    "Brown",
    "Cobalt",
    "Black",
]
OUTPUTS = [
    "model.blend",
    "model.glb",
    "preview.png",
    "debug.png",
    "report.md",
    "report.json",
    "profile.json",
    "job.yaml",
]


def job_path_for(pdf: Path) -> Path:
    """Settings saved next to a drawing: `drawing.job.yaml`."""
    return pdf.with_suffix(".job.yaml")


class ImageView(QScrollArea):
    """A picture scaled to fit the panel."""

    def __init__(self, placeholder: str) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self._label = QLabel(placeholder)
        self._label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self._label.setWordWrap(True)
        self.setWidget(self._label)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder

    def show_file(self, path: Path | None) -> None:
        if path and path.exists():
            self._pixmap = QPixmap(str(path))
        else:
            self._pixmap = None
            self._label.setPixmap(QPixmap())
            self._label.setText(self._placeholder)
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        size = self.viewport().size()
        self._label.setPixmap(
            self._pixmap.scaled(
                max(size.width() - 4, 50),
                max(size.height() - 4, 50),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class Worker(QObject):
    """Runs the pipeline and Blender off the UI thread."""

    stage = Signal(str)
    done = Signal(object, str)  # Result, warning text ("" if Blender ran fine)
    failed = Signal(str, str)  # short message, details

    def __init__(
        self, pdf: Path, job: Job, out: Path, blender: str | None, engine: str, samples: int
    ) -> None:
        super().__init__()
        self.pdf, self.job, self.out = pdf, job, out
        self.blender, self.engine, self.samples = blender, engine, samples
        self.cancel = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            self.stage.emit("Разбираю чертёж…")
            result = pipeline.run(self.pdf, self.job, self.out)
        except Exception as e:  # noqa: BLE001 — every failure is shown to the person
            self.failed.emit(str(e), traceback.format_exc())
            return
        warning = ""
        try:
            self.stage.emit("Строю модель и рендер в Blender (это может занять минуту)…")
            blender_runner.build(
                self.out / "profile.json",
                self.out,
                blender=self.blender,
                glb=True,
                engine=self.engine,
                samples=self.samples,
                cancel=self.cancel,
            )
        except FileNotFoundError as e:
            warning = str(e)
        except InterruptedError:
            warning = "Рендер остановлен."
        except Exception as e:  # noqa: BLE001
            warning = f"Blender: {e}"
        self.done.emit(result, warning)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"PDF → Blender {__version__}")
        self.setAcceptDrops(True)
        self.resize(1280, 860)
        self.settings = settings or QSettings("bottle3d", "bottle3d")
        self.pdf: Path | None = None
        self.out: Path | None = None
        self.result: pipeline.Result | None = None
        self.job = Job()
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        # Left panel: drop zone, drawing data, actions.
        self.drop = QLabel("Перетащите сюда PDF-чертёж\nили нажмите «Открыть PDF…»")
        self.drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop.setMinimumHeight(90)
        self.drop.setWordWrap(True)
        self.drop.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; padding: 12px; }"
        )
        open_btn = QPushButton("Открыть PDF…")
        open_btn.clicked.connect(self.choose_pdf)

        self.name = QLineEdit()
        self.fields: dict[str, QLineEdit] = {}
        form = QFormLayout()
        form.addRow("Название", self.name)
        for key, label in SPEC_FIELDS:
            edit = QLineEdit()
            edit.setPlaceholderText("не указано")
            self.fields[key] = edit
            form.addRow(label, edit)
        self.colour = QComboBox()
        self.colour.setEditable(True)
        self.colour.addItems(COLOURS)
        form.addRow("Цвет стекла", self.colour)
        self.base_t = QDoubleSpinBox()
        self.base_t.setRange(1, 40)
        self.base_t.setDecimals(1)
        self.base_t.setSuffix(" мм")
        self.base_t.setValue(10)
        self.base_t.setToolTip("Из штампа не определяется. У тяжёлых бутылок под виски 8–15 мм.")
        form.addRow("Толщина дна", self.base_t)
        spec_box = QGroupBox("Данные со штампа и размеры")
        spec_box.setLayout(form)

        self.engine = QComboBox()
        self.engine.addItem("Качественный (Cycles)", ("CYCLES", 32))
        self.engine.addItem("Быстрый (EEVEE, нужна видеокарта)", ("BLENDER_EEVEE_NEXT", 32))
        self.engine.setCurrentIndex(int(self.settings.value("engine", 0)))
        self.build_btn = QPushButton("Построить модель")
        self.build_btn.setEnabled(False)
        self.build_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.save_btn = QPushButton("Сохранить результат…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        self.open_blend_btn = QPushButton("Открыть в Blender")
        self.open_blend_btn.setEnabled(False)
        self.open_blend_btn.clicked.connect(self.open_in_blender)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        left = QVBoxLayout()
        left.addWidget(self.drop)
        left.addWidget(open_btn)
        left.addWidget(spec_box)
        left.addWidget(QLabel("Рендер:"))
        left.addWidget(self.engine)
        row = QHBoxLayout()
        row.addWidget(self.build_btn)
        row.addWidget(self.stop_btn)
        left.addLayout(row)
        left.addWidget(self.save_btn)
        left.addWidget(self.open_blend_btn)
        left.addWidget(self.status)
        left.addStretch(1)
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(330)
        left_w.setMaximumWidth(420)

        # Right panel: what the person checks.
        self.tabs = QTabWidget()
        self.contour_view = ImageView("Здесь появится чертёж с найденным контуром.")
        self.render_view = ImageView("Здесь появится рендер.")
        self.report = QTextBrowser()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.contour_view, "Контур")
        self.tabs.addTab(self.render_view, "Рендер")
        self.tabs.addTab(self.report, "Отчёт")
        self.tabs.addTab(self.log, "Журнал")

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        menu = self.menuBar().addMenu("Настройки")
        act = QAction("Путь к Blender…", self)
        act.triggered.connect(self.choose_blender)
        menu.addAction(act)

    # ---- input ---------------------------------------------------------------------------

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802
        if any(u.toLocalFile().lower().endswith(".pdf") for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        for u in e.mimeData().urls():
            if u.toLocalFile().lower().endswith(".pdf"):
                self.load_pdf(Path(u.toLocalFile()))
                break

    def choose_pdf(self) -> None:
        start = self.settings.value("last_dir", str(Path.home()))
        name, _ = QFileDialog.getOpenFileName(self, "PDF-чертёж", start, "PDF (*.pdf)")
        if name:
            self.load_pdf(Path(name))

    def load_pdf(self, pdf: Path) -> None:
        self.pdf = pdf
        self.settings.setValue("last_dir", str(pdf.parent))
        saved = job_path_for(pdf)
        try:
            self.job = Job.load(saved) if saved.exists() else Job()
        except Exception as e:  # noqa: BLE001
            self.job = Job()
            self.append_log(f"Не удалось прочитать {saved.name}: {e}")
        self.fill_form(self.job)
        self.drop.setText(pdf.name)
        self.build_btn.setEnabled(True)
        self.contour_view.show_file(None)
        self.render_view.show_file(None)
        self.report.clear()
        hint = (
            "Настройки загружены из " + saved.name
            if saved.exists()
            else (
                "Заполните размеры со штампа: без общей высоты или диаметров масштаб не определить."
            )
        )
        self.status.setText(hint)

    def fill_form(self, job: Job) -> None:
        sp = job.spec
        self.name.setText(sp.name or "")
        for key, edit in self.fields.items():
            v = getattr(sp, key)
            edit.setText("" if v is None else f"{v:g}")
        self.colour.setCurrentText(sp.glass_colour or "Flint")
        self.base_t.setValue(job.base_thickness_mm)

    def read_form(self) -> Job:
        values: dict[str, object] = {"name": self.name.text().strip() or None}
        for key, edit in self.fields.items():
            text = edit.text().strip().replace(",", ".")
            if not text:
                values[key] = None
                continue
            try:
                values[key] = float(text)
            except ValueError:
                raise ValueError(f"«{text}» — не число") from None
        values["glass_colour"] = self.colour.currentText().strip() or None
        spec = self.job.spec.model_copy(update=values)
        return self.job.model_copy(
            update={
                "spec": BottleSpec.model_validate(spec.model_dump()),
                "base_thickness_mm": self.base_t.value(),
            }
        )

    # ---- run -----------------------------------------------------------------------------

    def start(self) -> None:
        if self.pdf is None or self._thread is not None:
            return
        try:
            job = self.read_form()
        except ValueError as e:
            QMessageBox.warning(self, "Проверьте поля", str(e))
            return
        self.job = job
        self.out = Path(tempfile.mkdtemp(prefix="bottle3d_"))
        engine, samples = self.engine.currentData()
        self.settings.setValue("engine", self.engine.currentIndex())
        self.append_log(f"Запуск: {self.pdf} → {self.out}")

        self._thread = QThread(self)
        self._worker = Worker(
            self.pdf, job, self.out, self.settings.value("blender") or None, engine, samples
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.stage.connect(self.on_stage)
        self._worker.done.connect(self.on_done)
        self._worker.failed.connect(self.on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self.set_busy(True)
        self._thread.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel.set()
            self.status.setText("Останавливаю…")

    def set_busy(self, busy: bool) -> None:
        self.build_btn.setEnabled(not busy and self.pdf is not None)
        self.stop_btn.setEnabled(busy)
        if busy:
            self.save_btn.setEnabled(False)
            self.open_blend_btn.setEnabled(False)

    @Slot()
    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self.set_busy(False)

    @Slot(str)
    def on_stage(self, text: str) -> None:
        self.status.setText(text)
        self.append_log(text)

    @Slot(object, str)
    def on_done(self, result: pipeline.Result, warning: str) -> None:
        self.result = result
        self.job = result.job  # keep auto-detected axis/view so a re-run reproduces them
        assert self.out is not None
        self.contour_view.show_file(self.out / "debug.png")
        has_render = (self.out / "preview.png").exists()
        self.render_view.show_file(self.out / "preview.png" if has_render else None)
        self.report.setMarkdown((self.out / "report.md").read_text(encoding="utf-8"))
        ok = result.status == "ok"
        text = "Готово: всё сходится." if ok else "Готово, но проверьте отчёт: есть замечания."
        if warning:
            text += "\n" + warning
            self.append_log(warning)
        log = self.out / "blender.log"
        if log.exists():
            self.append_log(log.read_text(encoding="utf-8", errors="replace")[-4000:])
        self.status.setText(text)
        self.status.setStyleSheet("color: #2e7d32;" if ok and not warning else "color: #b26a00;")
        self.tabs.setCurrentWidget(self.render_view if has_render else self.contour_view)
        self.save_btn.setEnabled(True)
        self.open_blend_btn.setEnabled((self.out / "model.blend").exists())

    @Slot(str, str)
    def on_failed(self, message: str, details: str) -> None:
        self.status.setText("Ошибка: " + message)
        self.status.setStyleSheet("color: #c62828;")
        self.append_log(details)

    # ---- output --------------------------------------------------------------------------

    def save(self) -> None:
        if self.out is None or self.pdf is None:
            return
        start = self.settings.value("save_dir", str(self.pdf.parent))
        target = QFileDialog.getExistingDirectory(self, "Куда сохранить", start)
        if not target:
            return
        self.settings.setValue("save_dir", target)
        dest = Path(target) / self.pdf.stem
        dest.mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:
            if (self.out / name).exists():
                shutil.copy2(self.out / name, dest / name)
        # Settings next to the drawing, so the next drop of this PDF restores them.
        try:
            self.job.save(job_path_for(self.pdf))
        except OSError as e:
            self.append_log(f"Не удалось сохранить настройки рядом с PDF: {e}")
        self.status.setText(f"Сохранено в {dest}")

    def open_in_blender(self) -> None:
        if self.out is None:
            return
        try:
            exe = blender_runner.find_blender(self.settings.value("blender") or None)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Blender", str(e))
            return
        subprocess.Popen([str(exe), str(self.out / "model.blend")])

    def choose_blender(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "blender.exe", self.settings.value("blender", ""), "Blender (blender*)"
        )
        if not name:
            return
        if blender_runner.version(Path(name)) is None:
            QMessageBox.warning(self, "Blender", "Это не похоже на Blender.")
            return
        self.settings.setValue("blender", name)
        self.status.setText(f"Blender: {name}")

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PDF → Blender")
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".pdf"):
        w.load_pdf(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
