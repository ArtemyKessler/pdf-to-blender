# PyInstaller spec for the Windows desktop app:  uv run pyinstaller packaging/bottle3d.spec
# Output: dist/PdfToBlender/PdfToBlender.exe (one folder; zip it to hand over).
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

root = Path(SPECPATH).parent

a = Analysis(
    [str(root / "packaging" / "launch.py")],
    pathex=[str(root / "src")],
    # Blender scripts run inside Blender, so they ship as plain files, not compiled modules.
    datas=[(str(root / "blender"), "blender"), *copy_metadata("bottle3d")],
    hiddenimports=[],
    excludes=[
        "tkinter",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.QtMultimedia",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
    ],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PdfToBlender",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PdfToBlender")
