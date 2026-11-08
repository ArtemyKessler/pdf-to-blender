"""Find a Blender 4.2+ executable and run blender/build_bottle.py in it."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "blender" / "build_bottle.py"


def candidates() -> list[Path]:
    found: list[Path] = []
    env = os.environ.get("BLENDER")
    if env:
        found.append(Path(env))
    on_path = shutil.which("blender")
    if on_path:
        found.append(Path(on_path))
    # Standard Windows installs: C:\Program Files\Blender Foundation\Blender 4.2\blender.exe
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if root:
            base = Path(root) / "Blender Foundation"
            if base.is_dir():
                found.extend(sorted(base.glob("Blender */blender.exe"), reverse=True))
    return found


def version(exe: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [str(exe), "-b", "--factory-startup", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"Blender (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_blender(explicit: str | Path | None = None) -> Path:
    options = [Path(explicit)] if explicit else candidates()
    for exe in options:
        v = version(exe)
        if v and v >= (4, 2):
            return exe
    raise FileNotFoundError(
        "Blender 4.2+ не найден. Установите Blender 4.2 LTS или укажите путь через --blender "
        "или переменную окружения BLENDER."
    )


def build(
    profile_json: Path,
    out_dir: Path,
    *,
    blender: str | Path | None = None,
    preview: bool = True,
    glb: bool = False,
    engine: str = "CYCLES",
    samples: int = 64,
    timeout: int = 900,
) -> Path:
    exe = find_blender(blender)
    args = [
        str(exe),
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "-P",
        str(SCRIPT),
        "--",
        str(profile_json),
        str(out_dir),
        "--engine",
        engine,
        "--samples",
        str(samples),
    ]
    if preview:
        args.append("--preview")
    if glb:
        args.append("--glb")
    log = out_dir / "blender.log"
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender завершился с ошибкой, подробности в {log}")
    return out_dir / "model.blend"
