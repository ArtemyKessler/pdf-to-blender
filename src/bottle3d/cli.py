"""bottle3d run drawing.pdf --job job.yaml --out out/"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bottle3d import blender_runner, pipeline
from bottle3d.spec.schema import Job


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bottle3d")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="PDF → profile.json, report, .blend")
    r.add_argument("pdf", type=Path)
    r.add_argument("--out", type=Path, default=Path("out"))
    r.add_argument("--job", type=Path, help="job.yaml with manual settings and spec")
    r.add_argument("--profile", type=Path, help="manufacturer defaults (same format as job.yaml)")
    r.add_argument("--blender", help="path to blender.exe (default: auto-detect)")
    r.add_argument("--no-blender", action="store_true", help="stop after profile.json")
    r.add_argument("--glb", action="store_true")
    r.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    r.add_argument("--samples", type=int, default=64)
    a = ap.parse_args(argv)

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console and Cyrillic messages

    job = Job.load(a.job) if a.job else Job()
    if a.profile:
        job = job.merged_with(Job.load(a.profile))
    res = pipeline.run(a.pdf, job, a.out)

    print(f"Статус: {res.status}")
    for p in res.problems:
        print(f"  - {p}")
    for n in res.notes:
        print(f"  * {n}")
    print(f"Отчёт: {a.out / 'report.md'}, наложение: {a.out / 'debug.png'}")

    if not a.no_blender:
        blend = blender_runner.build(
            a.out / "profile.json",
            a.out,
            blender=a.blender,
            glb=a.glb,
            engine=a.engine,
            samples=a.samples,
        )
        print(f"Модель: {blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
