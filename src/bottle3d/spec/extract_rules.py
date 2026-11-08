"""Drawing data from the PDF text, without any network or LLM.

Two sources:

* the title block (stamp): bilingual labels like "Butelio svoris/Glass weight" with the value in
  the same table row to the right;
* dimension texts of the main view (Ø88.3±1.6, 246.0±1.6): matched to the measured contour, so
  "body diameter" is the Ø text whose value agrees with the widest part of the outline.

Everything found is a suggestion: the job (what a person typed) always wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from bottle3d.geometry.contour import Contour
from bottle3d.pdf.vectors import PT_TO_MM

NUM = r"(\d+(?:[.,]\d+)?)"


@dataclass
class TextLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float  # page points, Y down
    horizontal: bool

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def read_lines(pdf: str | Path, page_no: int = 0) -> list[TextLine]:
    out = []
    with pymupdf.open(pdf) as doc:
        page = doc[page_no]
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    x0, y0, x1, y1 = line["bbox"]
                    out.append(TextLine(text, x0, y0, x1, y1, abs(line["dir"][1]) < 0.1))
    return out


def _num(s: str) -> float | None:
    m = re.search(NUM, s)
    return float(m.group(1).replace(",", ".")) if m else None


# (field, label regex, value regex). Labels are matched case-insensitively.
STAMP_FIELDS: list[tuple[str, str, str]] = [
    ("glass_weight_g", r"glass\s*weight|svoris", rf"~?\s*{NUM}\s*g\b"),
    ("nominal_capacity_ml", r"nominal\s*capacity|nominali", rf"{NUM}\s*ml"),
    ("brimful_capacity_ml", r"brim+ful+\s*capacity|pilnoji", rf"~?\s*{NUM}\s*ml"),
    ("fill_height_mm", r"fill\s*hei\w*", rf"{NUM}\s*mm"),
    ("glass_colour", r"glass\s*colou?r|spalva", r"([A-Za-z][A-Za-z .\-]{1,30})"),
    ("drawing_no", r"drawing\s*no|brėž", r"([A-Z]{1,4}\d[\w\-./]{2,})"),
]


def _value_in_row(lines: list[TextLine], label: TextLine, label_re: str, value_re: str):
    """Value after the label on the same line, or in the nearest cell to its right."""
    # Bilingual labels match twice ("Stiklo spalva/Glass colour"): look after the last match.
    ends = [m.end() for m in re.finditer(label_re, label.text, flags=re.I)]
    rest = label.text[ends[-1] :] if ends else ""
    m = re.search(value_re, rest, flags=re.I)
    if m:
        return m.group(1)
    h = label.y1 - label.y0
    row = [
        ln
        for ln in lines
        if ln.horizontal
        and ln is not label
        and ln.x0 >= label.x1 - 1
        and abs(ln.cy - label.cy) <= 0.6 * h
    ]
    for ln in sorted(row, key=lambda ln: ln.x0):
        m = re.fullmatch(r"\s*" + value_re + r"\s*", ln.text, flags=re.I)
        if m:
            return m.group(1)
    return None


def read_stamp(lines: list[TextLine]) -> dict[str, object]:
    found: dict[str, object] = {}
    for field, label_re, value_re in STAMP_FIELDS:
        for ln in lines:
            if not ln.horizontal or not re.search(label_re, ln.text, flags=re.I):
                continue
            v = _value_in_row(lines, ln, label_re, value_re)
            if v is None:
                continue
            if field in ("glass_colour", "drawing_no"):
                found[field] = v.strip()
            else:
                found[field] = float(v.replace(",", "."))
            break

    # Name: the cell just above its caption ("Pavadinimas/Name.").
    for ln in lines:
        if re.search(r"\bname\b", ln.text, flags=re.I) and len(ln.text) < 30:
            above = [
                o
                for o in lines
                if o.horizontal
                and o is not ln
                and o.y1 <= ln.y0 + 1
                and ln.y0 - o.y1 < 2 * (ln.y1 - ln.y0)
                and min(o.x1, ln.x1) - max(o.x0, ln.x0) > 0
            ]
            if above:
                found["name"] = max(above, key=lambda o: o.y1).text.strip()
                break

    # Neck bore: "d(18.7±0.3)" or "Ød*(18.7…)".
    for ln in lines:
        m = re.search(r"Ø?d\*?\s*\(\s*" + NUM, ln.text)
        if m:
            found["neck_inner_diameter_mm"] = float(m.group(1).replace(",", "."))
            break
    return found


def read_dimensions(
    lines: list[TextLine],
    contour: Contour,
    view_bbox_mm: tuple[float, float, float, float],
    page_height_mm: float,
    tol: float = 0.06,
) -> dict[str, float]:
    """Overall height and diameters: dimension texts whose value matches the outline.

    The drawing scale is unknown but close to the printed one, so a text is accepted when it is
    within `tol` of the measured size, and the closest one wins.
    """
    x0, y0, x1, y1 = view_bbox_mm
    margin = 60.0

    def near_view(ln: TextLine) -> bool:
        cx = (ln.x0 + ln.x1) / 2 * PT_TO_MM
        cy = page_height_mm - ln.cy * PT_TO_MM
        return x0 - margin <= cx <= x1 + margin and y0 - margin <= cy <= y1 + margin

    dia, plain = [], []
    for ln in lines:
        if not near_view(ln):
            continue
        for m in re.finditer(r"(Ø|⌀)?\s*" + NUM, ln.text):
            v = float(m.group(2).replace(",", "."))
            (dia if m.group(1) else plain).append(v)

    def best(cands: list[float], measured: float) -> float | None:
        ok = [c for c in cands if abs(c / measured - 1) <= tol]
        return min(ok, key=lambda c: abs(c / measured - 1)) if ok else None

    out: dict[str, float] = {}
    for field, cands, measured in (
        ("overall_height_mm", plain, contour.height),
        ("body_diameter_mm", dia, contour.max_diameter),
        ("base_diameter_mm", dia, contour.base_diameter),
    ):
        v = best(cands, measured)
        if v is not None:
            out[field] = v
    return out
