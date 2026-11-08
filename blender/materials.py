"""Glass materials by the colour name used on bottle drawings."""

from __future__ import annotations

import bpy

# Linear RGB tints. Extend as new drawings come in.
GLASS_COLOURS: dict[str, tuple[float, float, float]] = {
    "flint": (0.95, 0.98, 0.97),
    "extra flint": (0.98, 0.99, 0.99),
    "super flint": (0.98, 0.99, 0.99),
    "half white": (0.88, 0.95, 0.90),
    "green": (0.18, 0.42, 0.16),
    "dead leaf green": (0.40, 0.42, 0.12),
    "antique green": (0.22, 0.30, 0.10),
    "uvag": (0.20, 0.30, 0.10),
    "amber": (0.55, 0.28, 0.06),
    "brown": (0.40, 0.18, 0.04),
    "cobalt": (0.10, 0.20, 0.65),
    "blue": (0.15, 0.35, 0.70),
    "black": (0.03, 0.03, 0.03),
}


def glass_colour(name: str | None) -> tuple[float, float, float]:
    key = (name or "flint").strip().lower()
    return GLASS_COLOURS.get(key, GLASS_COLOURS["flint"])


def glass_material(name: str | None) -> bpy.types.Material:
    mat = bpy.data.materials.new(f"Glass {name or 'Flint'}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    r, g, b = glass_colour(name)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["IOR"].default_value = 1.52
    bsdf.inputs["Roughness"].default_value = 0.02
    return mat
