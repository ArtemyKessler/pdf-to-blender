"""Runs inside Blender: profile.json → model.blend (+ preview.png, model.glb, stats.json).

    blender -b --factory-startup -P blender/build_bottle.py -- profile.json out/ [--preview]

Only bpy, bmesh, mathutils and the standard library are used here. Geometry is in metres
(scale_length = 1): Cycles lights, camera clipping and glTF all treat Blender units as metres.
Millimetres are used only for display.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from materials import glass_material  # noqa: E402

STEPS = 128


def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    us = scene.unit_settings
    us.system = "METRIC"
    us.scale_length = 1.0
    us.length_unit = "MILLIMETERS"
    return scene


def build_mesh(section: list[list[float]], steps: int = STEPS) -> bpy.types.Object:
    """Revolve the open (r, z) polyline around Z. Both ends must lie on the axis (r = 0)."""
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0.0, z)) for r, z in section]
    edges = [bm.edges.new((verts[i], verts[i + 1])) for i in range(len(verts) - 1)]
    bmesh.ops.spin(
        bm,
        geom=verts + edges,
        cent=(0, 0, 0),
        axis=(0, 0, 1),
        angle=2 * math.pi,
        steps=steps,
        use_merge=True,
    )
    # Points on the axis were copied at every step; weld them into poles.
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges[:], dist=1e-9)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # Smooth shading with sharp edges where the profile has corners (rim, bead, base edge).
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(35):
            e.smooth = False

    mesh = bpy.data.meshes.new("Bottle")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Bottle", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _wrap(verts_uvw, side: list[list[float]], z_center: float, angle: float):
    """Map flat (u, v, w) onto the outer surface: u around the bottle, v along its side from
    the point at height `z_center`, w outwards along the surface normal."""
    rs = [p[0] for p in side]
    zs = [p[1] for p in side]
    s = [0.0]
    for i in range(1, len(side)):
        s.append(s[-1] + math.hypot(rs[i] - rs[i - 1], zs[i] - zs[i - 1]))

    def at(arc: float):
        arc = min(max(arc, 0.0), s[-1])
        i = max(1, min(len(s) - 1, next((k for k in range(1, len(s)) if s[k] >= arc), len(s) - 1)))
        ds = s[i] - s[i - 1] or 1e-12
        t = (arc - s[i - 1]) / ds
        r = rs[i - 1] + t * (rs[i] - rs[i - 1])
        z = zs[i - 1] + t * (zs[i] - zs[i - 1])
        return r, z, (zs[i] - zs[i - 1]) / ds, -(rs[i] - rs[i - 1]) / ds

    # Arc length at the centre height (the side profile goes up monotonically).
    k = next((k for k in range(1, len(zs)) if zs[k] >= z_center), len(zs) - 1)
    dz = zs[k] - zs[k - 1] or 1e-12
    s_c = s[k - 1] + (s[k] - s[k - 1]) * (z_center - zs[k - 1]) / dz
    out = []
    for u, v, w in verts_uvw:
        r, z, nr, nz = at(s_c + v)
        theta = angle + u / max(r, 1e-6)
        rr = r + w * nr
        out.append((rr * math.cos(theta), rr * math.sin(theta), z + w * nz))
    return out


def build_engraving(e: dict, side: list[list[float]], sink: float = 0.0002) -> bpy.types.Object:
    """Raised lettering: the flat outline extruded by its relief, sunk slightly into the glass
    so that the union with the bottle is clean, then wrapped onto the surface."""
    bm = bmesh.new()
    verts = [bm.verts.new((u, v, -sink)) for u, v in e["verts"]]
    for f in e["faces"]:
        try:
            bm.faces.new([verts[i] for i in f])
        except ValueError:  # duplicate face
            pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-8)
    res = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    top = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    for vtx in top:
        vtx.co.z = e["relief"]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.verts.ensure_lookup_table()
    mapped = _wrap(
        [tuple(v.co) for v in bm.verts], side, e["z_center"], math.radians(e["angle_deg"])
    )
    for vtx, co in zip(bm.verts, mapped):
        vtx.co = co
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new("Engraving")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Engraving", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_engravings(bottle: bpy.types.Object, profile: dict) -> int:
    """Union the engravings into the bottle. Returns how many were added."""
    side = profile.get("outer_side") or []
    n = 0
    for e in profile.get("engravings", []):
        if not e.get("faces") or len(side) < 2:
            continue
        eng = build_engraving(e, side)
        mod = bottle.modifiers.new("Engraving", "BOOLEAN")
        mod.operation = "UNION"
        mod.solver = "EXACT"
        mod.object = eng
        bpy.context.view_layer.objects.active = bottle
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(eng, do_unlink=True)
        n += 1
    return n


def mesh_stats(obj: bpy.types.Object) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    volume = bm.calc_volume(signed=True)
    bm.free()
    dims = obj.dimensions
    return {
        "height_mm": dims.z * 1000,
        "diameter_mm": max(dims.x, dims.y) * 1000,
        "non_manifold_edges": non_manifold,
        "volume_cm3": volume * 1e6,
        "faces": len(obj.data.polygons),
    }


def setup_studio(scene: bpy.types.Scene, obj: bpy.types.Object) -> None:
    h = obj.dimensions.z
    d = max(obj.dimensions.x, obj.dimensions.y)

    # Dark studio with tall strip lights: glass reads by its edge highlights and refractions,
    # and disappears on a bright background.
    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.025, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=h * 8, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"
    bpy.ops.mesh.primitive_plane_add(
        size=h * 8, location=(0, h * 2.0, h * 2), rotation=(math.pi / 2, 0, 0)
    )
    back = bpy.context.object
    back.name = "Backdrop"
    for o, colour, rough in ((floor, 0.04, 0.25), (back, 0.015, 0.8)):
        mat = bpy.data.materials.new(o.name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (colour, colour, colour, 1)
        bsdf.inputs["Roughness"].default_value = rough
        o.data.materials.append(mat)

    scale = (h / 0.25) ** 2  # tuned for a 250 mm bottle
    for name, loc, energy, size_x, size_y in (
        ("Strip L", (-h * 0.9, h * 0.6, h * 0.55), 40, h * 0.12, h * 1.4),
        ("Strip R", (h * 0.9, h * 0.6, h * 0.55), 40, h * 0.12, h * 1.4),
        ("Top", (0, -h * 0.2, h * 1.8), 25, h * 0.8, h * 0.8),
        ("Front", (-h * 0.8, -h * 1.6, h * 0.7), 6, h * 0.6, h * 1.0),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.shape = "RECTANGLE"
        light.size = size_x
        light.size_y = size_y
        light.energy = energy * scale
        lo = bpy.data.objects.new(name, light)
        lo.location = loc
        _look_at(lo, Vector((0, 0, h * 0.5)))
        scene.collection.objects.link(lo)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 85
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    # Frame the bottle with a margin: distance from the vertical field of view.
    sensor = cam_data.sensor_width
    fov_v = 2 * math.atan(sensor / 2 / cam_data.lens)
    dist = (h * 1.15) / 2 / math.tan(fov_v / 2) + d / 2
    target = Vector((0, 0, h * 0.5))
    cam.location = target + Vector((0, -dist, h * 0.12))
    _look_at(cam, target)


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render(scene: bpy.types.Scene, path: Path, engine: str, samples: int) -> None:
    scene.render.engine = engine
    scene.render.resolution_x = 768
    scene.render.resolution_y = 1024
    scene.render.film_transparent = False
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 16
        scene.cycles.transmission_bounces = 16
        scene.cycles.transparent_max_bounces = 16
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def build(
    profile_path: Path,
    out_dir: Path,
    *,
    preview: bool = False,
    glb: bool = False,
    engine: str = "CYCLES",
    samples: int = 64,
) -> dict:
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if profile.get("units") != "m":
        raise ValueError("profile.json must be in metres")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = reset_scene()
    obj = build_mesh(profile["section"])
    n_engr = add_engravings(obj, profile)
    obj.data.materials.append(glass_material(profile.get("material", {}).get("glass_colour")))
    stats = mesh_stats(obj)
    stats["engravings"] = n_engr
    (out_dir / "blender_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")

    if glb:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(out_dir / "model.glb"), use_selection=True)

    setup_studio(scene, obj)
    bpy.ops.wm.save_as_mainfile(filepath=str((out_dir / "model.blend").resolve()))
    if preview:
        render(scene, (out_dir / "preview.png").resolve(), engine, samples)
    return stats


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--glb", action="store_true")
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--samples", type=int, default=64)
    a = ap.parse_args(argv)
    stats = build(
        a.profile, a.out, preview=a.preview, glb=a.glb, engine=a.engine, samples=a.samples
    )
    print("bottle3d stats:", json.dumps(stats))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
