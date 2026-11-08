from shapely.geometry import LineString, MultiPolygon, Polygon, box

from bottle3d.geometry.engraving import _faces, flat_mesh


def _area(verts, faces) -> float:
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    return float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2)


def test_outline_inside_outline_is_a_hole():
    outer = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    inner = LineString([(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)])
    shape = _faces([outer, inner])
    assert shape.area == 100 - 16


def test_flat_mesh_covers_the_shape_counter_clockwise():
    o = box(0, 0, 10.5, 4)
    shape = MultiPolygon([Polygon(o.exterior, [box(2, 1, 4, 3).exterior.coords])])
    verts, faces = flat_mesh(shape, cell=1.0)
    # Positive signed area: every triangle faces the same way (+w, away from the glass).
    assert _area(verts, faces) == shape.area
    assert len(faces) > 40  # cut into cells so it can bend
