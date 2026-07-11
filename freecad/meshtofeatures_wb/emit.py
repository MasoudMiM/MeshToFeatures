# SPDX-License-Identifier: LGPL-2.1-or-later
"""FreeCAD-side emission: realize tested :class:`PatchSpec` plans as
``Part`` faces in a document.

This module is the *only* place that touches FreeCAD geometry APIs; all
geometric decisions (frames, parameter ranges, hulls) were made and
tested in ``meshtofeatures.emission``. Keep it thin.

OCC parametrization notes (why the conversions below look the way they do):

* ``Part.Cylinder``: P(u, v) = C + r cos(u) X + r sin(u) Y + v Z --
  our (u_range, v_range) map 1:1.
* ``Part.Cone``: P(u, v) = C + (R + v sin(a)) (cos u X + sin u Y)
  + v cos(a) Z -- ``v`` is *slant* distance from the reference circle of
  radius R at C. We place C at the patch's lower height ``h0`` above the
  apex with R = h0 tan(a), so v runs 0 .. (h1 - h0)/cos(a).
* ``Part.Sphere``: u = azimuth, v = elevation in [-pi/2, pi/2] -- 1:1.
"""

from __future__ import annotations

import math

import FreeCAD as App  # type: ignore
import Part  # type: ignore

_KIND_COLORS = {
    "plane": (0.35, 0.65, 0.95),
    "cylinder": (0.95, 0.60, 0.20),
    "cone": (0.60, 0.85, 0.35),
    "sphere": (0.85, 0.40, 0.75),
}


def mesh_object_to_trimesh(mesh_obj):
    """Convert a FreeCAD ``Mesh::Feature`` into a ``trimesh.Trimesh``."""
    import numpy as np
    import trimesh

    points, facets = mesh_obj.Mesh.Topology
    vertices = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
    faces = np.array(facets, dtype=np.int64)
    # process=True welds duplicate vertices (unwelded STLs would otherwise
    # have no face adjacency and segmentation would see disconnected soup)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _placement(origin, x_dir, y_dir, z_dir) -> "App.Placement":
    m = App.Matrix(
        float(x_dir[0]), float(y_dir[0]), float(z_dir[0]), float(origin[0]),
        float(x_dir[1]), float(y_dir[1]), float(z_dir[1]), float(origin[1]),
        float(x_dir[2]), float(y_dir[2]), float(z_dir[2]), float(origin[2]),
        0.0, 0.0, 0.0, 1.0,
    )
    return App.Placement(m)


def patch_to_shape(spec) -> "Part.Shape":
    """Build a bounded ``Part`` face from a :class:`PatchSpec`."""
    kind = spec.kind
    if kind == "plane":
        def wire_of(loop2d):
            pts3d = [
                App.Vector(*(spec.origin + a * spec.x_dir + b * spec.y_dir))
                for a, b in loop2d
            ]
            return Part.makePolygon(pts3d + [pts3d[0]])

        face = Part.Face(wire_of(spec.polygon))
        for hole in getattr(spec, "holes", []) or []:
            face = face.cut(Part.Face(wire_of(hole)))
        return face

    u0, u1 = float(spec.u_range[0]), float(spec.u_range[1])
    v0, v1 = float(spec.v_range[0]), float(spec.v_range[1])

    if kind == "cylinder":
        surf = Part.Cylinder()
        surf.Radius = float(spec.primitive.radius)
        shape = surf.toShape(u0, u1, v0, v1)
        shape.Placement = _placement(spec.origin, spec.x_dir, spec.y_dir, spec.z_dir)
        return shape

    if kind == "sphere":
        surf = Part.Sphere()
        surf.Radius = float(spec.primitive.radius)
        shape = surf.toShape(u0, u1, v0, v1)
        shape.Placement = _placement(spec.origin, spec.x_dir, spec.y_dir, spec.z_dir)
        return shape

    if kind == "cone":
        alpha = float(spec.primitive.half_angle)
        h0, h1 = v0, v1
        surf = Part.Cone()
        surf.SemiAngle = alpha
        surf.Radius = h0 * math.tan(alpha)   # reference circle at height h0
        slant = (h1 - h0) / math.cos(alpha)
        shape = surf.toShape(u0, u1, 0.0, slant)
        ref_origin = spec.origin + h0 * spec.z_dir  # apex + h0 along axis
        shape.Placement = _placement(ref_origin, spec.x_dir, spec.y_dir, spec.z_dir)
        return shape

    raise ValueError(f"unknown patch kind: {kind}")


def emit_report(doc, patches, actions=None, group_label="Reconstruction",
                features=None, patterns=None):
    """Create one ``Part::Feature`` per patch inside a group; returns the group.

    ``actions`` (snapping audit trail) is stored on the group as a string
    list property so the inference record travels with the document.
    """
    group = doc.addObject("App::DocumentObjectGroup", "Reconstruction")
    group.Label = group_label

    for i, spec in enumerate(patches):
        try:
            shape = patch_to_shape(spec)
        except Exception as exc:  # noqa: BLE001 - never lose the whole run
            App.Console.PrintWarning(
                f"[meshtofeatures] patch {i} ({spec.kind}) failed: {exc}\n")
            continue
        obj = doc.addObject("Part::Feature", f"Surface{i:03d}")
        obj.Shape = shape
        obj.Label = f"{i:03d} {spec.label}"
        group.addObject(obj)
        if App.GuiUp and hasattr(obj, "ViewObject") and obj.ViewObject:
            color = _KIND_COLORS.get(spec.kind, (0.7, 0.7, 0.7))
            try:
                obj.ViewObject.ShapeColor = color
                obj.ViewObject.Transparency = 30
            except Exception:  # noqa: BLE001 - cosmetics must never fail a run
                pass

    if actions:
        group.addProperty("App::PropertyStringList", "SnapActions", "meshtofeatures",
                          "Design-intent inferences applied during snapping")
        group.SnapActions = [
            f"[{a.kind}] {'OK ' if a.accepted else 'REJECTED '} {a.detail}"
            for a in actions
        ]
    if features is not None and features.features:
        group.addProperty("App::PropertyStringList", "Features", "meshtofeatures",
                          "Detected manufacturing features")
        lines = []
        in_pattern = set()
        if patterns is not None:
            lines += [p.description for p in patterns.patterns]
            in_pattern = {id(m) for p in patterns.patterns for m in p.members}
        lines += [f.description for f in features.features
                  if id(f) not in in_pattern]
        group.Features = lines
    doc.recompute()
    return group
