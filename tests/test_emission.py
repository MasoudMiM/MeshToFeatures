# SPDX-License-Identifier: LGPL-2.1-or-later
"""Emission-planner tests, written before the implementation.

The planner converts recognized (infinite) primitives into *bounded*
patch specs -- local frame + parameter ranges -- that a CAD kernel can
turn into faces. Everything here is pure numpy and must hold for any
downstream kernel:

* every segment sample must lie inside its patch's parameter ranges,
* full revolutions must be detected as such,
* partial coverage (half-pipes, spherical caps) must produce tight ranges,
* plane patches must produce a convex polygon containing all points.
"""

import numpy as np
import trimesh

from meshtofeatures.emission import PatchSpec, plan_patches
from meshtofeatures.pipeline import reconstruct

Z = np.array([0.0, 0.0, 1.0])


def _frame_is_right_handed(p: PatchSpec):
    assert np.allclose(np.cross(p.x_dir, p.y_dir), p.z_dir, atol=1e-12)
    for v in (p.x_dir, p.y_dir, p.z_dir):
        assert np.isclose(np.linalg.norm(v), 1.0)


def _half(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep faces whose centroid has x > 0 (approx. half the solid)."""
    out = mesh.copy()
    out.update_faces(out.triangles_center[:, 0] > 0)
    out.remove_unreferenced_vertices()
    return out


class TestCylinderPatch:
    def _patch(self, mesh):
        rep = reconstruct(mesh)
        patches = plan_patches(rep)
        return [p for p in patches if p.kind == "cylinder"][0]

    def test_full_revolution_detected(self):
        p = self._patch(trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64))
        assert p.full_u
        assert np.allclose(p.u_range, (0.0, 2 * np.pi))

    def test_v_range_covers_height(self):
        p = self._patch(trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64))
        v0, v1 = p.v_range
        assert np.isclose(v1 - v0, 3.0, atol=1e-9)
        # origin is placed so the patch starts at v=0
        assert np.isclose(v0, 0.0)
        _frame_is_right_handed(p)

    def test_half_pipe_angular_range(self):
        p = self._patch(_half(trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)))
        assert not p.full_u
        u0, u1 = p.u_range
        assert 0.9 * np.pi < (u1 - u0) < 1.1 * np.pi

    def test_all_points_inside_ranges(self):
        mesh = _half(trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64))
        rep = reconstruct(mesh)
        for p in plan_patches(rep):
            if p.kind != "cylinder":
                continue
            rel = p.primitive_points - p.origin
            h = rel @ p.z_dir
            v0, v1 = p.v_range
            assert h.min() >= v0 - 1e-9 and h.max() <= v1 + 1e-9
            ang = np.arctan2(rel @ p.y_dir, rel @ p.x_dir) % (2 * np.pi)
            u0, u1 = p.u_range
            shifted = (ang - u0) % (2 * np.pi)
            assert shifted.max() <= (u1 - u0) + 1e-9


class TestPlanePatch:
    def test_polygon_contains_all_points(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        rep = reconstruct(mesh)
        patches = [p for p in plan_patches(rep) if p.kind == "plane"]
        assert len(patches) == 6
        for p in patches:
            assert p.polygon is not None and len(p.polygon) >= 3
            _frame_is_right_handed(p)
            rel = p.primitive_points - p.origin
            uv = np.column_stack([rel @ p.x_dir, rel @ p.y_dir])
            # convex-hull containment: every point is a convex combination;
            # cheap check: point inside all hull half-planes
            poly = p.polygon
            for k in range(len(poly)):
                a, b = poly[k], poly[(k + 1) % len(poly)]
                edge = b - a
                normal_in = np.array([-edge[1], edge[0]])
                assert np.all((uv - a) @ normal_in >= -1e-9)

    def test_box_face_polygon_is_rectangle(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        rep = reconstruct(mesh)
        p = [q for q in plan_patches(rep) if q.kind == "plane"][0]
        assert len(p.polygon) == 4


class TestSpherePatch:
    def test_full_sphere_ranges(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
        rep = reconstruct(mesh)
        p = [q for q in plan_patches(rep) if q.kind == "sphere"][0]
        assert p.full_u
        v0, v1 = p.v_range
        assert v0 <= -np.deg2rad(80) and v1 >= np.deg2rad(80)

    def test_cap_elevation_range_is_tight(self):
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        mesh.update_faces(mesh.triangles_center[:, 2] > 0.5)  # cap above 30 deg
        mesh.remove_unreferenced_vertices()
        rep = reconstruct(mesh)
        p = [q for q in plan_patches(rep) if q.kind == "sphere"][0]
        v0, v1 = p.v_range
        # frame z is the cap's mean direction (~ +Z); elevations of a cap
        # from ~30 deg upward, measured from its own pole frame, span less
        # than the full hemisphere
        assert v1 - v0 < np.deg2rad(75)
        _frame_is_right_handed(p)


class TestConePatch:
    def test_cone_ranges(self):
        mesh = trimesh.creation.cone(radius=1.0, height=2.0, sections=64)
        rep = reconstruct(mesh)
        p = [q for q in plan_patches(rep) if q.kind == "cone"][0]
        assert p.full_u
        v0, v1 = p.v_range  # height above apex, along the cone axis
        assert np.isclose(v0, 0.0, atol=1e-6)
        assert np.isclose(v1, 2.0, atol=1e-6)
        # origin is the apex
        assert np.allclose(p.origin, p.primitive.apex)


class TestPlanCompleteness:
    def test_one_patch_per_recognized_surface(self):
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        rep = reconstruct(mesh)
        patches = plan_patches(rep)
        assert len(patches) == len(rep.surfaces)
        assert sorted(p.kind for p in patches) == rep.kinds()
