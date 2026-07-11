# SPDX-License-Identifier: LGPL-2.1-or-later
"""Boundary-loop tests, written before the implementation.

A segment's boundary consists of the directed mesh edges used by exactly
one of its faces. Chained, they form closed loops: one outer outline plus
one loop per hole. Because face winding is counter-clockwise seen from
outside, the projected outer loop is CCW and hole loops are CW when the
projection frame's z matches the outward normal.

Fixtures with known loop structure:
* box face      -> exactly 1 loop with 4 vertices,
* cylinder barrel -> exactly 2 loops (the two rims),
* annulus top   -> exactly 2 loops (outline + hole) -- the case that
  produces a *plane patch with a hole*, fixing the filled-hole artefact
  observed on 1002_tray_bottom.STL.
"""

import numpy as np
import trimesh

from meshtofeatures.pipeline import reconstruct
from meshtofeatures.segmentation import segment_mesh
from meshtofeatures.emission import plan_patches


def _shoelace(loop2d: np.ndarray) -> float:
    x, y = loop2d[:, 0], loop2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ------------------------------------------------------- loop extraction

class TestBoundaryLoops:
    def test_box_face_has_one_square_loop(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        for seg in segment_mesh(mesh):
            assert len(seg.boundary_loops) == 1
            assert len(seg.boundary_loops[0]) == 4

    def test_cylinder_barrel_has_two_rim_loops(self):
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        barrel = segment_mesh(mesh)[0]
        assert len(barrel.boundary_loops) == 2
        for loop in barrel.boundary_loops:
            # every rim vertex sits at radius 1 and |z| = 1.5
            r = np.linalg.norm(loop[:, :2], axis=1)
            assert np.allclose(r, 1.0, atol=1e-9)
            assert np.allclose(np.abs(loop[:, 2]), 1.5, atol=1e-9)

    def test_sphere_has_no_boundary(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        seg = segment_mesh(mesh)[0]
        assert seg.boundary_loops == []

    def test_annulus_plane_has_two_loops(self):
        mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.5, height=1.0, sections=64)
        planes = [s for s in segment_mesh(mesh)
                  if np.allclose(np.abs(s.face_normals @ [0, 0, 1.0]), 1.0, atol=1e-6)]
        assert len(planes) == 2
        for seg in planes:
            assert len(seg.boundary_loops) == 2

    def test_loops_are_closed_chains_of_segment_vertices(self):
        mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.5, height=1.0, sections=64)
        for seg in segment_mesh(mesh):
            seg_pts = {tuple(np.round(p, 12)) for p in seg.points}
            for loop in seg.boundary_loops:
                assert len(loop) >= 3
                # no duplicated closing vertex; all vertices belong to segment
                assert len({tuple(np.round(p, 12)) for p in loop}) == len(loop)
                for p in loop:
                    assert tuple(np.round(p, 12)) in seg_pts


# ------------------------------------------------- hole-aware plane patch

class TestPlanePatchWithHoles:
    def _top_patch(self):
        mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.5, height=1.0, sections=64)
        rep = reconstruct(mesh)
        patches = [p for p in plan_patches(rep) if p.kind == "plane"]
        assert len(patches) == 2
        return patches[0]

    def test_outer_polygon_and_one_hole(self):
        p = self._top_patch()
        assert p.polygon is not None
        assert len(p.holes) == 1

    def test_orientation_outer_ccw_hole_cw(self):
        p = self._top_patch()
        assert _shoelace(p.polygon) > 0
        assert _shoelace(p.holes[0]) < 0

    def test_net_area_matches_annulus(self):
        p = self._top_patch()
        net = _shoelace(p.polygon) + _shoelace(p.holes[0])
        expected = np.pi * (1.5**2 - 0.5**2)
        # 64-gon area deficit vs circle is ~0.16%; allow 1%
        assert abs(net - expected) / expected < 0.01

    def test_hole_surrounds_plane_origin_region(self):
        # the annulus axis passes through the hole: the frame-origin
        # projection of the axis point must lie inside the hole loop
        p = self._top_patch()
        rel = np.zeros(3) - p.origin           # axis point (0,0,z) projected
        pt = np.array([rel @ p.x_dir, rel @ p.y_dir])
        hole = p.holes[0]
        # winding-number point-in-polygon (hole is CW: winding = -1 inside)
        wn = 0.0
        for k in range(len(hole)):
            a, b = hole[k] - pt, hole[(k + 1) % len(hole)] - pt
            wn += np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))
        assert abs(wn) > np.pi  # nonzero winding: inside the hole loop

    def test_convex_solid_planes_have_no_holes(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        rep = reconstruct(mesh)
        for p in plan_patches(rep):
            if p.kind == "plane":
                assert p.holes == []
                assert len(p.polygon) == 4
