# SPDX-License-Identifier: LGPL-2.1-or-later
"""Robustness suite (v0.5), written before the implementation.

Four mesh realities, in dependency order:

1. **Dirty meshes** -- unwelded triangle soup (how STLs actually arrive)
   and degenerate faces must be conditioned away transparently.
2. **Coarse / varied tessellation** -- the segmentation angle threshold
   must adapt to the mesh's own dihedral-angle distribution instead of
   assuming fine tessellation.
3. **Noise + outliers** -- fitting must survive scan-like noise and gross
   outlier vertices via trimmed refitting.
4. **Tangent fillets** -- blends meet their neighbours with NO sharp
   edge, so dihedral growing merges them; failed segments must be
   re-split by discrete curvature (plane k~0 vs fillet k~1/r) and then
   recognized -- fillets are just partial cylinders.
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.conditioning import condition_mesh
from meshtofeatures.fitting import fit_cylinder
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.segmentation import adaptive_angle_threshold, segment_mesh
from meshtofeatures.snapping import snap_report

Z = np.array([0.0, 0.0, 1.0])
RNG = np.random.default_rng(11)


def _soup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Explode a mesh into unwelded triangle soup (each face owns 3
    private vertices), mimicking a raw STL load without vertex merging."""
    v = mesh.triangles.reshape(-1, 3).copy()
    f = np.arange(len(v)).reshape(-1, 3)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def rounded_slab() -> trimesh.Trimesh:
    """40 x 30 x 5 slab whose four vertical edges are r=3 fillets,
    *tangent* to the side faces: the canonical dihedral-blind case."""
    from shapely.geometry import box as sbox
    poly = sbox(-20.0, -15.0, 20.0, 15.0).buffer(-3.0).buffer(3.0, quad_segs=24)
    return trimesh.creation.extrude_polygon(poly, height=5.0)


# ------------------------------------------------------------ 1. dirty

class TestConditioning:
    def test_soup_is_welded(self):
        mesh = _soup(trimesh.creation.box(extents=[2, 2, 2]))
        assert len(mesh.face_adjacency) == 0  # soup has no shared edges
        clean, rep = condition_mesh(mesh)
        assert len(clean.vertices) == 8
        assert clean.is_watertight
        assert rep.vertices_merged == len(mesh.vertices) - 8

    def test_degenerate_faces_removed(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
        f = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 1], [2, 2, 2]])  # 2 degenerate
        clean, rep = condition_mesh(trimesh.Trimesh(vertices=v, faces=f, process=False))
        assert len(clean.faces) == 2
        assert rep.faces_removed == 2

    def test_reconstruct_accepts_soup_transparently(self):
        pristine = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        rep = reconstruct(_soup(pristine))
        assert rep.kinds() == ["cylinder", "plane", "plane"]
        assert np.isclose(rep.by_kind("cylinder")[0].fit.primitive.radius, 1.0,
                          atol=1e-9)


# ------------------------------------------------------------ 2. coarse

class TestAdaptiveThreshold:
    def test_fine_mesh_threshold_is_moderate(self):
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        t = adaptive_angle_threshold(mesh)
        assert np.deg2rad(15) <= t <= np.deg2rad(60)
        assert t > np.deg2rad(5.7)   # above the tessellation angle

    def test_coarse_mesh_threshold_exceeds_facet_angle(self):
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=8)
        t = adaptive_angle_threshold(mesh)
        assert t > np.deg2rad(45.0)  # 8 sections -> 45 deg facet steps

    def test_featureless_mesh_falls_back(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        t = adaptive_angle_threshold(mesh)
        assert np.deg2rad(15) <= t <= np.deg2rad(60)

    def test_coarse_cylinder_reconstructs_by_default(self):
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=8)
        result = snap_report(reconstruct(mesh))
        rep = result.report
        assert rep.kinds() == ["cylinder", "plane", "plane"]
        # vertices lie exactly on r=1: snapping must land it
        assert rep.by_kind("cylinder")[0].fit.primitive.radius == 1.0

    def test_mixed_tessellation_part(self):
        # coarse cylinder next to a fine box in one mesh: the global
        # adaptive threshold must serve both
        cyl = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=10)
        box = trimesh.creation.box(extents=[2, 2, 2])
        box.apply_translation([5.0, 0, 0])
        mesh = trimesh.util.concatenate([cyl, box])
        rep = reconstruct(mesh)
        assert sorted(rep.kinds()) == ["cylinder"] + ["plane"] * 8
        assert rep.coverage == 1.0


# ------------------------------------------------------------ 3. noise

class TestRobustFitting:
    def _noisy_outlier_cylinder(self):
        pts, normals, truth = _sample_cyl()
        pts = pts + 0.005 * RNG.normal(size=pts.shape)
        # gross outliers: 8 points shoved far off the surface
        idx = RNG.choice(len(pts), 8, replace=False)
        pts[idx] += 0.4 * normals[idx]
        return pts, normals, truth

    def test_trimmed_fit_ignores_outliers(self):
        pts, normals, truth = self._noisy_outlier_cylinder()
        fit = fit_cylinder(pts, normals)
        assert np.isclose(fit.primitive.radius, truth.radius, atol=0.01)

    def test_trimmed_beats_untrimmed(self):
        pts, normals, truth = self._noisy_outlier_cylinder()
        robust = fit_cylinder(pts, normals, trim=True)
        naive = fit_cylinder(pts, normals, trim=False)
        assert (abs(robust.primitive.radius - truth.radius)
                < abs(naive.primitive.radius - truth.radius))

    def test_exact_data_unaffected_by_trimming(self):
        pts, normals, truth = _sample_cyl()
        fit = fit_cylinder(pts, normals, trim=True)
        assert fit.rms < 1e-9
        assert np.isclose(fit.primitive.radius, truth.radius, atol=1e-8)

    def test_noisy_scanlike_reconstruction(self):
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        mesh.vertices = mesh.vertices + 0.004 * RNG.normal(size=mesh.vertices.shape)
        rep = reconstruct(mesh, accept_rms=0.05)
        cyls = rep.by_kind("cylinder")
        assert len(cyls) == 1
        assert np.isclose(cyls[0].fit.primitive.radius, 1.0, atol=0.02)


def _sample_cyl(n=400):
    from meshtofeatures.primitives import Cylinder
    point = np.array([0.2, -0.3, 0.5])
    axis = np.array([1.0, 0.5, 2.0]); axis /= np.linalg.norm(axis)
    u = np.cross(axis, [0, 0, 1.0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = RNG.uniform(0, 2 * np.pi, n)
    h = RNG.uniform(-2, 2, n)
    radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    pts = point + 1.0 * radial + h[:, None] * axis
    return pts, radial, Cylinder(point=point, axis=axis, radius=1.0)


# ------------------------------------------------------------ 4. fillets

class TestTangentFillets:
    def test_rounded_slab_fully_recognized(self):
        result = snap_report(reconstruct(rounded_slab()))
        rep = result.report
        # 2 z-planes + 4 side planes + 4 quarter-cylinder fillets
        assert rep.kinds() == ["cylinder"] * 4 + ["plane"] * 6
        assert rep.unrecognized == []
        assert rep.coverage == 1.0

    def test_fillet_radii_exact(self):
        result = snap_report(reconstruct(rounded_slab()))
        radii = [s.fit.primitive.radius
                 for s in result.report.by_kind("cylinder")]
        assert radii == [3.0] * 4

    def test_fillets_are_vertical_quarter_arcs(self):
        from meshtofeatures.emission import plan_patches
        result = snap_report(reconstruct(rounded_slab()))
        for p in plan_patches(result.report):
            if p.kind == "cylinder":
                assert abs(p.z_dir @ Z) > 1.0 - 1e-9
                u0, u1 = p.u_range
                assert not p.full_u
                assert np.isclose(u1 - u0, np.pi / 2, atol=np.deg2rad(10))

    def test_pristine_parts_untouched_by_second_pass(self):
        # the curvature pass must only fire on FAILED segments: a clean
        # composite must reconstruct identically to the v0.4 contract
        plate = trimesh.creation.box(extents=[40.0, 30.0, 5.0])
        drill = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=64)
        drill.apply_translation([10.0, 5.0, 0.0])
        rep = reconstruct(plate.difference(drill))
        assert rep.kinds() == ["cylinder"] + ["plane"] * 6
        assert rep.coverage == 1.0


# ------------------------------------------------------ 5. evidence rules

class TestEvidenceRules:
    def test_single_triangle_is_not_a_recognized_plane(self):
        # 3 points define a plane exactly: interpolation, not evidence
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0.0]],
                               faces=[[0, 1, 2]])
        rep = reconstruct(mesh)
        assert rep.surfaces == []
        assert len(rep.unrecognized) == 1

    def test_two_triangles_are_sufficient_for_a_plane(self):
        # 4 points, 3 dof: minimally overdetermined -- the rounded slab's
        # side faces depend on this staying legal
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]],
            faces=[[0, 1, 2], [1, 3, 2]])
        rep = reconstruct(mesh)
        assert rep.kinds() == ["plane"]
