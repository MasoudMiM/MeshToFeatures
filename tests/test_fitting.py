# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fitting tests: sample points from primitives with known parameters and
verify the fitters recover them.

Ground-truth comparisons are sign/gauge invariant:
* directions compared via |dot| ~ 1,
* cylinder axis point compared by distance to the true axis line.
"""

import numpy as np
import pytest

from meshtofeatures.fitting import fit_plane, fit_sphere, fit_cylinder, fit_cone, fit_best
from meshtofeatures.primitives import Plane, Sphere, Cylinder, Cone

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- samplers

def sample_plane(n=200, noise=0.0):
    normal = np.array([1.0, 2.0, -0.5])
    normal /= np.linalg.norm(normal)
    point = np.array([0.3, -1.0, 2.0])
    # basis in plane
    u = np.cross(normal, [0, 0, 1.0]); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    ab = RNG.uniform(-3, 3, size=(n, 2))
    pts = point + ab[:, :1] * u + ab[:, 1:] * v
    pts += noise * RNG.normal(size=pts.shape)
    normals = np.tile(normal, (n, 1))
    return pts, normals, Plane(point=point, normal=normal)


def sample_sphere(n=300, noise=0.0, cap=None):
    center = np.array([1.0, -2.0, 0.5])
    radius = 2.5
    dirs = RNG.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    if cap is not None:  # restrict to a spherical cap (partial coverage)
        dirs = dirs[dirs[:, 2] > cap]
    pts = center + radius * dirs
    pts += noise * RNG.normal(size=pts.shape)
    return pts, dirs, Sphere(center=center, radius=radius)


def sample_cylinder(n=300, noise=0.0, arc=2 * np.pi):
    point = np.array([0.5, 0.5, -1.0])
    axis = np.array([1.0, 1.0, 2.0]); axis /= np.linalg.norm(axis)
    radius = 1.2
    u = np.cross(axis, [0, 0, 1.0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = RNG.uniform(0, arc, n)
    h = RNG.uniform(-2, 2, n)
    radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    pts = point + radius * radial + h[:, None] * axis
    pts += noise * RNG.normal(size=pts.shape)
    return pts, radial, Cylinder(point=point, axis=axis, radius=radius)


def sample_cone(n=400, noise=0.0):
    apex = np.array([0.0, 1.0, -0.5])
    axis = np.array([0.2, -0.3, 1.0]); axis /= np.linalg.norm(axis)
    alpha = np.deg2rad(25)
    u = np.cross(axis, [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = RNG.uniform(0, 2 * np.pi, n)
    h = RNG.uniform(0.5, 3.0, n)  # stay away from apex
    r = h * np.tan(alpha)
    radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    pts = apex + h[:, None] * axis + r[:, None] * radial
    # outward normal: radial*cos(a) - axis*sin(a)
    normals = radial * np.cos(alpha) - axis * np.sin(alpha)
    pts += noise * RNG.normal(size=pts.shape)
    return pts, normals, Cone(apex=apex, axis=axis, half_angle=alpha)


def sample_cone_frustum(alpha_deg=45.0, r_small=2.5, r_big=5.0, sections=64,
                        apex=(0.3, -0.4, 0.6), axis=(0.1, 0.2, 1.0),
                        noise=0.0, concave=False):
    """A cone frustum sampled as TWO vertex rings -- exactly the vertex
    pattern a tessellated countersink produces. Two rings lie on a common
    sphere (the classic impostor of design note 1), and the frustum spans
    only a short axial band, so the nonlinear cone fit's half-angle
    parameter can wander far along the periodic residual valley. A robust
    fitter must still recover the true half-angle.

    ``concave=True`` flips the surface normals to point toward the axis, as
    a conical HOLE (countersink) does: then n . axis = +sin(alpha), the
    opposite sign from a convex cone. The fitter must handle both."""
    apex = np.asarray(apex, dtype=float)
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    alpha = np.deg2rad(alpha_deg)
    u = np.cross(axis, [1.0, 0.0, 0.0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.linspace(0.0, 2 * np.pi, sections, endpoint=False)
    ring = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    t = np.tan(alpha)
    sign = -1.0 if concave else 1.0
    pts, normals = [], []
    for r in (r_small, r_big):
        h = r / t
        pts.append(apex + h * axis + r * ring)
        normals.append(sign * (ring * np.cos(alpha) - axis * np.sin(alpha)))
    pts = np.vstack(pts)
    normals = np.vstack(normals)
    if noise:
        pts = pts + noise * RNG.normal(size=pts.shape)
    return pts, normals, Cone(apex=apex, axis=axis, half_angle=alpha)


def assert_same_direction(a, b, atol=1e-6):
    assert np.isclose(abs(np.dot(a, b)), 1.0, atol=atol), f"{a} vs {b}"


# ---------------------------------------------------------------- plane

class TestFitPlane:
    def test_exact_recovery(self):
        pts, _, truth = sample_plane()
        fit = fit_plane(pts)
        assert fit.rms < 1e-10
        assert_same_direction(fit.primitive.normal, truth.normal)

    def test_noisy_recovery(self):
        pts, _, truth = sample_plane(noise=0.01)
        fit = fit_plane(pts)
        assert fit.rms < 0.02
        assert_same_direction(fit.primitive.normal, truth.normal, atol=1e-3)


# ---------------------------------------------------------------- sphere

class TestFitSphere:
    def test_exact_recovery(self):
        pts, _, truth = sample_sphere()
        fit = fit_sphere(pts)
        assert fit.rms < 1e-9
        assert np.allclose(fit.primitive.center, truth.center, atol=1e-8)
        assert np.isclose(fit.primitive.radius, truth.radius, atol=1e-8)

    def test_partial_cap_recovery(self):
        # only ~30% of the sphere visible: algebraic fit alone is biased,
        # refinement must fix it
        pts, _, truth = sample_sphere(n=2000, cap=0.4)
        fit = fit_sphere(pts)
        assert np.allclose(fit.primitive.center, truth.center, atol=1e-6)
        assert np.isclose(fit.primitive.radius, truth.radius, atol=1e-6)

    def test_noisy_recovery(self):
        pts, _, truth = sample_sphere(noise=0.01)
        fit = fit_sphere(pts)
        assert np.isclose(fit.primitive.radius, truth.radius, atol=0.01)


# ---------------------------------------------------------------- cylinder

class TestFitCylinder:
    def test_exact_recovery_with_normals(self):
        pts, normals, truth = sample_cylinder()
        fit = fit_cylinder(pts, normals)
        assert fit.rms < 1e-9
        assert_same_direction(fit.primitive.axis, truth.axis, atol=1e-8)
        assert np.isclose(fit.primitive.radius, truth.radius, atol=1e-8)
        # fitted axis point must lie on the true axis
        assert truth.radial_distance([fit.primitive.point])[0] < 1e-7

    def test_exact_recovery_without_normals(self):
        pts, _, truth = sample_cylinder()
        fit = fit_cylinder(pts, normals=None)
        assert fit.rms < 1e-8
        assert np.isclose(fit.primitive.radius, truth.radius, atol=1e-6)

    def test_partial_arc(self):
        # quarter pipe: hard case for algebraic circle fit
        pts, normals, truth = sample_cylinder(n=800, arc=np.pi / 2)
        fit = fit_cylinder(pts, normals)
        assert np.isclose(fit.primitive.radius, truth.radius, atol=1e-6)
        assert_same_direction(fit.primitive.axis, truth.axis, atol=1e-6)

    def test_noisy_recovery(self):
        pts, normals, truth = sample_cylinder(noise=0.01)
        fit = fit_cylinder(pts, normals)
        assert np.isclose(fit.primitive.radius, truth.radius, atol=0.01)
        assert_same_direction(fit.primitive.axis, truth.axis, atol=1e-3)


# ---------------------------------------------------------------- cone

class TestFitCone:
    def test_exact_recovery(self):
        pts, normals, truth = sample_cone()
        fit = fit_cone(pts, normals)
        assert fit.rms < 1e-9
        assert np.allclose(fit.primitive.apex, truth.apex, atol=1e-7)
        assert np.isclose(fit.primitive.half_angle, truth.half_angle, atol=1e-8)
        # cone axis orientation matters (apex -> opening), sign must match
        assert np.dot(fit.primitive.axis, truth.axis) > 0.999999

    def test_noisy_recovery(self):
        pts, normals, truth = sample_cone(noise=0.005)
        fit = fit_cone(pts, normals)
        assert np.isclose(fit.primitive.half_angle, truth.half_angle, atol=5e-3)
        assert np.allclose(fit.primitive.apex, truth.apex, atol=0.05)

    @pytest.mark.parametrize("alpha_deg", [20.0, 30.0, 41.0, 45.0, 50.0, 60.0])
    def test_frustum_two_rings_recovers_half_angle(self, alpha_deg):
        # a short two-ring frustum (a real tessellated countersink) must not
        # let the half-angle wander to a wrapped/degenerate value
        pts, normals, truth = sample_cone_frustum(alpha_deg=alpha_deg)
        fit = fit_cone(pts, normals)
        assert isinstance(fit.primitive, Cone)
        assert np.isclose(fit.primitive.half_angle, truth.half_angle, atol=1e-4)
        # axis points apex -> opening
        assert np.dot(fit.primitive.axis, truth.axis) > 0.9999
        # the surface itself is recovered (distance to true-cone samples ~ 0)
        assert fit.rms < 1e-6

    def test_frustum_noisy(self):
        pts, normals, truth = sample_cone_frustum(alpha_deg=41.0, noise=0.003)
        fit = fit_cone(pts, normals)
        assert np.isclose(fit.primitive.half_angle, truth.half_angle, atol=5e-3)

    @pytest.mark.parametrize("alpha_deg", [30.0, 45.0, 60.0])
    def test_concave_cone_recovered(self, alpha_deg):
        # a conical HOLE: normals point toward the axis (n . axis = +sin a),
        # the opposite sign from a convex cone. The init must not assume a
        # sign; recovery must still give the true half-angle and an axis
        # pointing apex -> opening.
        pts, normals, truth = sample_cone_frustum(alpha_deg=alpha_deg,
                                                  concave=True)
        fit = fit_cone(pts, normals)
        assert np.isclose(fit.primitive.half_angle, truth.half_angle,
                          atol=1e-4)
        assert np.dot(fit.primitive.axis, truth.axis) > 0.9999
        assert fit.rms < 1e-6

    @pytest.mark.parametrize("half_angle_deg", [30.0, 41.0, 45.0, 60.0])
    def test_real_countersink_mesh_recognizes_cone(self, half_angle_deg):
        # end-to-end regression pin: a tessellated countersunk hole segments
        # into a short two-ring cone whose averaged normals send the raw
        # nonlinear fit's half-angle wandering. Before the geometry-based
        # recovery the conical wall was mis-selected as a Sphere (its two
        # rings lie on a common sphere); it must be a Cone now.
        pytest.importorskip("trimesh")
        import trimesh
        from meshtofeatures.pipeline import reconstruct
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        drill = trimesh.creation.cylinder(radius=2.5, height=30.0, sections=64)
        ha = np.deg2rad(half_angle_deg)
        h_wide = 5.0 / np.tan(ha)
        cone = trimesh.creation.cone(radius=5.0, height=h_wide, sections=64)
        cone.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        cone.apply_translation([0, 0, 5.0])          # wide rim at top face
        mesh = plate.difference(drill.union(cone))
        rep = reconstruct(mesh)
        kinds = [s.fit.primitive.kind for s in rep.surfaces]
        assert "cone" in kinds, f"no cone recognized; got {kinds}"
        cones = [s.fit.primitive for s in rep.surfaces
                 if s.fit.primitive.kind == "cone"]
        assert any(np.isclose(c.half_angle, ha, atol=np.deg2rad(1.5))
                   for c in cones)


# ---------------------------------------------------------------- selection

class TestFitBest:
    def test_plane_wins_on_plane(self):
        pts, normals, _ = sample_plane()
        assert fit_best(pts, normals).primitive.kind == "plane"

    def test_sphere_wins_on_sphere(self):
        pts, normals, _ = sample_sphere()
        assert fit_best(pts, normals).primitive.kind == "sphere"

    def test_cylinder_wins_on_cylinder(self):
        pts, normals, _ = sample_cylinder()
        assert fit_best(pts, normals).primitive.kind == "cylinder"

    def test_cone_wins_on_cone(self):
        pts, normals, _ = sample_cone()
        assert fit_best(pts, normals).primitive.kind == "cone"

    def test_cone_wins_on_two_ring_frustum(self):
        # the two rings lie on a common sphere (design note 1); dense
        # samples on the true surface must still select the cone
        pts, normals, truth = sample_cone_frustum(alpha_deg=45.0)
        # score on dense surface samples between the rings (expose the sphere)
        u = np.cross(truth.axis, [1.0, 0.0, 0.0]); u /= np.linalg.norm(u)
        v = np.cross(truth.axis, u)
        th = RNG.uniform(0, 2 * np.pi, 500)
        r = RNG.uniform(2.5, 5.0, 500)
        h = r / np.tan(truth.half_angle)
        samples = (truth.apex + h[:, None] * truth.axis
                   + r[:, None] * (np.cos(th)[:, None] * u
                                   + np.sin(th)[:, None] * v))
        assert fit_best(pts, normals,
                        score_points=samples).primitive.kind == "cone"

    def test_plane_preferred_over_giant_sphere(self):
        # noisy plane: a huge sphere fits noisy planar data almost as well;
        # simplicity gating must still choose the plane
        pts, normals, _ = sample_plane(noise=0.005)
        fit = fit_best(pts, normals, tolerance=0.02)
        assert fit.primitive.kind == "plane"
