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

    def test_plane_preferred_over_giant_sphere(self):
        # noisy plane: a huge sphere fits noisy planar data almost as well;
        # simplicity gating must still choose the plane
        pts, normals, _ = sample_plane(noise=0.005)
        fit = fit_best(pts, normals, tolerance=0.02)
        assert fit.primitive.kind == "plane"
