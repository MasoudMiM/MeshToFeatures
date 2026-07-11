# SPDX-License-Identifier: LGPL-2.1-or-later
"""Unit tests for primitive distance functions.

Strategy: for each primitive, construct points at *known* distances
(on-surface, offset along known directions) and check the distance
function reproduces them exactly.
"""

import numpy as np
import pytest

from meshtofeatures.primitives import Plane, Sphere, Cylinder, Cone

RNG = np.random.default_rng(42)


class TestPlane:
    def test_on_surface_is_zero(self):
        p = Plane(point=[1, 2, 3], normal=[0, 0, 2])  # non-unit normal ok
        pts = np.array([[5, -7, 3], [0, 0, 3], [1, 2, 3]])
        assert np.allclose(p.distance(pts), 0.0)

    def test_known_offset(self):
        p = Plane(point=[0, 0, 0], normal=[0, 0, 1])
        pts = np.array([[10, 10, 2.5], [-3, 1, -4.0]])
        assert np.allclose(p.distance(pts), [2.5, 4.0])

    def test_normal_is_normalized(self):
        p = Plane(point=[0, 0, 0], normal=[3, 0, 4])
        assert np.isclose(np.linalg.norm(p.normal), 1.0)


class TestSphere:
    def test_on_surface_is_zero(self):
        s = Sphere(center=[1, 1, 1], radius=2.0)
        dirs = RNG.normal(size=(50, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = s.center + 2.0 * dirs
        assert np.allclose(s.distance(pts), 0.0)

    def test_center_distance_equals_radius(self):
        s = Sphere(center=[0, 0, 0], radius=3.0)
        assert np.allclose(s.distance([[0, 0, 0]]), 3.0)

    def test_known_offsets(self):
        s = Sphere(center=[0, 0, 0], radius=1.0)
        assert np.allclose(s.distance([[0, 0, 2.0], [0.5, 0, 0]]), [1.0, 0.5])

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError):
            Sphere(center=[0, 0, 0], radius=-1.0)


class TestCylinder:
    def test_on_surface_is_zero(self):
        c = Cylinder(point=[0, 0, 0], axis=[0, 0, 1], radius=1.5)
        theta = RNG.uniform(0, 2 * np.pi, 50)
        z = RNG.uniform(-10, 10, 50)
        pts = np.column_stack([1.5 * np.cos(theta), 1.5 * np.sin(theta), z])
        assert np.allclose(c.distance(pts), 0.0)

    def test_invariance_along_axis(self):
        c = Cylinder(point=[0, 0, 0], axis=[0, 0, 1], radius=1.0)
        # same radial position, wildly different heights -> same distance
        pts = np.array([[3, 0, 0], [3, 0, 1e6]])
        d = c.distance(pts)
        assert np.allclose(d, 2.0)

    def test_axis_sign_irrelevant(self):
        pts = RNG.normal(size=(20, 3))
        c1 = Cylinder(point=[1, 2, 3], axis=[1, 1, 0], radius=2.0)
        c2 = Cylinder(point=[1, 2, 3], axis=[-1, -1, 0], radius=2.0)
        assert np.allclose(c1.distance(pts), c2.distance(pts))

    def test_oblique_axis_known_offset(self):
        c = Cylinder(point=[0, 0, 0], axis=[1, 1, 1], radius=1.0)
        # point on axis -> distance = radius
        assert np.allclose(c.distance([[2, 2, 2]]), 1.0)


class TestCone:
    def test_on_surface_is_zero(self):
        alpha = np.deg2rad(30)
        cone = Cone(apex=[0, 0, 0], axis=[0, 0, 1], half_angle=alpha)
        h = RNG.uniform(0.1, 5.0, 50)
        theta = RNG.uniform(0, 2 * np.pi, 50)
        r = h * np.tan(alpha)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), h])
        assert np.allclose(cone.distance(pts), 0.0, atol=1e-12)

    def test_point_on_axis(self):
        alpha = np.deg2rad(45)
        cone = Cone(apex=[0, 0, 0], axis=[0, 0, 1], half_angle=alpha)
        # Point on the axis at height h above the apex: in the (h, r) plane
        # it sits at (h, 0); the surface line has direction (cos a, sin a),
        # so the perpendicular distance is |0*cos(a) - h*sin(a)| = h*sin(a).
        d = cone.distance([[0, 0, 2.0]])
        assert np.allclose(d, 2.0 * np.sin(alpha))

    def test_behind_apex_measured_to_apex(self):
        cone = Cone(apex=[0, 0, 0], axis=[0, 0, 1], half_angle=np.deg2rad(10))
        # far behind the apex, nearest surface point is the apex itself
        assert np.allclose(cone.distance([[0, 0, -3.0]]), 3.0)

    def test_apex_is_zero(self):
        cone = Cone(apex=[1, 1, 1], axis=[0, 1, 0], half_angle=np.deg2rad(20))
        assert np.allclose(cone.distance([[1, 1, 1]]), 0.0)

    def test_invalid_half_angle(self):
        with pytest.raises(ValueError):
            Cone(apex=[0, 0, 0], axis=[0, 0, 1], half_angle=np.pi / 2)
