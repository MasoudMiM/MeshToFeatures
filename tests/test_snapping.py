# SPDX-License-Identifier: LGPL-2.1-or-later
"""Snapping tests, written before the implementation.

Layers under test:
1. pure helpers: snap_value / snap_angle / snap_direction / clustering
2. report-level snap_report: direction unification, coaxiality, radius
   equalization, grid snapping
3. the guard: a snap that degrades the fit beyond tolerance is rejected
"""

import numpy as np
import trimesh

from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import (
    SnapConfig,
    cluster_directions,
    cluster_scalars,
    snap_angle,
    snap_direction,
    snap_report,
    snap_value,
)

Z = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------ snap_value

class TestSnapValue:
    def test_snaps_to_nice_value(self):
        assert snap_value(7.998, atol=0.01) == 8.0

    def test_prefers_coarsest_grid(self):
        # 9.999 could snap to 9.999... on a fine grid; 10 (grid 10) wins
        assert snap_value(9.999, atol=0.01) == 10.0

    def test_half_grid(self):
        assert snap_value(2.503, atol=0.01) == 2.5

    def test_negative(self):
        assert snap_value(-4.999, atol=0.01) == -5.0

    def test_out_of_tolerance_returns_none(self):
        assert snap_value(7.62, atol=0.01) is None

    def test_zero(self):
        assert snap_value(0.0004, atol=0.001) == 0.0


class TestSnapAngle:
    def test_45_degrees(self):
        a = snap_angle(np.deg2rad(44.8), atol=np.deg2rad(0.5))
        assert np.isclose(a, np.deg2rad(45.0))

    def test_out_of_tolerance(self):
        assert snap_angle(np.deg2rad(26.57), atol=np.deg2rad(0.2)) is None


class TestSnapDirection:
    def test_near_z_snaps_to_z(self):
        d = np.array([0.005, -0.003, 0.9999])
        d /= np.linalg.norm(d)
        snapped = snap_direction(d, angle_tol=np.deg2rad(1.0))
        assert np.allclose(snapped, Z)

    def test_sign_preserved(self):
        d = np.array([0.001, 0.0, -1.0])
        d /= np.linalg.norm(d)
        snapped = snap_direction(d, angle_tol=np.deg2rad(1.0))
        assert np.allclose(snapped, -Z)

    def test_diagonal_not_snapped(self):
        d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        assert snap_direction(d, angle_tol=np.deg2rad(1.0)) is None


# ------------------------------------------------------------ clustering

class TestClusterDirections:
    def test_near_parallel_directions_unified(self):
        base = np.array([1.0, 2.0, 0.5])
        base /= np.linalg.norm(base)
        tilt = np.deg2rad(0.3)
        d2 = np.array([base[0], base[1] * np.cos(tilt) - base[2] * np.sin(tilt),
                       base[1] * np.sin(tilt) + base[2] * np.cos(tilt)])
        dirs = np.array([base, d2, -base])  # anti-parallel joins the same cluster
        labels, means = cluster_directions(dirs, weights=np.ones(3),
                                           angle_tol=np.deg2rad(1.0))
        assert labels[0] == labels[1] == labels[2]
        # cluster mean is a unit vector parallel to base within the tilt
        assert abs(means[labels[0]] @ base) > np.cos(np.deg2rad(0.3))

    def test_distinct_directions_stay_apart(self):
        dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        labels, _ = cluster_directions(dirs, np.ones(3), np.deg2rad(1.0))
        assert len(set(labels)) == 3

    def test_weighting_pulls_towards_heavy_member(self):
        a = np.array([0.0, 0.0, 1.0])
        b = np.array([np.sin(np.deg2rad(0.8)), 0.0, np.cos(np.deg2rad(0.8))])
        labels, means = cluster_directions(
            np.array([a, b]), weights=np.array([1000.0, 1.0]),
            angle_tol=np.deg2rad(1.0),
        )
        assert labels[0] == labels[1]
        assert means[labels[0]] @ a > np.cos(np.deg2rad(0.01))


class TestClusterScalars:
    def test_merges_within_tolerance(self):
        labels = cluster_scalars(np.array([1.0, 1.0008, 2.0]), atol=0.01)
        assert labels[0] == labels[1] != labels[2]

    def test_chain_merging(self):
        # 1.00, 1.008, 1.016: adjacent pairs within tol -> single cluster
        labels = cluster_scalars(np.array([1.0, 1.008, 1.016]), atol=0.01)
        assert len(set(labels)) == 1


# ------------------------------------------------------------ report level

def _tilted_noisy_cylinder(radius=0.9995, tilt_deg=0.3, sigma=0.0015, seed=1):
    mesh = trimesh.creation.cylinder(radius=radius, height=3.0, sections=64)
    rot = trimesh.transformations.rotation_matrix(np.deg2rad(tilt_deg), [1, 0, 0])
    mesh.apply_transform(rot)
    rng = np.random.default_rng(seed)
    mesh.vertices = mesh.vertices + rng.normal(scale=sigma, size=mesh.vertices.shape)
    return mesh


class TestSnapReport:
    def test_cylinder_axis_and_radius_snap(self):
        rep = reconstruct(_tilted_noisy_cylinder(), accept_rms=0.05)
        result = snap_report(rep)
        cyl = result.report.by_kind("cylinder")[0].fit.primitive
        assert np.isclose(abs(cyl.axis @ Z), 1.0)          # exactly canonical
        assert cyl.radius == 1.0                            # exactly on grid
        # caps became exactly perpendicular to the snapped axis
        for s in result.report.by_kind("plane"):
            assert np.isclose(abs(s.fit.primitive.normal @ Z), 1.0)

    def test_actions_are_logged(self):
        rep = reconstruct(_tilted_noisy_cylinder(), accept_rms=0.05)
        result = snap_report(rep)
        assert any(a.accepted for a in result.actions)
        assert all(isinstance(a.detail, str) and a.detail for a in result.actions)

    def test_original_report_not_mutated(self):
        rep = reconstruct(_tilted_noisy_cylinder(), accept_rms=0.05)
        before = rep.by_kind("cylinder")[0].fit.primitive.radius
        snap_report(rep)
        after = rep.by_kind("cylinder")[0].fit.primitive.radius
        assert before == after != 1.0

    def test_cone_half_angle_not_snapped(self):
        # arctan(1/2) = 26.565 deg is NOT a nice angle. Under the
        # informativeness rule no grid coarse enough to be meaningful at
        # this tolerance contains a value within tolerance, so no snap is
        # proposed and the true half-angle survives untouched.
        mesh = trimesh.creation.cone(radius=1.0, height=2.0, sections=64)
        rep = reconstruct(mesh)
        config = SnapConfig(angle_tol=np.deg2rad(1.0))
        result = snap_report(rep, config)
        cone = result.report.by_kind("cone")[0].fit.primitive
        assert np.isclose(cone.half_angle, np.arctan2(1.0, 2.0), atol=1e-6)

    def test_guard_rejects_out_of_band_radius(self):
        # radius 0.97 with a sloppy value tolerance: grid proposes 1.0,
        # but that moves the surface by 0.03 -- guard must veto and log
        # the rejection.
        mesh = trimesh.creation.cylinder(radius=0.97, height=3.0, sections=64)
        rep = reconstruct(mesh)
        config = SnapConfig(value_atol=0.05, max_extra_rms=0.005)
        result = snap_report(rep, config)
        cyl = result.report.by_kind("cylinder")[0].fit.primitive
        assert np.isclose(cyl.radius, 0.97, atol=1e-6)
        assert any(not a.accepted for a in result.actions)

    def test_snapped_fits_are_rescored(self):
        rep = reconstruct(_tilted_noisy_cylinder(), accept_rms=0.05)
        result = snap_report(rep)
        for s in result.report.surfaces:
            d = s.fit.primitive.distance(s.segment.samples)
            assert np.isclose(s.fit.rms, float(np.sqrt(np.mean(d * d))))


class TestRelations:
    def test_coaxial_counterbore(self):
        # two coaxial cylinders of different radius, slightly perturbed:
        # snapping must make them *exactly* coaxial
        c1 = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
        c2 = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=64)
        c2.apply_translation([0.0008, -0.0005, 1.0])  # nearly coaxial, stacked
        mesh = trimesh.util.concatenate([c1, c2])
        rep = reconstruct(mesh)
        result = snap_report(rep)
        cyls = [s.fit.primitive for s in result.report.by_kind("cylinder")]
        assert len(cyls) == 2
        assert np.isclose(abs(cyls[0].axis @ cyls[1].axis), 1.0)
        # axis lines coincide: each axis point lies on the other axis
        assert cyls[0].radial_distance([cyls[1].point])[0] < 1e-12

    def test_equal_radii_unified(self):
        # two separate cylinders with radii 1.0 and 1.0006: equalized
        c1 = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
        c2 = trimesh.creation.cylinder(radius=1.0006, height=1.0, sections=64)
        c2.apply_translation([5.0, 0, 0])
        mesh = trimesh.util.concatenate([c1, c2])
        rep = reconstruct(mesh)
        result = snap_report(rep)
        cyls = [s.fit.primitive for s in result.report.by_kind("cylinder")]
        assert cyls[0].radius == cyls[1].radius == 1.0
