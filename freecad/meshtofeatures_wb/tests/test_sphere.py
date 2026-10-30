# SPDX-License-Identifier: LGPL-2.1-or-later
"""Spherical feature (dome / socket) tests, written before the implementation.

A full-revolution spherical CAP: a convex cap standing on a face is a
spherical BOSS (dome), a concave cap cut into a face is a spherical SEAT
(socket). The cap's rim (its low-elevation end) opens at a face; the apex
(high-elevation end) is the cap pole (a smooth apex / bottom) or a flat
truncation (a dome may carry a flat top).

Rebuilt as a `PartDesign::Revolution` (dome, additive) or
`PartDesign::Groove` (socket, subtractive) of a circular-arc profile; the
headless rebuild unions / differences a spherical cap solid. The rim leg is
sunk a hair off the coincident face, the apex placed exactly (the cone
boss/pocket doctrine).

The pre-existing gap this suite pins: the core FITS spheres (coverage 1.0)
but the planner/executor had no spherical path, so a dome or socket was left
unassigned and the terrace machinery substituted a straight-walled pad or
pocket of the rim radius -- the right footprint at the wrong profile,
reported as a clean reconstruction (the cone_boss gap, for spheres).
"""

import math

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from freecad.meshtofeatures_wb.core.emission import plan_patches
from freecad.meshtofeatures_wb.core.features import detect_features
from freecad.meshtofeatures_wb.core.history import plan_history
from freecad.meshtofeatures_wb.core.patterns import detect_patterns
from freecad.meshtofeatures_wb.core.pipeline import reconstruct
from freecad.meshtofeatures_wb.core.primitives import Sphere
from freecad.meshtofeatures_wb.core.snapping import snap_report

from .test_adversarial import ROT, assert_geometry_match

SECTIONS = 64


# --------------------------------------------------------------- mesh builders

def _plate(ext, t, center=(0.0, 0.0)):
    p = trimesh.creation.box(extents=[ext[0], ext[1], t])
    p.apply_translation([center[0], center[1], 0.0])   # z in [-t/2, t/2]
    return p


def _placed_sphere(R, center):
    s = trimesh.creation.uv_sphere(radius=R, count=[SECTIONS, SECTIONS // 2])
    s.apply_translation(center)
    return s


def dome_plate(radius, rim_radius, top_radius=0.0, t=10.0,
               ext=(50.0, 40.0), center=(0.0, 0.0)):
    """Plate with a convex spherical cap (dome) standing on the TOP face.

    The rim (radius ``rim_radius``) lies on the top face; the cap runs to a
    smooth pole (``top_radius`` = 0) or a flat truncation (``top_radius`` > 0).
    The sphere centre sits below the top face, so the cap grows UP out of the
    material.
    """
    R = float(radius)
    a_lo = math.sqrt(max(R * R - rim_radius * rim_radius, 0.0))
    a_hi = math.sqrt(max(R * R - top_radius * top_radius, 0.0))
    top = t / 2.0
    plate = _plate(ext, t, center)
    zc = top - a_lo                       # sphere centre below the top face
    sph = _placed_sphere(R, [center[0], center[1], zc])
    z_top = top + (a_hi - a_lo)           # apex (smooth) or truncation
    box = trimesh.creation.box(
        extents=[2 * R, 2 * R, z_top - top + 1e-6])
    box.apply_translation([center[0], center[1], 0.5 * (top + z_top)])
    return plate.union(sph.intersection(box))


def socket_plate(radius, rim_radius, t=10.0,
                 ext=(50.0, 40.0), center=(0.0, 0.0)):
    """Plate with a concave spherical cap (socket) cut into the TOP face,
    running to a smooth bottom. The rim (radius ``rim_radius``) lies on the
    top face; the sphere centre sits above it, so the cut runs DOWN into the
    material."""
    R = float(radius)
    a_lo = math.sqrt(max(R * R - rim_radius * rim_radius, 0.0))
    top = t / 2.0
    plate = _plate(ext, t, center)
    zc = top + a_lo                       # sphere centre above the top face
    sph = _placed_sphere(R, [center[0], center[1], zc])
    z_bot = top - (R - a_lo)              # smooth socket bottom
    box = trimesh.creation.box(
        extents=[2 * R, 2 * R, top - z_bot + 1e-6])
    box.apply_translation([center[0], center[1], 0.5 * (z_bot + top)])
    return plate.difference(sph.intersection(box))


def flat_socket_plate(radius, rim_radius, top_radius, t=10.0,
                      ext=(50.0, 40.0), center=(0.0, 0.0)):
    """A socket with a FLAT bottom (radius ``top_radius`` > 0) cut into the
    TOP face. This is a different feature from a smooth spherical seat (a
    flat-bottomed spherical counterbore) and must stay out of scope."""
    R = float(radius)
    a_lo = math.sqrt(max(R * R - rim_radius * rim_radius, 0.0))
    a_hi = math.sqrt(max(R * R - top_radius * top_radius, 0.0))
    top = t / 2.0
    plate = _plate(ext, t, center)
    zc = top + a_lo                       # sphere centre above the top face
    sph = _placed_sphere(R, [center[0], center[1], zc])
    z_bot = top - (a_hi - a_lo)           # flat truncation (above the pole)
    box = trimesh.creation.box(
        extents=[2 * R, 2 * R, top - z_bot + 1e-6])
    box.apply_translation([center[0], center[1], 0.5 * (z_bot + top)])
    return plate.difference(sph.intersection(box))


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


def _only_dome(feats):
    cs = feats.by_kind("sphere_boss")
    assert len(cs) == 1, f"expected 1 dome, got {len(cs)}"
    return cs[0]


def _only_seat(feats):
    cs = feats.by_kind("sphere_seat")
    assert len(cs) == 1, f"expected 1 socket, got {len(cs)}"
    return cs[0]


# ------------------------------------------------------------------ detection

class TestSphericalDomeDetection:
    def test_smooth_dome_recognized(self):
        _, feats, _ = _full(dome_plate(radius=10.0, rim_radius=8.0))
        d = _only_dome(feats)
        assert np.isclose(d.params["radius"], 10.0, atol=0.15)
        assert np.isclose(d.params["rim_radius"], 8.0, atol=0.15)
        assert d.params["top_radius"] < 0.3          # runs to a smooth pole
        assert np.isclose(d.params["height"], 4.0, atol=0.1)   # R - a_lo

    def test_truncated_dome_recognized(self):
        _, feats, _ = _full(dome_plate(
            radius=10.0, rim_radius=8.0, top_radius=6.0))
        d = _only_dome(feats)
        assert np.isclose(d.params["radius"], 10.0, atol=0.15)
        assert np.isclose(d.params["rim_radius"], 8.0, atol=0.15)
        assert np.isclose(d.params["top_radius"], 6.0, atol=0.15)
        assert np.isclose(d.params["height"], 2.0, atol=0.1)   # a_hi - a_lo

    def test_not_mistaken_for_a_seat(self):
        _, feats, _ = _full(dome_plate(radius=10.0, rim_radius=8.0))
        assert feats.by_kind("sphere_seat") == []

    def test_sphere_consumed(self):
        report, feats, _ = _full(dome_plate(radius=10.0, rim_radius=8.0))
        for i in feats.unassigned:
            assert not isinstance(report.surfaces[i].fit.primitive, Sphere), \
                "sphere left unassigned"

    def test_axis_points_to_the_apex(self):
        # the feature axis is centre -> apex: for a dome on the top face that
        # is straight UP, out of the material
        _, feats, _ = _full(dome_plate(radius=10.0, rim_radius=8.0))
        d = _only_dome(feats)
        axis = np.asarray(d.params["axis"], dtype=float)
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
        assert axis[2] > 0.999

    def test_offcenter_position(self):
        _, feats, _ = _full(dome_plate(
            radius=10.0, rim_radius=8.0, center=(10.0, -7.0)))
        d = _only_dome(feats)
        pos = np.asarray(d.params["position"])
        assert np.isclose(pos[0], 10.0, atol=0.15)
        assert np.isclose(pos[1], -7.0, atol=0.15)

    def test_rotated_part_still_recognized(self):
        mesh = dome_plate(radius=10.0, rim_radius=8.0).copy()
        mesh.apply_transform(ROT)
        _, feats, _ = _full(mesh)
        d = _only_dome(feats)
        assert np.isclose(d.params["radius"], 10.0, atol=0.25)

    @pytest.mark.parametrize("radius,rim_radius,top_radius", [
        (12.0, 9.0, 0.0), (8.0, 6.0, 0.0), (10.0, 7.0, 4.0)])
    def test_dimension_variants(self, radius, rim_radius, top_radius):
        a_lo = math.sqrt(radius * radius - rim_radius * rim_radius)
        a_hi = math.sqrt(radius * radius - top_radius * top_radius)
        _, feats, _ = _full(dome_plate(
            radius=radius, rim_radius=rim_radius, top_radius=top_radius,
            t=a_hi - a_lo + 8.0))
        d = _only_dome(feats)
        assert np.isclose(d.params["radius"], radius, atol=0.2)
        assert np.isclose(d.params["rim_radius"], rim_radius, atol=0.2)
        assert np.isclose(d.params["top_radius"], top_radius, atol=0.2)
        assert np.isclose(d.params["height"], a_hi - a_lo, atol=0.15)


class TestSphericalSocketDetection:
    def test_socket_recognized(self):
        _, feats, _ = _full(socket_plate(radius=10.0, rim_radius=8.0))
        s = _only_seat(feats)
        assert np.isclose(s.params["radius"], 10.0, atol=0.15)
        assert np.isclose(s.params["rim_radius"], 8.0, atol=0.15)
        assert s.params["top_radius"] < 0.3          # smooth bottom
        assert np.isclose(s.params["height"], 4.0, atol=0.1)   # depth R - a_lo

    def test_not_mistaken_for_a_dome(self):
        _, feats, _ = _full(socket_plate(radius=10.0, rim_radius=8.0))
        assert feats.by_kind("sphere_boss") == []

    def test_socket_axis_points_into_the_cut(self):
        # a socket's centre is above the face; centre -> apex (the bottom)
        # runs DOWN, into the material
        _, feats, _ = _full(socket_plate(radius=10.0, rim_radius=8.0))
        s = _only_seat(feats)
        axis = np.asarray(s.params["axis"], dtype=float)
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
        assert axis[2] < -0.999


class TestSphericalNegatives:
    def test_cylindrical_boss_is_not_a_dome(self):
        # a constant-radius boss is a plain (cylindrical) boss, not spherical
        plate = _plate((50.0, 40.0), 10.0)
        boss = trimesh.creation.cylinder(radius=6.0, height=8.0,
                                         sections=SECTIONS)
        boss.apply_translation([0.0, 0.0, 9.0])          # z in [5, 13]
        _, feats, _ = _full(plate.union(boss))
        assert feats.by_kind("sphere_boss") == []
        assert feats.by_kind("sphere_seat") == []
        assert len(feats.by_kind("boss")) == 1

    def test_flat_bottomed_socket_out_of_scope(self):
        # a socket with a flat bottom is a different feature, not a smooth
        # spherical seat: it must not be reported as a sphere_seat
        _, feats, _ = _full(flat_socket_plate(
            radius=10.0, rim_radius=8.0, top_radius=6.0))
        assert feats.by_kind("sphere_seat") == []
        assert feats.by_kind("sphere_boss") == []

    def test_full_ball_is_not_a_cap(self):
        # a closed sphere has no rim opening at a face: not a cap
        ball = _placed_sphere(8.0, [0.0, 0.0, 0.0])
        report = snap_report(reconstruct(ball)).report
        patches = plan_patches(report)
        feats = detect_features(report, patches)
        assert feats.by_kind("sphere_boss") == []
        assert feats.by_kind("sphere_seat") == []


# -------------------------------------------------------------------- planning

class TestSphericalPlanning:
    def test_dome_planned_as_an_additive_sphere_op(self):
        _, _, plan = _full(dome_plate(radius=10.0, rim_radius=8.0))
        assert len(plan.spheres) == 1
        assert plan.unplanned == []
        op = plan.spheres[0]
        assert np.isclose(op.radius, 10.0, atol=0.15)
        assert np.isclose(op.rim_radius, 8.0, atol=0.15)
        assert op.top_radius < 0.3
        assert np.isclose(op.depth, 4.0, atol=0.1)
        assert op.from_top is True
        assert op.additive is True

    def test_truncated_dome_planned(self):
        _, _, plan = _full(dome_plate(
            radius=10.0, rim_radius=8.0, top_radius=6.0))
        assert len(plan.spheres) == 1
        op = plan.spheres[0]
        assert np.isclose(op.top_radius, 6.0, atol=0.15)
        assert np.isclose(op.depth, 2.0, atol=0.1)
        assert op.additive is True

    def test_socket_planned_as_a_subtractive_sphere_op(self):
        _, _, plan = _full(socket_plate(radius=10.0, rim_radius=8.0))
        assert len(plan.spheres) == 1
        op = plan.spheres[0]
        assert np.isclose(op.radius, 10.0, atol=0.15)
        assert np.isclose(op.rim_radius, 8.0, atol=0.15)
        assert op.top_radius < 0.3
        assert np.isclose(op.depth, 4.0, atol=0.1)
        assert op.from_top is True
        assert op.additive is False

    def test_no_spurious_terrace_pad(self):
        # the cap (and a truncated dome's flat top) must not also become a
        # straight-walled terrace pad
        _, _, plan = _full(dome_plate(
            radius=10.0, rim_radius=8.0, top_radius=6.0))
        assert plan.pads == []

    def test_bottom_face_dome(self):
        mesh = dome_plate(radius=10.0, rim_radius=8.0).copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert len(plan.spheres) == 1
        op = plan.spheres[0]
        assert op.from_top is False
        assert op.additive is True
        assert np.isclose(op.rim_radius, 8.0, atol=0.2)


# ------------------------------------------------------------------ round-trip

class TestSphericalRoundtrip:
    def test_smooth_dome_roundtrip(self):
        mesh = dome_plate(radius=10.0, rim_radius=8.0)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_truncated_dome_roundtrip(self):
        mesh = dome_plate(radius=10.0, rim_radius=8.0, top_radius=6.0)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_socket_roundtrip(self):
        mesh = socket_plate(radius=10.0, rim_radius=8.0)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_rotated_roundtrip(self):
        mesh = dome_plate(radius=10.0, rim_radius=8.0).copy()
        mesh.apply_transform(ROT)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_bottom_face_roundtrip(self):
        mesh = dome_plate(radius=10.0, rim_radius=8.0).copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)
