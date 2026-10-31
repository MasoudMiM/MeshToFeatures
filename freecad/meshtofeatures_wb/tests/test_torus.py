# SPDX-License-Identifier: LGPL-2.1-or-later
"""Rounded ring band (torus) tests, written before the implementation.

A full-revolution ring band is a torus: a tube (major radius R, minor radius
r) of revolution standing on a face. A convex band growing OUT of the face it
sits on is a rounded ring BOSS (a donut blend the partial-revolution fillet
rule cannot claim -- fillets are partial revolutions, a ring band is a full
one); a band cut INTO a face is a ring SEAT (a ring groove). The band's rim
is an annulus (two concentric circles) where the tube is cut by the face; the
tube top (the outermost point) is the far end.

The core had no torus primitive at all, so a ring band was either left
unassigned or absorbed by a straight-walled terrace pad -- the right
footprint at the wrong profile (the cone_boss/sphere gap, for tori). Rebuilt
as a ``PartDesign::Revolution`` (boss) / ``PartDesign::Groove`` (seat) of a
circular-arc profile; the headless rebuild unions/differences a torus-band
solid.
"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from freecad.meshtofeatures_wb.core.emission import plan_patches
from freecad.meshtofeatures_wb.core.features import detect_features
from freecad.meshtofeatures_wb.core.history import plan_history
from freecad.meshtofeatures_wb.core.patterns import detect_patterns
from freecad.meshtofeatures_wb.core.pipeline import reconstruct
from freecad.meshtofeatures_wb.core.primitives import Torus
from freecad.meshtofeatures_wb.core.snapping import snap_report

from .test_adversarial import ROT, assert_geometry_match

# the band is a closed curve of revolution: use more sections than the
# default 64 so the fit and the boolean are both clean
SECTIONS = 96


# --------------------------------------------------------------- mesh builders

def _torus_solid(R, r, zc, sections=SECTIONS):
    """Watertight solid torus about the z axis, tube centred at height ``zc``
    (major R, minor r). A closed minor-circle profile revolved 360 deg."""
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    circ = np.column_stack([R + r * np.cos(theta), r * np.sin(theta)])
    circ = np.vstack([circ, circ[:1]])
    t = trimesh.creation.revolve(circ, sections=sections)
    t.process()
    t.apply_translation([0.0, 0.0, zc])
    return t


def rounded_ring_boss(R=10.0, r=3.0, a_center=2.0, t=10.0,
                      ext=(50.0, 40.0), center=(0.0, 0.0)):
    """Plate (z in [-t/2, t/2]) with a rounded ring boss on the top face: the
    tube centre sits ``a_center`` above the top face and the tube is sunk
    into the plate (its bottom is below the face) for a clean union."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    tor = _torus_solid(R, r, t / 2.0 + a_center)
    tor.apply_translation([center[0], center[1], 0.0])
    return plate.union(tor)


def ring_seat(R=10.0, r=3.0, a_center=2.0, t=10.0,
              ext=(50.0, 40.0), center=(0.0, 0.0)):
    """Plate (z in [-t/2, t/2]) with a ring groove (seat) cut into the top
    face: the tube centre sits ``a_center`` below the top face and is
    subtracted, leaving an annular depression."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    tor = _torus_solid(R, r, t / 2.0 - a_center)
    tor.apply_translation([center[0], center[1], 0.0])
    return plate.difference(tor)


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


def _only(feats, kind):
    cs = feats.by_kind(kind)
    assert len(cs) == 1, f"expected 1 {kind}, got {len(cs)}"
    return cs[0]


# ------------------------------------------------------------------ detection

class TestTorusBossDetection:
    def test_boss_recognized(self):
        _, feats, _ = _full(rounded_ring_boss())
        b = _only(feats, "torus_boss")
        assert np.isclose(b.params["major_radius"], 10.0, atol=0.15)
        assert np.isclose(b.params["minor_radius"], 3.0, atol=0.15)
        assert np.isclose(b.params["a_center"], 2.0, atol=0.15)
        assert np.isclose(b.params["height"], 5.0, atol=0.15)

    def test_rim_radii(self):
        # rim annulus: [R - sqrt(r^2 - a^2), R + sqrt(r^2 - a^2)]
        s = np.sqrt(3.0 ** 2 - 2.0 ** 2)
        _, feats, _ = _full(rounded_ring_boss())
        b = _only(feats, "torus_boss")
        assert np.isclose(b.params["rim_inner_radius"], 10.0 - s, atol=0.15)
        assert np.isclose(b.params["rim_outer_radius"], 10.0 + s, atol=0.15)

    def test_not_mistaken_for_a_plain_boss(self):
        _, feats, _ = _full(rounded_ring_boss())
        assert feats.by_kind("boss") == []

    def test_torus_consumed(self):
        report, feats, _ = _full(rounded_ring_boss())
        for i in feats.unassigned:
            assert not isinstance(report.surfaces[i].fit.primitive, Torus), \
                "torus left unassigned"

    def test_axis_points_out_of_the_face(self):
        # a boss grows OUT of the face it sits on: the axis (rim -> band)
        # points away from the material. Base on top -> up.
        _, feats, _ = _full(rounded_ring_boss())
        axis = np.asarray(_only(feats, "torus_boss").params["axis"])
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
        assert axis[2] > 0.999

    def test_offcenter_position(self):
        _, feats, _ = _full(rounded_ring_boss(center=(10.0, -7.0)))
        pos = np.asarray(_only(feats, "torus_boss").params["position"])
        assert np.isclose(pos[0], 10.0, atol=0.15)
        assert np.isclose(pos[1], -7.0, atol=0.15)

    def test_rotated_part_still_recognized(self):
        mesh = rounded_ring_boss().copy()
        mesh.apply_transform(ROT)
        _, feats, _ = _full(mesh)
        b = _only(feats, "torus_boss")
        assert np.isclose(b.params["major_radius"], 10.0, atol=0.25)
        assert np.isclose(b.params["minor_radius"], 3.0, atol=0.25)

    @pytest.mark.parametrize("R,r,a_center", [
        (12.0, 2.0, 1.0), (8.0, 4.0, 3.0), (15.0, 3.0, 1.5)])
    def test_dimension_variants(self, R, r, a_center):
        _, feats, _ = _full(rounded_ring_boss(
            R=R, r=r, a_center=a_center, t=a_center + r + 6.0))
        b = _only(feats, "torus_boss")
        assert np.isclose(b.params["major_radius"], R, atol=0.2)
        assert np.isclose(b.params["minor_radius"], r, atol=0.2)
        assert np.isclose(b.params["a_center"], a_center, atol=0.2)


class TestTorusSeatDetection:
    def test_seat_recognized(self):
        _, feats, _ = _full(ring_seat())
        s = _only(feats, "torus_recess")
        assert np.isclose(s.params["major_radius"], 10.0, atol=0.15)
        assert np.isclose(s.params["minor_radius"], 3.0, atol=0.15)
        assert np.isclose(s.params["a_center"], 2.0, atol=0.15)

    def test_seat_axis_points_into_the_material(self):
        # a groove is cut INTO the face: the axis (rim -> band) points away
        # from the outward normal. Base on top -> down.
        _, feats, _ = _full(ring_seat())
        axis = np.asarray(_only(feats, "torus_recess").params["axis"])
        assert axis[2] < -0.999

    def test_not_mistaken_for_a_boss(self):
        _, feats, _ = _full(ring_seat())
        assert feats.by_kind("torus_boss") == []


# ------------------------------------------------------- negative / robustness

class TestTorusNegatives:
    def test_cylindrical_boss_is_not_a_torus(self):
        # a constant-radius boss is a plain (cylindrical) boss, not a ring
        plate = trimesh.creation.box(extents=[50.0, 40.0, 10.0])
        boss = trimesh.creation.cylinder(radius=6.0, height=8.0,
                                         sections=SECTIONS)
        boss.apply_translation([0.0, 0.0, 9.0])           # z in [5, 13]
        _, feats, _ = _full(plate.union(boss))
        assert feats.by_kind("torus_boss") == []
        assert len(feats.by_kind("boss")) == 1

    def test_standalone_donut_is_not_claimed(self):
        # a full donut not fused to a face has no rim (and no planar base):
        # the band is fitted but left unassigned rather than guessed as a
        # boss. Detection only -- a base-less part has no prismatic plan.
        donut = _torus_solid(10.0, 3.0, 15.0)
        report = snap_report(reconstruct(donut)).report
        feats = detect_features(report, plan_patches(report))
        assert feats.by_kind("torus_boss") == []
        assert feats.by_kind("torus_recess") == []

    def test_slot_rounded_end_is_not_a_torus(self):
        # a stadium (slot) opening's rounded ends are half-cylinders: local
        # curved patches that do not wrap around a major axis. The flexible
        # 7-dof torus fit grabs one when the cylinder fit fails on the partial
        # circumference -- it must be rejected so the slot is still detected.
        r, a, h = 4.0, 11.0, 6.0
        tool = trimesh.creation.box(extents=[2 * a, 2 * r, h])
        for sx in (-a, a):
            c = trimesh.creation.cylinder(radius=r, height=h,
                                          sections=SECTIONS)
            c.apply_translation([sx, 0.0, 0.0])
            tool = tool.union(c)
        plate = trimesh.creation.box(extents=[60.0, 40.0, h])
        slot = plate.difference(tool)
        _, feats, _ = _full(slot)
        assert feats.by_kind("torus_boss") == []
        assert feats.by_kind("torus_recess") == []
        assert len(feats.by_kind("slot")) == 1


# -------------------------------------------------------------------- planning

class TestTorusPlanning:
    def test_planned_as_an_additive_torus_op(self):
        _, _, plan = _full(rounded_ring_boss())
        assert len(plan.tori) == 1
        assert plan.unplanned == []
        op = plan.tori[0]
        assert np.isclose(op.major_radius, 10.0, atol=0.15)
        assert np.isclose(op.minor_radius, 3.0, atol=0.15)
        assert np.isclose(op.a_center, 2.0, atol=0.15)
        assert op.from_top is True
        assert op.additive is True

    def test_seat_planned_as_a_subtractive_torus_op(self):
        _, _, plan = _full(ring_seat())
        assert len(plan.tori) == 1
        assert plan.unplanned == []
        op = plan.tori[0]
        assert op.additive is False
        assert op.from_top is True

    def test_no_spurious_terrace_pad(self):
        # the band must not also become a straight-walled terrace pad
        _, _, plan = _full(rounded_ring_boss())
        assert plan.pads == []

    def test_base_face_is_the_sketch_plane(self):
        _, _, plan = _full(rounded_ring_boss(t=10.0))
        # the ring sits on the plate top face (plan-frame z = L = 10)
        assert np.isclose(plan.tori[0].surface_z, 10.0, atol=0.15)

    def test_bottom_face_boss(self):
        mesh = rounded_ring_boss().copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert len(plan.tori) == 1
        assert plan.tori[0].from_top is False
        assert plan.tori[0].additive is True
        assert np.isclose(plan.tori[0].major_radius, 10.0, atol=0.2)

    def test_boss_alongside_a_hole(self):
        mesh = rounded_ring_boss(center=(-12.0, 0.0))
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([14.0, 8.0, 0.0])
        _, _, plan = _full(mesh.difference(drill))
        assert len(plan.tori) == 1
        assert len(plan.holes) == 1
        assert plan.unplanned == []


# ------------------------------------------------------------------ round-trip

class TestTorusRoundtrip:
    def test_boss_roundtrip(self):
        mesh = rounded_ring_boss()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_seat_roundtrip(self):
        mesh = ring_seat()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_rotated_roundtrip(self):
        mesh = rounded_ring_boss().copy()
        mesh.apply_transform(ROT)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_bottom_face_roundtrip(self):
        mesh = rounded_ring_boss().copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_alongside_hole_roundtrip(self):
        mesh = rounded_ring_boss(center=(-12.0, 0.0))
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([14.0, 8.0, 0.0])
        mesh = mesh.difference(drill)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    @pytest.mark.parametrize("R,r,a_center", [
        (12.0, 2.0, 1.0), (8.0, 4.0, 3.0)])
    def test_dimension_variants_roundtrip(self, R, r, a_center):
        mesh = rounded_ring_boss(R=R, r=r, a_center=a_center,
                                 t=a_center + r + 6.0)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)
