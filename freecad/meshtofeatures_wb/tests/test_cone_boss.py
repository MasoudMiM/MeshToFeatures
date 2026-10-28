# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tapered-boss (additive cone) tests, written before the implementation.

A tapered boss is the convex mirror image of a conical pocket: a
full-revolution CONE whose WIDE end (the base) opens at a face and whose
NARROW end (the tip) is either capped by a disk or runs to a point. It is a
conical stud / draft-angled boss, additive material standing on a face.

Rebuilt as a `PartDesign::AdditiveCone` primitive: the base (Radius1, the
wide end) sits on the face it stands on and the feature grows OUT of the
material. The base leg is sunk a hair into the face (off the coincident
boolean), the tip is placed exactly (it is the outermost point).

The pre-existing gap this suite pins: the core FITS convex cones (coverage
1.0) but the planner/executor had no additive path, so a tapered boss was
left unassigned and the terrace machinery either dropped it or substituted
a straight-walled pad of the base radius -- the right footprint at the wrong
profile, reported as a clean reconstruction.
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
from freecad.meshtofeatures_wb.core.primitives import Cone
from freecad.meshtofeatures_wb.core.snapping import snap_report

from .test_adversarial import ROT, assert_geometry_match

SECTIONS = 64


# --------------------------------------------------------------- mesh builders

def tapered_boss_plate(r_base=9.0, r_tip=5.0, height=6.0, t=10.0,
                       ext=(50.0, 40.0), center=(0.0, 0.0)):
    """Plate (z in [-t/2, t/2]) with a truncated tapered boss on the top:
    base radius ``r_base`` at the top face, tip radius ``r_tip`` at the top
    of the boss. The frustum is sunk 1.0 into the plate for a clean union."""
    from shapely.geometry import Polygon
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    top = t / 2.0
    slope = (r_base - r_tip) / height
    sink = 1.0
    prof = [(0.0, top - sink), (r_base + slope * sink, top - sink),
            (r_tip, top + height), (0.0, top + height)]
    boss = trimesh.creation.revolve(Polygon(prof).exterior.coords,
                                    sections=SECTIONS)
    boss.apply_translation([center[0], center[1], 0.0])
    return plate.union(boss)


def pointed_boss_plate(r_base=8.0, height=7.0, t=10.0,
                       ext=(50.0, 40.0), center=(0.0, 0.0)):
    """Conical stud running to a point (no tip cap)."""
    from shapely.geometry import Polygon
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    top = t / 2.0
    sink = 1.0
    prof = [(0.0, top - sink), (r_base + (r_base / height) * sink,
                                top - sink), (0.0, top + height)]
    boss = trimesh.creation.revolve(Polygon(prof).exterior.coords,
                                    sections=SECTIONS)
    boss.apply_translation([center[0], center[1], 0.0])
    return plate.union(boss)


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


def _only_boss(feats):
    cs = feats.by_kind("cone_boss")
    assert len(cs) == 1, f"expected 1 tapered boss, got {len(cs)}"
    return cs[0]


# ------------------------------------------------------------------ detection

class TestTaperedBossDetection:
    def test_truncated_boss_recognized(self):
        _, feats, _ = _full(tapered_boss_plate())
        c = _only_boss(feats)
        assert np.isclose(c.params["base_radius"], 9.0, atol=0.15)
        assert np.isclose(c.params["tip_radius"], 5.0, atol=0.15)
        assert np.isclose(c.params["height"], 6.0, atol=0.1)

    def test_pointed_boss_recognized(self):
        _, feats, _ = _full(pointed_boss_plate())
        c = _only_boss(feats)
        assert np.isclose(c.params["base_radius"], 8.0, atol=0.15)
        assert c.params["tip_radius"] < 0.3             # runs to a point
        assert np.isclose(c.params["height"], 7.0, atol=0.15)

    def test_not_mistaken_for_a_pocket(self):
        _, feats, _ = _full(tapered_boss_plate())
        assert feats.by_kind("cone_pocket") == []

    def test_cone_and_cap_consumed(self):
        report, feats, _ = _full(tapered_boss_plate())
        for i in feats.unassigned:
            assert not isinstance(report.surfaces[i].fit.primitive, Cone), \
                "cone left unassigned"

    def test_axis_points_into_the_material(self):
        # like the conical pocket, the feature axis is stored pointing INTO
        # the material (the boss grows the other way). Base on top -> down.
        _, feats, _ = _full(tapered_boss_plate())
        c = _only_boss(feats)
        axis = np.asarray(c.params["axis"], dtype=float)
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
        assert axis[2] < -0.999

    def test_offcenter_position(self):
        _, feats, _ = _full(tapered_boss_plate(center=(10.0, -7.0)))
        c = _only_boss(feats)
        pos = np.asarray(c.params["position"])
        assert np.isclose(pos[0], 10.0, atol=0.15)
        assert np.isclose(pos[1], -7.0, atol=0.15)

    def test_rotated_part_still_recognized(self):
        mesh = tapered_boss_plate().copy()
        mesh.apply_transform(ROT)
        _, feats, _ = _full(mesh)
        c = _only_boss(feats)
        assert np.isclose(c.params["base_radius"], 9.0, atol=0.25)

    @pytest.mark.parametrize("r_base,r_tip,height", [
        (8.0, 4.0, 5.0), (12.0, 7.0, 4.0), (6.0, 1.0, 8.0)])
    def test_dimension_variants(self, r_base, r_tip, height):
        _, feats, _ = _full(tapered_boss_plate(
            r_base=r_base, r_tip=r_tip, height=height, t=height + 6.0))
        c = _only_boss(feats)
        assert np.isclose(c.params["base_radius"], r_base, atol=0.2)
        assert np.isclose(c.params["tip_radius"], r_tip, atol=0.2)
        assert np.isclose(c.params["height"], height, atol=0.15)


# ------------------------------------------------------- negative / robustness

class TestTaperedBossNegatives:
    def test_cylindrical_boss_is_not_a_cone_boss(self):
        # a constant-radius boss is a plain (cylindrical) boss, not tapered
        plate = trimesh.creation.box(extents=[50.0, 40.0, 10.0])
        boss = trimesh.creation.cylinder(radius=6.0, height=8.0,
                                         sections=SECTIONS)
        boss.apply_translation([0.0, 0.0, 9.0])           # z in [5, 13]
        _, feats, _ = _full(plate.union(boss))
        assert feats.by_kind("cone_boss") == []
        assert len(feats.by_kind("boss")) == 1

    def test_conical_pocket_is_not_a_cone_boss(self):
        from .test_cone_pocket import conical_pocket_plate
        _, feats, _ = _full(conical_pocket_plate())
        assert feats.by_kind("cone_boss") == []


# -------------------------------------------------------------------- planning

class TestTaperedBossPlanning:
    def test_planned_as_an_additive_cone_op(self):
        _, _, plan = _full(tapered_boss_plate())
        assert len(plan.cones) == 1
        assert plan.unplanned == []
        op = plan.cones[0]
        assert np.isclose(op.r_mouth, 9.0, atol=0.15)     # base (wide)
        assert np.isclose(op.r_far, 5.0, atol=0.15)       # tip (narrow)
        assert np.isclose(op.depth, 6.0, atol=0.1)
        assert op.from_top is True
        assert op.additive is True
        assert op.through is False

    def test_pointed_planned_as_an_additive_cone_op(self):
        _, _, plan = _full(pointed_boss_plate())
        assert len(plan.cones) == 1
        assert plan.cones[0].additive is True
        assert np.isclose(plan.cones[0].r_mouth, 8.0, atol=0.15)
        assert plan.cones[0].r_far < 0.3

    def test_no_spurious_terrace_pad(self):
        # the tip cap must not also become a straight-walled terrace pad
        _, _, plan = _full(tapered_boss_plate())
        assert plan.pads == []

    def test_base_face_is_the_sketch_plane(self):
        _, _, plan = _full(tapered_boss_plate(t=10.0))
        # the boss stands on the plate top face (plan-frame z = L = 10)
        assert np.isclose(plan.cones[0].surface_z, 10.0, atol=0.15)

    def test_bottom_face_boss(self):
        mesh = tapered_boss_plate().copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert len(plan.cones) == 1
        assert plan.cones[0].from_top is False
        assert plan.cones[0].additive is True
        assert np.isclose(plan.cones[0].r_mouth, 9.0, atol=0.2)

    def test_boss_alongside_a_hole(self):
        mesh = tapered_boss_plate(center=(-12.0, 0.0))
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([14.0, 8.0, 0.0])
        _, _, plan = _full(mesh.difference(drill))
        assert len(plan.cones) == 1
        assert len(plan.holes) == 1
        assert plan.unplanned == []


# ------------------------------------------------------------------ round-trip

class TestTaperedBossRoundtrip:
    def test_truncated_roundtrip(self):
        mesh = tapered_boss_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_pointed_roundtrip(self):
        mesh = pointed_boss_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_rotated_roundtrip(self):
        mesh = tapered_boss_plate().copy()
        mesh.apply_transform(ROT)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_bottom_face_roundtrip(self):
        mesh = tapered_boss_plate().copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_alongside_hole_roundtrip(self):
        mesh = tapered_boss_plate(center=(-12.0, 0.0))
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([14.0, 8.0, 0.0])
        mesh = mesh.difference(drill)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)
