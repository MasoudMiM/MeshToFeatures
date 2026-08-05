# SPDX-License-Identifier: LGPL-2.1-or-later
"""Standalone conical pocket tests, written before the implementation.

A conical pocket is a concave full-revolution CONE that is NOT capped by a
coaxial drill (that would be a countersink -- see test_countersink.py) and
NOT capped by a coaxial bore (a counterdrill -- test_counterdrill.py). It
opens at a face with its wide rim and ends either at a flat floor disk, at
its own apex (a pointed recess), or at the far face (a tapered through
hole).

Rebuilt as a `PartDesign::SubtractiveCone` primitive: Radius1 at the base,
Radius2 at the top, Height along the axis. A placed primitive is used in
preference to a tapered pocket because it carries no taper-sign convention
to get wrong, and in preference to a sketch-and-revolve because there is no
profile wire to close.

The pre-existing failure this suite pins: the cone was fitted (coverage
1.0) and then silently dropped, and the terrace machinery substituted a
STRAIGHT-WALLED pocket built from the floor loop -- so an r8->r3 x 5 deep
recess was rebuilt as an r3 x 5 cylinder, removing ~28% of the right
volume, with nothing reported as unplanned.
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

def _cone_tool(r_mouth, r_far, mouth_z, far_z, sections=SECTIONS):
    """A revolved frustum cavity: r_mouth at ``mouth_z``, r_far at
    ``far_z``. r_far = 0 gives a pointed recess."""
    from shapely.geometry import Polygon
    prof = [(0.0, far_z)]
    if r_far > 0:
        prof.append((r_far, far_z))
    prof += [(r_mouth, mouth_z), (0.0, mouth_z)]
    return trimesh.creation.revolve(Polygon(prof).exterior.coords,
                                    sections=sections)


def conical_pocket_plate(r_mouth=8.0, r_far=3.0, depth=5.0, t=12.0,
                         ext=(50.0, 40.0), center=(0.0, 0.0)):
    """Plate (z in [-t/2, t/2]) with a truncated conical recess in the top."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    top = t / 2.0
    tool = _cone_tool(r_mouth, r_far, top, top - depth)
    tool.apply_translation([center[0], center[1], 0.0])
    return plate.difference(tool)


def pointed_pocket_plate(r_mouth=7.0, depth=6.0, t=14.0, ext=(50.0, 40.0)):
    """Conical recess running to a point (no floor disk)."""
    return conical_pocket_plate(r_mouth=r_mouth, r_far=0.0, depth=depth, t=t,
                                ext=ext)


def tapered_through_plate(r_mouth=8.0, r_far=3.0, t=10.0, ext=(50.0, 40.0)):
    """A tapered THROUGH hole: wide at the top face, narrow at the bottom."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    # extend a hair past both faces so the cut is clean at each end
    top, bot = t / 2.0, -t / 2.0
    slope = (r_mouth - r_far) / t
    tool = _cone_tool(r_mouth + slope * 1.0, r_far - slope * 1.0,
                      top + 1.0, bot - 1.0)
    return plate.difference(tool)


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


def _only_cone(feats):
    cs = feats.by_kind("cone_pocket")
    assert len(cs) == 1, f"expected 1 conical pocket, got {len(cs)}"
    return cs[0]


# ------------------------------------------------------------------ detection

class TestConicalPocketDetection:
    def test_truncated_pocket_recognized(self):
        _, feats, _ = _full(conical_pocket_plate())
        c = _only_cone(feats)
        assert np.isclose(c.params["mouth_radius"], 8.0, atol=0.15)
        assert np.isclose(c.params["far_radius"], 3.0, atol=0.15)
        assert np.isclose(c.params["depth"], 5.0, atol=0.1)
        assert c.params["through"] is False

    def test_pointed_pocket_recognized(self):
        _, feats, _ = _full(pointed_pocket_plate())
        c = _only_cone(feats)
        assert np.isclose(c.params["mouth_radius"], 7.0, atol=0.15)
        assert c.params["far_radius"] < 0.3          # runs to a point
        assert np.isclose(c.params["depth"], 6.0, atol=0.15)

    def test_tapered_through_hole_recognized(self):
        _, feats, _ = _full(tapered_through_plate())
        c = _only_cone(feats)
        assert c.params["through"] is True
        assert np.isclose(c.params["mouth_radius"], 8.0, atol=0.2)
        assert np.isclose(c.params["far_radius"], 3.0, atol=0.2)

    def test_cone_and_floor_consumed(self):
        report, feats, _ = _full(conical_pocket_plate())
        for i in feats.unassigned:
            assert not isinstance(report.surfaces[i].fit.primitive, Cone), \
                "cone left unassigned"

    def test_axis_points_into_the_material(self):
        _, feats, _ = _full(conical_pocket_plate())
        c = _only_cone(feats)
        axis = np.asarray(c.params["axis"], dtype=float)
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
        assert axis[2] < -0.999                     # mouth on top -> points down

    def test_offcenter_position(self):
        _, feats, _ = _full(conical_pocket_plate(center=(10.0, -7.0)))
        c = _only_cone(feats)
        pos = np.asarray(c.params["position"])
        assert np.isclose(pos[0], 10.0, atol=0.15)
        assert np.isclose(pos[1], -7.0, atol=0.15)

    def test_rotated_part_still_recognized(self):
        mesh = conical_pocket_plate().copy()
        mesh.apply_transform(ROT)
        _, feats, _ = _full(mesh)
        c = _only_cone(feats)
        assert np.isclose(c.params["mouth_radius"], 8.0, atol=0.25)

    @pytest.mark.parametrize("r_mouth,r_far,depth", [
        (6.0, 2.0, 4.0), (10.0, 6.0, 3.0), (5.0, 1.0, 6.0)])
    def test_dimension_variants(self, r_mouth, r_far, depth):
        _, feats, _ = _full(conical_pocket_plate(
            r_mouth=r_mouth, r_far=r_far, depth=depth, t=depth + 6.0))
        c = _only_cone(feats)
        assert np.isclose(c.params["mouth_radius"], r_mouth, atol=0.2)
        assert np.isclose(c.params["far_radius"], r_far, atol=0.2)
        assert np.isclose(c.params["depth"], depth, atol=0.15)


# ------------------------------------------------------- negative / robustness

class TestConicalPocketNegatives:
    def test_countersink_is_not_a_conical_pocket(self):
        from .test_countersink import countersunk_through_plate
        _, feats, _ = _full(countersunk_through_plate())
        assert feats.by_kind("cone_pocket") == []
        assert len(feats.by_kind("hole")) == 1

    def test_counterdrill_is_not_a_conical_pocket(self):
        from .test_counterdrill import counterdrilled_through_plate
        _, feats, _ = _full(counterdrilled_through_plate())
        assert feats.by_kind("cone_pocket") == []

    def test_plain_pocket_is_not_conical(self):
        plate = trimesh.creation.box(extents=[50.0, 40.0, 12.0])
        cut = trimesh.creation.box(extents=[20.0, 14.0, 6.0])
        cut.apply_translation([0.0, 0.0, 6.0])
        _, feats, _ = _full(plate.difference(cut))
        assert feats.by_kind("cone_pocket") == []

    def test_convex_cone_is_not_a_pocket(self):
        # a tapered BOSS is additive: out of scope for this rule, and it must
        # not be mistaken for a recess
        plate = trimesh.creation.box(extents=[50.0, 40.0, 10.0])
        from shapely.geometry import Polygon
        prof = [(0.0, 5.0), (9.0, 5.0), (5.0, 11.0), (0.0, 11.0)]
        boss = trimesh.creation.revolve(Polygon(prof).exterior.coords,
                                        sections=SECTIONS)
        _, feats, _ = _full(plate.union(boss))
        assert feats.by_kind("cone_pocket") == []


# -------------------------------------------------------------------- planning

class TestConicalPocketPlanning:
    def test_planned_as_a_cone_op(self):
        _, _, plan = _full(conical_pocket_plate())
        assert len(plan.cones) == 1
        assert plan.unplanned == []
        op = plan.cones[0]
        assert np.isclose(op.r_mouth, 8.0, atol=0.15)
        assert np.isclose(op.r_far, 3.0, atol=0.15)
        assert np.isclose(op.depth, 5.0, atol=0.1)
        assert op.from_top is True
        assert op.additive is False

    def test_no_spurious_terrace_pocket(self):
        # the floor disk must not also become a straight-walled terrace
        _, _, plan = _full(conical_pocket_plate())
        assert plan.pockets == []

    def test_sketch_plane_is_the_mouth_face(self):
        _, _, plan = _full(conical_pocket_plate(t=12.0))
        assert np.isclose(plan.cones[0].surface_z, 12.0, atol=0.15)

    def test_bottom_face_pocket(self):
        mesh = conical_pocket_plate().copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert len(plan.cones) == 1
        assert plan.cones[0].from_top is False
        assert np.isclose(plan.cones[0].r_mouth, 8.0, atol=0.2)

    def test_conical_pocket_alongside_a_hole(self):
        mesh = conical_pocket_plate(center=(-12.0, 0.0))
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([14.0, 8.0, 0.0])
        _, _, plan = _full(mesh.difference(drill))
        assert len(plan.cones) == 1
        assert len(plan.holes) == 1
        assert plan.unplanned == []


# ------------------------------------------------------------------ round-trip

class TestConicalPocketRoundtrip:
    def test_truncated_roundtrip(self):
        mesh = conical_pocket_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_pointed_roundtrip(self):
        mesh = pointed_pocket_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_through_roundtrip(self):
        mesh = tapered_through_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_rotated_roundtrip(self):
        mesh = conical_pocket_plate().copy()
        mesh.apply_transform(ROT)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_bottom_face_roundtrip(self):
        mesh = conical_pocket_plate().copy()
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_alongside_hole_roundtrip(self):
        mesh = conical_pocket_plate(center=(-12.0, 0.0))
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([14.0, 8.0, 0.0])
        mesh = mesh.difference(drill)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)
