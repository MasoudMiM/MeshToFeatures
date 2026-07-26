# SPDX-License-Identifier: LGPL-2.1-or-later
"""Counterdrill tests, written before the implementation.

A counterdrilled hole is a countersink AND a counterbore on one hole: from
the mouth down, a cylindrical recess (the counterbore), then a conical
transition tapering from the bore diameter to the drill diameter, then the
drill. PartDesign::Hole models it natively as
``HoleCutType = "Counterdrill"``: ``HoleCutDiameter`` is the bore diameter,
``HoleCutDepth`` the CYLINDRICAL part of the cut, and
``HoleCutCountersinkAngle`` the cone's included angle.

Two facts distinguish it from the shapes already recognized:

* unlike a plain counterbore, there is **no annular shoulder plane** -- the
  bore wall runs straight into the cone, so the shoulder test in the
  coaxial-stack pass cannot fire;
* unlike a plain countersink, the cone's wide rim does **not** open at a
  face -- it opens at the bore floor, so the mouth (and hence the machining
  side and the sketch plane) is the bore's far end, not the cone's.

The pre-existing failure this suite pins: the countersink pass claimed the
cone + drill and stripped the drill from the concave list, leaving the bore
cylinder to be re-emitted as a spurious blind hole of the bore diameter,
with the countersink's mouth at the bore floor.
"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.history import HoleOp, hole_op_properties, plan_history
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.primitives import Cone, Cylinder
from meshtofeatures.snapping import snap_report

from .test_adversarial import ROT, assert_geometry_match

SECTIONS = 64


# --------------------------------------------------------------- mesh builders
# One revolved cavity (drill -> cone -> bore), not a boolean union of a
# cylinder, a cone and a second cylinder: a union leaves seams that fragment
# the conical band and is not what a CAD STL export looks like.

def _counterdrill_tool(drill_r, bore_r, half_angle_deg, cb_depth, mouth_z,
                       floor_z, sections=SECTIONS):
    from shapely.geometry import Polygon
    ha = np.deg2rad(half_angle_deg)
    cone_top = mouth_z - cb_depth                  # bore floor = cone wide rim
    throat = cone_top - (bore_r - drill_r) / np.tan(ha)
    prof = [(0.0, floor_z), (drill_r, floor_z), (drill_r, throat),
            (bore_r, cone_top), (bore_r, mouth_z), (0.0, mouth_z)]
    return trimesh.creation.revolve(Polygon(prof).exterior.coords,
                                    sections=sections)


def counterdrilled_through_plate(drill_r=2.5, bore_r=5.0, half_angle_deg=45.0,
                                 cb_depth=3.0, t=14.0, ext=(40.0, 30.0),
                                 center=(0.0, 0.0)):
    """Plate (z in [-t/2, t/2]) with a counterdrilled THROUGH hole."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    tool = _counterdrill_tool(drill_r, bore_r, half_angle_deg, cb_depth,
                              t / 2.0, -t / 2.0 - 2.0)
    tool.apply_translation([center[0], center[1], 0.0])
    return plate.difference(tool)


def counterdrilled_blind_plate(drill_r=2.5, bore_r=5.0, half_angle_deg=45.0,
                               cb_depth=3.0, t=16.0, depth=11.0,
                               ext=(40.0, 30.0), center=(0.0, 0.0)):
    """Counterdrilled BLIND hole: flat drill floor ``depth`` below the top."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    tool = _counterdrill_tool(drill_r, bore_r, half_angle_deg, cb_depth,
                              t / 2.0, t / 2.0 - depth)
    tool.apply_translation([center[0], center[1], 0.0])
    return plate.difference(tool)


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


def _only_counterdrill(feats):
    cd = [h for h in feats.by_kind("hole")
          if h.params.get("countersink")
          and h.params.get("counterbore_diameter")]
    assert len(cd) == 1, f"expected 1 counterdrill, got {len(cd)}"
    return cd[0]


# ------------------------------------------------------------------ detection

class TestCounterdrillDetection:
    def test_through_counterdrill_recognized(self):
        _, feats, _ = _full(counterdrilled_through_plate())
        cd = _only_counterdrill(feats)
        assert cd.params["through"] is True
        assert np.isclose(cd.params["diameter"], 5.0, atol=0.05)
        assert np.isclose(cd.params["counterbore_diameter"], 10.0, atol=0.15)
        assert np.isclose(cd.params["counterbore_depth"], 3.0, atol=0.1)
        assert np.isclose(cd.params["countersink_angle"], 90.0, atol=1.5)

    def test_no_spurious_second_hole(self):
        # the bore must not be re-emitted as a blind hole of the bore diameter
        _, feats, _ = _full(counterdrilled_through_plate())
        assert len(feats.by_kind("hole")) == 1
        assert feats.by_kind("counterbore") == []

    def test_depth_measured_from_the_true_mouth(self):
        # depth spans the TOP FACE to the far face, not the bore floor down
        _, feats, _ = _full(counterdrilled_through_plate(t=14.0))
        cd = _only_counterdrill(feats)
        assert np.isclose(cd.params["depth"], 14.0, atol=0.15)

    def test_mouth_is_the_top_face_not_the_bore_floor(self):
        _, feats, _ = _full(counterdrilled_through_plate(t=14.0))
        cd = _only_counterdrill(feats)
        assert np.isclose(np.asarray(cd.params["mouth"])[2], 7.0, atol=0.15)

    def test_blind_counterdrill_recognized(self):
        _, feats, _ = _full(counterdrilled_blind_plate(depth=11.0))
        cd = _only_counterdrill(feats)
        assert cd.params["through"] is False
        assert np.isclose(cd.params["depth"], 11.0, atol=0.15)
        assert np.isclose(cd.params["counterbore_depth"], 3.0, atol=0.1)

    def test_cone_and_both_cylinders_consumed(self):
        report, feats, _ = _full(counterdrilled_through_plate())
        for i in feats.unassigned:
            prim = report.surfaces[i].fit.primitive
            assert not isinstance(prim, Cone), "cone left unassigned"
            assert not isinstance(prim, Cylinder), "cylinder left unassigned"

    @pytest.mark.parametrize("drill_r,bore_r,cb_depth", [
        (1.5, 3.0, 2.0), (2.5, 5.0, 4.0), (4.0, 7.0, 2.5)])
    def test_dimension_variants(self, drill_r, bore_r, cb_depth):
        _, feats, _ = _full(counterdrilled_through_plate(
            drill_r=drill_r, bore_r=bore_r, cb_depth=cb_depth))
        cd = _only_counterdrill(feats)
        assert np.isclose(cd.params["diameter"], 2 * drill_r, atol=0.05)
        assert np.isclose(cd.params["counterbore_diameter"], 2 * bore_r,
                          atol=0.2)
        assert np.isclose(cd.params["counterbore_depth"], cb_depth, atol=0.15)

    @pytest.mark.parametrize("half_angle_deg,included", [
        (41.0, 82.0), (45.0, 90.0), (60.0, 120.0)])
    def test_included_angle_recovered(self, half_angle_deg, included):
        _, feats, _ = _full(counterdrilled_through_plate(
            half_angle_deg=half_angle_deg))
        cd = _only_counterdrill(feats)
        assert np.isclose(cd.params["countersink_angle"], included, atol=2.0)

    def test_offcenter_position(self):
        _, feats, _ = _full(counterdrilled_through_plate(center=(8.0, -6.0)))
        cd = _only_counterdrill(feats)
        pos = np.asarray(cd.params["position"])
        assert np.isclose(pos[0], 8.0, atol=0.1)
        assert np.isclose(pos[1], -6.0, atol=0.1)

    def test_rotated_part_still_recognized(self):
        mesh = counterdrilled_through_plate().copy()
        mesh.apply_transform(ROT)
        _, feats, _ = _full(mesh)
        cd = _only_counterdrill(feats)
        assert cd.params["through"] is True
        assert np.isclose(cd.params["counterbore_diameter"], 10.0, atol=0.25)

    def test_description_names_both_cuts(self):
        _, feats, _ = _full(counterdrilled_through_plate())
        cd = _only_counterdrill(feats)
        assert "counterdrill" in cd.description.lower()


# ------------------------------------------------------- negative / robustness

class TestCounterdrillNegatives:
    def test_plain_countersink_is_not_a_counterdrill(self):
        from .test_countersink import countersunk_through_plate
        _, feats, _ = _full(countersunk_through_plate())
        holes = feats.by_kind("hole")
        assert len(holes) == 1
        assert holes[0].params.get("countersink")
        assert holes[0].params.get("counterbore_diameter") is None  # noqa

    def test_plain_counterbore_is_not_a_counterdrill(self):
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        bore = trimesh.creation.cylinder(radius=4.5, height=4.0,
                                         sections=SECTIONS)
        bore.apply_translation([0.0, 0.0, 3.0])
        _, feats, _ = _full(plate.difference(drill).difference(bore))
        cbs = feats.by_kind("counterbore")
        assert len(cbs) == 1
        assert not cbs[0].params.get("countersink")

    def test_bore_wider_than_the_cone_rim_is_not_a_counterdrill(self):
        # a counterbore with a REAL annular shoulder, and a countersink at
        # its floor: the cone rim is narrower than the bore, so this must
        # not be collapsed into a counterdrill
        plate = trimesh.creation.box(extents=[40.0, 30.0, 14.0])
        from .test_countersink import _revolved_tool
        tool = _revolved_tool(2.5, 4.0, 45.0, 4.0, -9.0)   # csink rim d8
        bore = trimesh.creation.cylinder(radius=6.0, height=6.0,
                                         sections=SECTIONS)
        bore.apply_translation([0.0, 0.0, 7.0])            # z in [4, 10]
        _, feats, _ = _full(plate.difference(tool).difference(bore))
        assert not any(h.params.get("countersink")
                       and h.params.get("counterbore_diameter")
                       for h in feats.by_kind("hole"))


# ------------------------------------------------- pure property mapping (exec)

class TestCounterdrillPropertyMapping:
    def test_both_cuts_map_to_counterdrill(self):
        op = HoleOp(diameter=5.0, through=True, depth=14.0, positions=[(0, 0)],
                    counterbore_diameter=10.0, counterbore_depth=3.0,
                    countersink_diameter=10.0, countersink_angle=90.0)
        p = hole_op_properties(op)
        assert p["HoleCutType"] == "Counterdrill"
        assert p["HoleCutDiameter"] == 10.0
        assert p["HoleCutDepth"] == 3.0            # CYLINDRICAL part only
        assert np.isclose(p["HoleCutCountersinkAngle"], 90.0)

    def test_counterdrill_angle_defaults_to_90(self):
        op = HoleOp(diameter=5.0, through=True, depth=14.0, positions=[(0, 0)],
                    counterbore_diameter=10.0, counterbore_depth=3.0,
                    countersink_diameter=10.0)
        p = hole_op_properties(op)
        assert p["HoleCutType"] == "Counterdrill"
        assert np.isclose(p["HoleCutCountersinkAngle"], 90.0)

    def test_countersink_only_still_countersink(self):
        op = HoleOp(diameter=5.0, through=True, depth=10.0, positions=[(0, 0)],
                    countersink_diameter=10.0, countersink_angle=90.0)
        p = hole_op_properties(op)
        assert p["HoleCutType"] == "Countersink"
        assert "HoleCutDepth" not in p

    def test_counterbore_only_still_counterbore(self):
        op = HoleOp(diameter=5.0, through=True, depth=10.0, positions=[(0, 0)],
                    counterbore_diameter=9.0, counterbore_depth=4.0)
        p = hole_op_properties(op)
        assert p["HoleCutType"] == "Counterbore"
        assert "HoleCutCountersinkAngle" not in p

    def test_blind_counterdrill_depth(self):
        op = HoleOp(diameter=5.0, through=False, depth=11.0, positions=[(0, 0)],
                    counterbore_diameter=10.0, counterbore_depth=3.0,
                    countersink_diameter=10.0, countersink_angle=90.0)
        p = hole_op_properties(op)
        assert p["DepthType"] == "Dimension"
        assert p["Depth"] == 11.0
        assert p["HoleCutType"] == "Counterdrill"


# -------------------------------------------------------------------- planning

class TestCounterdrillPlanning:
    def test_through_planned_as_one_hole_op(self):
        _, _, plan = _full(counterdrilled_through_plate())
        assert len(plan.holes) == 1
        assert plan.unplanned == []
        op = plan.holes[0]
        assert op.through is True
        assert np.isclose(op.counterbore_diameter, 10.0, atol=0.15)
        assert np.isclose(op.counterbore_depth, 3.0, atol=0.1)
        assert op.countersink_diameter is not None

    def test_sketch_plane_is_the_top_face(self):
        # surface_z must be the plate top (plan frame), not the bore floor
        _, _, plan = _full(counterdrilled_through_plate(t=14.0))
        op = plan.holes[0]
        assert op.from_top is True
        assert np.isclose(op.surface_z, 14.0, atol=0.15)

    def test_blind_planned_as_one_hole_op(self):
        _, _, plan = _full(counterdrilled_blind_plate(depth=11.0))
        assert len(plan.holes) == 1
        assert plan.unplanned == []
        op = plan.holes[0]
        assert op.through is False
        assert np.isclose(op.depth, 11.0, atol=0.15)

    def test_counterdrill_alongside_plain_hole(self):
        mesh = counterdrilled_through_plate(center=(-8.0, 0.0),
                                            ext=(50.0, 30.0))
        plain = trimesh.creation.cylinder(radius=2.0, height=40.0,
                                          sections=SECTIONS)
        plain.apply_translation([12.0, 6.0, 0.0])
        _, _, plan = _full(mesh.difference(plain))
        assert len(plan.holes) == 2
        assert sum(op.counterbore_diameter is not None
                   and op.countersink_diameter is not None
                   for op in plan.holes) == 1
        assert plan.unplanned == []


# ------------------------------------------------------------------ round-trip

class TestCounterdrillRoundtrip:
    def test_through_roundtrip(self):
        mesh = counterdrilled_through_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_blind_roundtrip(self):
        mesh = counterdrilled_blind_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_rotated_roundtrip(self):
        mesh = counterdrilled_through_plate().copy()
        mesh.apply_transform(ROT)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    @pytest.mark.parametrize("half_angle_deg", [41.0, 60.0])
    def test_angle_variants_roundtrip(self, half_angle_deg):
        mesh = counterdrilled_through_plate(half_angle_deg=half_angle_deg)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)


# ---------------------------------------------------------------- edge cases

def _grid_plate():
    plate = trimesh.creation.box(extents=[44.0, 44.0, 14.0])
    for cx in (-11.0, 11.0):
        for cy in (-11.0, 11.0):
            tool = _counterdrill_tool(2.5, 5.0, 45.0, 3.0, 7.0, -9.0)
            tool.apply_translation([cx, cy, 0.0])
            plate = plate.difference(tool)
    return plate


def _bottom_counterdrill_plate():
    mesh = counterdrilled_through_plate().copy()
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return mesh


class TestCounterdrillEdgeCases:
    def test_grid_of_counterdrills(self):
        _, feats, plan = _full(_grid_plate())
        cd = [h for h in feats.by_kind("hole")
              if h.params.get("countersink")
              and h.params.get("counterbore_diameter")]
        assert len(cd) == 4
        ops = [op for op in plan.holes
               if op.counterbore_diameter and op.countersink_diameter]
        assert len(ops) == 1                      # collapsed into one pattern
        assert len(ops[0].positions) == 4
        assert plan.unplanned == []

    def test_grid_roundtrip(self):
        mesh = _grid_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_bottom_face_counterdrill(self):
        _, _, plan = _full(_bottom_counterdrill_plate())
        assert len(plan.holes) == 1
        op = plan.holes[0]
        assert op.from_top is False
        assert np.isclose(op.counterbore_diameter, 10.0, atol=0.25)
        assert np.isclose(op.counterbore_depth, 3.0, atol=0.15)

    def test_bottom_face_roundtrip(self):
        mesh = _bottom_counterdrill_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_shallow_bore_counterdrill(self):
        # a bore only just deeper than the mesh tolerance must still pair
        _, feats, _ = _full(counterdrilled_through_plate(cb_depth=1.5))
        cd = _only_counterdrill(feats)
        assert np.isclose(cd.params["counterbore_depth"], 1.5, atol=0.15)
