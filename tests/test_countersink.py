# SPDX-License-Identifier: LGPL-2.1-or-later
"""Countersink tests, written before the implementation.

A countersunk hole is a coaxial CONE (the conical entry) sitting on the
opening side of a drilled CYLINDER. The cone's wide rim opens at a face;
its narrow end meets the drill. Recognized as a hole feature carrying
countersink parameters (diameter, included angle) and rebuilt as a
PartDesign::Hole with HoleCutType = Countersink.

Geometry facts (verified): for an included angle 2*alpha, the cone's
half-angle is alpha; the cone radius at axial height h above the apex is
h*tan(alpha); the narrow end matches the drill radius; the wide rim gives
the countersink diameter.
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
from meshtofeatures.snapping import snap_report

from .test_adversarial import assert_geometry_match, ROT

SECTIONS = 64


# --------------------------------------------------------------- mesh builders
# A drilled + countersunk hole is a SINGLE revolved cavity (drill up to the
# throat, cone up to the mouth). Building it as one revolution -- rather than
# union-ing a separate cylinder and cone -- gives the regular tessellation a
# real CAD STL export has, instead of a messy boolean seam.

def _revolved_tool(drill_r, csink_r, half_angle_deg, mouth_z, floor_z,
                   sections=SECTIONS):
    from shapely.geometry import Polygon
    ha = np.deg2rad(half_angle_deg)
    throat_z = mouth_z - (csink_r - drill_r) / np.tan(ha)
    prof = [(0.0, floor_z), (drill_r, floor_z), (drill_r, throat_z),
            (csink_r, mouth_z), (0.0, mouth_z)]
    return trimesh.creation.revolve(Polygon(prof).exterior.coords,
                                    sections=sections)


def countersunk_through_plate(drill_r=2.5, csink_r=5.0, half_angle_deg=45.0,
                              t=10.0, ext=(40.0, 30.0), center=(0.0, 0.0)):
    """Plate (z in [-t/2, t/2]) with a countersunk THROUGH hole."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    tool = _revolved_tool(drill_r, csink_r, half_angle_deg,
                          t / 2.0, -t / 2.0 - 2.0)
    tool.apply_translation([center[0], center[1], 0.0])
    return plate.difference(tool)


def countersunk_blind_plate(drill_r=2.5, csink_r=5.0, half_angle_deg=45.0,
                            t=12.0, depth=8.0, ext=(40.0, 30.0),
                            center=(0.0, 0.0)):
    """Plate (z in [-t/2, t/2]) with a countersunk BLIND hole: the drill
    reaches ``depth`` below the top face, capped by a flat floor."""
    plate = trimesh.creation.box(extents=[ext[0], ext[1], t])
    tool = _revolved_tool(drill_r, csink_r, half_angle_deg,
                          t / 2.0, t / 2.0 - depth)
    tool.apply_translation([center[0], center[1], 0.0])
    return plate.difference(tool)


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


def _only_countersink(feats):
    cs = [h for h in feats.by_kind("hole") if h.params.get("countersink")]
    assert len(cs) == 1, f"expected 1 countersink, got {len(cs)}"
    return cs[0]


# ------------------------------------------------------------------ detection

class TestCountersinkDetection:
    def test_through_countersink_recognized(self):
        _, feats, _ = _full(countersunk_through_plate())
        cs = _only_countersink(feats)
        assert cs.params["through"] is True
        assert np.isclose(cs.params["diameter"], 5.0, atol=0.05)
        assert np.isclose(cs.params["countersink_diameter"], 10.0, atol=0.1)
        assert np.isclose(cs.params["countersink_angle"], 90.0, atol=1.5)

    def test_blind_countersink_recognized(self):
        _, feats, _ = _full(countersunk_blind_plate(depth=8.0))
        cs = _only_countersink(feats)
        assert cs.params["through"] is False
        assert np.isclose(cs.params["diameter"], 5.0, atol=0.05)
        assert np.isclose(cs.params["countersink_diameter"], 10.0, atol=0.1)
        # depth measured from the MOUTH (top face) to the flat floor
        assert np.isclose(cs.params["depth"], 8.0, atol=0.1)

    @pytest.mark.parametrize("half_angle_deg,included", [
        (41.0, 82.0), (45.0, 90.0), (50.0, 100.0), (60.0, 120.0)])
    def test_included_angle_recovered(self, half_angle_deg, included):
        _, feats, _ = _full(
            countersunk_through_plate(half_angle_deg=half_angle_deg))
        cs = _only_countersink(feats)
        assert np.isclose(cs.params["countersink_angle"], included, atol=2.0)

    @pytest.mark.parametrize("drill_r,csink_r", [
        (1.5, 3.0), (2.5, 5.0), (3.0, 7.0), (4.0, 8.0)])
    def test_diameter_ratios(self, drill_r, csink_r):
        _, feats, _ = _full(countersunk_through_plate(
            drill_r=drill_r, csink_r=csink_r))
        cs = _only_countersink(feats)
        assert np.isclose(cs.params["diameter"], 2 * drill_r, atol=0.05)
        assert np.isclose(cs.params["countersink_diameter"], 2 * csink_r,
                          atol=0.15)

    def test_cone_and_drill_both_consumed(self):
        report, feats, _ = _full(countersunk_through_plate())
        # nothing conical or the drill cylinder left unassigned
        from meshtofeatures.primitives import Cone, Cylinder
        for i in feats.unassigned:
            prim = report.surfaces[i].fit.primitive
            assert not isinstance(prim, Cone), "cone left unassigned"
            assert not isinstance(prim, Cylinder), "drill left unassigned"

    def test_axis_is_vertical(self):
        _, feats, _ = _full(countersunk_through_plate())
        cs = _only_countersink(feats)
        assert abs(float(np.asarray(cs.params["axis"]) @ [0, 0, 1.0])) > 0.999

    def test_offcenter_position(self):
        _, feats, _ = _full(countersunk_through_plate(center=(8.0, -6.0)))
        cs = _only_countersink(feats)
        pos = np.asarray(cs.params["position"])
        assert np.isclose(pos[0], 8.0, atol=0.1)
        assert np.isclose(pos[1], -6.0, atol=0.1)

    def test_rotated_part_still_recognized(self):
        mesh = countersunk_through_plate().copy()
        mesh.apply_transform(ROT)
        _, feats, _ = _full(mesh)
        cs = _only_countersink(feats)
        assert cs.params["through"] is True
        assert np.isclose(cs.params["countersink_diameter"], 10.0, atol=0.2)

    def test_countersink_alongside_plain_hole(self):
        mesh = countersunk_through_plate(center=(-8.0, 0.0))
        plain = trimesh.creation.cylinder(radius=2.0, height=30.0,
                                          sections=SECTIONS)
        plain.apply_translation([10.0, 6.0, 0.0])
        mesh = mesh.difference(plain)
        _, feats, _ = _full(mesh)
        holes = feats.by_kind("hole")
        cs = [h for h in holes if h.params.get("countersink")]
        plainh = [h for h in holes if not h.params.get("countersink")]
        assert len(cs) == 1
        assert len(plainh) == 1
        assert np.isclose(plainh[0].params["diameter"], 4.0, atol=0.05)


# ------------------------------------------------------- negative / robustness

class TestCountersinkNegatives:
    def test_plain_hole_has_no_countersink(self):
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        drill = trimesh.creation.cylinder(radius=3.0, height=30.0,
                                          sections=SECTIONS)
        _, feats, _ = _full(plate.difference(drill))
        holes = feats.by_kind("hole")
        assert len(holes) == 1
        assert not holes[0].params.get("countersink")

    def test_chamfered_edge_is_not_a_countersink(self):
        # a planar edge chamfer must stay a chamfer, never a countersink
        from .test_completeness import chamfered_top_plate
        _, feats, _ = _full(chamfered_top_plate())
        assert not any(h.params.get("countersink")
                       for h in feats.by_kind("hole"))
        assert len(feats.by_kind("chamfer")) == 2


# ------------------------------------------------- pure property mapping (exec)

class TestCountersinkPropertyMapping:
    def test_countersink_hole_cut_type(self):
        op = HoleOp(diameter=5.0, through=True, depth=10.0, positions=[(0, 0)],
                    countersink_diameter=10.0, countersink_angle=90.0)
        p = hole_op_properties(op)
        assert p["HoleCutType"] == "Countersink"
        assert p["HoleCutDiameter"] == 10.0
        assert np.isclose(p["HoleCutCountersinkAngle"], 90.0)

    def test_countersink_does_not_leak_counterbore(self):
        op = HoleOp(diameter=5.0, through=True, depth=10.0, positions=[(0, 0)],
                    countersink_diameter=10.0, countersink_angle=90.0)
        p = hole_op_properties(op)
        assert "HoleCutDepth" not in p        # angle-defined, not depth

    def test_blind_countersink_depth(self):
        op = HoleOp(diameter=5.0, through=False, depth=8.0, positions=[(0, 0)],
                    countersink_diameter=10.0, countersink_angle=90.0)
        p = hole_op_properties(op)
        assert p["DepthType"] == "Dimension"
        assert p["Depth"] == 8.0
        assert p["HoleCutType"] == "Countersink"


# -------------------------------------------------------------------- planning

class TestCountersinkPlanning:
    def test_through_planned_as_hole_op(self):
        _, _, plan = _full(countersunk_through_plate())
        assert len(plan.holes) == 1
        assert plan.unplanned == []
        op = plan.holes[0]
        assert op.countersink_diameter is not None
        assert np.isclose(op.countersink_diameter, 10.0, atol=0.1)
        assert op.through is True

    def test_blind_planned_as_hole_op(self):
        _, _, plan = _full(countersunk_blind_plate())
        assert len(plan.holes) == 1
        assert plan.unplanned == []
        op = plan.holes[0]
        assert op.through is False
        assert np.isclose(op.depth, 8.0, atol=0.1)

    def test_countersink_and_plain_both_planned(self):
        mesh = countersunk_through_plate(center=(-8.0, 0.0))
        plain = trimesh.creation.cylinder(radius=2.0, height=30.0,
                                          sections=SECTIONS)
        plain.apply_translation([10.0, 6.0, 0.0])
        _, _, plan = _full(mesh.difference(plain))
        assert len(plan.holes) == 2
        assert sum(op.countersink_diameter is not None
                   for op in plan.holes) == 1
        assert plan.unplanned == []


# ------------------------------------------------------------------ round-trip

class TestCountersinkRoundtrip:
    def test_through_roundtrip(self):
        mesh = countersunk_through_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_blind_roundtrip(self):
        mesh = countersunk_blind_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_rotated_roundtrip(self):
        mesh = countersunk_through_plate().copy()
        mesh.apply_transform(ROT)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    @pytest.mark.parametrize("half_angle_deg", [41.0, 60.0])
    def test_angle_variants_roundtrip(self, half_angle_deg):
        mesh = countersunk_through_plate(half_angle_deg=half_angle_deg)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)


# --------------------------------------------------------------- edge cases

def _grid_plate():
    plate = trimesh.creation.box(extents=[40.0, 40.0, 10.0])
    for cx in (-10.0, 10.0):
        for cy in (-10.0, 10.0):
            tool = _revolved_tool(2.5, 5.0, 45.0, 5.0, -7.0)
            tool.apply_translation([cx, cy, 0.0])
            plate = plate.difference(tool)
    return plate


def _bottom_countersink_plate():
    # a through countersink opening on the BOTTOM face
    mesh = countersunk_through_plate().copy()
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return mesh


class TestCountersinkEdgeCases:
    def test_grid_of_countersinks(self):
        _, feats, plan = _full(_grid_plate())
        cs = [h for h in feats.by_kind("hole") if h.params.get("countersink")]
        assert len(cs) == 4
        # the four identical countersinks collapse into one patterned op
        csops = [op for op in plan.holes if op.countersink_diameter]
        assert len(csops) == 1
        assert len(csops[0].positions) == 4
        assert plan.unplanned == []

    def test_grid_roundtrip(self):
        mesh = _grid_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_bottom_face_countersink(self):
        _, _, plan = _full(_bottom_countersink_plate())
        assert len(plan.holes) == 1
        assert plan.holes[0].from_top is False
        assert np.isclose(plan.holes[0].countersink_diameter, 10.0, atol=0.2)

    def test_bottom_face_roundtrip(self):
        mesh = _bottom_countersink_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_tiny_countersink(self):
        plate = trimesh.creation.box(extents=[10.0, 8.0, 3.0])
        tool = _revolved_tool(0.6, 1.2, 45.0, 1.5, -2.5)
        _, feats, _ = _full(plate.difference(tool))
        cs = [h for h in feats.by_kind("hole") if h.params.get("countersink")]
        assert len(cs) == 1
        assert np.isclose(cs[0].params["diameter"], 1.2, atol=0.03)
        assert np.isclose(cs[0].params["countersink_diameter"], 2.4, atol=0.1)

    def test_nonstandard_diameter_not_hallucinated(self):
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        tool = _revolved_tool(1.85, 4.0, 45.0, 5.0, -7.0)   # d3.7: no standard
        _, feats, _ = _full(plate.difference(tool))
        cs = _only_countersink(feats)
        assert cs.params.get("standard") is None

    def test_countersink_and_counterbore_coexist(self):
        plate = trimesh.creation.box(extents=[60.0, 30.0, 10.0])
        tool = _revolved_tool(2.5, 5.0, 45.0, 5.0, -7.0)
        tool.apply_translation([-15.0, 0.0, 0.0])
        plate = plate.difference(tool)
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([15.0, 0.0, 0.0])
        bore = trimesh.creation.cylinder(radius=4.5, height=4.0,
                                         sections=SECTIONS)
        bore.apply_translation([15.0, 0.0, 3.0])
        mesh = plate.difference(drill).difference(bore)
        _, _, plan = _full(mesh)
        assert sum(op.countersink_diameter is not None
                   for op in plan.holes) == 1
        assert sum(op.counterbore_diameter is not None
                   for op in plan.holes) == 1
        assert plan.unplanned == []

    def test_countersink_and_counterbore_roundtrip(self):
        plate = trimesh.creation.box(extents=[60.0, 30.0, 10.0])
        tool = _revolved_tool(2.5, 5.0, 45.0, 5.0, -7.0)
        tool.apply_translation([-15.0, 0.0, 0.0])
        plate = plate.difference(tool)
        drill = trimesh.creation.cylinder(radius=2.5, height=40.0,
                                          sections=SECTIONS)
        drill.apply_translation([15.0, 0.0, 0.0])
        bore = trimesh.creation.cylinder(radius=4.5, height=4.0,
                                         sections=SECTIONS)
        bore.apply_translation([15.0, 0.0, 3.0])
        mesh = plate.difference(drill).difference(bore)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)
