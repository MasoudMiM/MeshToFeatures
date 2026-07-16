# SPDX-License-Identifier: LGPL-2.1-or-later
"""v0.14 tests (completeness before release), written before implementation.

* CHAMFERS -- an edge chamfer is a narrow plane strip whose normal
  bisects two (roughly perpendicular) neighbour planes. Like fillets,
  vertical chamfers are absorbed into the base profile as plain line
  segments; horizontal ones become ChamferOps carrying the sharp edge
  (the intersection line of the two blended planes) and the equal-leg
  size, applied as PartDesign::Chamfer on the rebuilt body.

* CROSS-AXIS HOLES -- through holes whose axis is perpendicular to the
  extrusion direction become CrossHoleOps (3D axis + anchors) executed
  as midplane through-all pockets. This FLIPS the v0.8 contract that
  side holes land in `unplanned` (updated in test_adversarial).
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.history import plan_history
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

from .test_adversarial import ROT, assert_geometry_match, side_drilled_plate

pytest.importorskip("manifold3d")

SECTIONS = 64
Y = np.array([0.0, 1.0, 0.0])


def _wedge(corner_x, sign):
    """Chamfer cutting wedge for a top edge along y at x = corner_x."""
    s, m = 2.0, 1.0
    pts2d = [(m, m), (0.0, -s), (-s, 0.0)]     # (a, b) in (n_a, n_b) frame
    na = np.array([sign, 0.0, 0.0])
    nb = np.array([0.0, 0.0, 1.0])
    corner = np.array([corner_x, 0.0, 5.0])
    pts = []
    for y in (-25.0, 25.0):
        for a, b in pts2d:
            pts.append(corner + a * na + b * nb + np.array([0.0, y, 0.0]))
    return trimesh.Trimesh(vertices=np.array(pts)).convex_hull


def chamfered_top_plate():
    """40 x 30 x 10 plate, both long TOP edges chamfered 2 x 2 (45 deg)."""
    plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])   # z in [-5, 5]
    for sx in (1.0, -1.0):
        plate = plate.difference(_wedge(sx * 20.0, sx))
    return plate


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    return report, feats, plan_history(report, feats, detect_patterns(feats),
                                       patches)


class TestChamferFeatures:
    def test_two_chamfers_recognized(self):
        _, feats, _ = _full(chamfered_top_plate())
        cs = feats.by_kind("chamfer")
        assert len(cs) == 2
        for c in cs:
            assert np.isclose(c.params["size"], 2.0, atol=0.02)
            assert np.isclose(c.params["angle_deg"], 45.0, atol=2.0)
            assert abs(np.asarray(c.params["axis"]) @ Y) > 0.999

    def test_ordinary_walls_are_not_chamfers(self):
        from .test_composites import pocketed_plate
        _, feats, _ = _full(pocketed_plate())
        assert feats.by_kind("chamfer") == []


class TestChamferPlan:
    def test_plan_carries_two_chamfer_ops(self):
        _, _, plan = _full(chamfered_top_plate())
        assert len(plan.chamfers) == 2
        xs = sorted(round(abs(float(op.edge_start[0])), 6)
                    for op in plan.chamfers)
        assert xs == [20.0, 20.0]              # sharp edges at x = +-20
        for op in plan.chamfers:
            assert np.isclose(op.size, 2.0, atol=0.02)
            assert np.isclose(op.edge_start[2], 5.0, atol=1e-6)
            span = np.linalg.norm(op.edge_end - op.edge_start)
            assert np.isclose(span, 30.0, atol=0.05)

    def test_geometry_roundtrip(self):
        mesh = chamfered_top_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)


class TestCrossAxisHoles:
    def test_side_hole_becomes_a_cross_hole_op(self):
        _, feats, plan = _full(side_drilled_plate())
        assert len(feats.by_kind("hole")) == 1
        assert len(plan.cross_holes) == 1
        ch = plan.cross_holes[0]
        assert np.isclose(ch.diameter, 6.0, atol=0.01)
        assert abs(float(np.asarray(ch.axis)
                         @ np.array([1.0, 0.0, 0.0]))) > 0.999
        assert plan.unplanned == []            # nothing silently dropped

    def test_geometry_roundtrip(self):
        mesh = side_drilled_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_combined_top_and_side_features(self):
        # plate with a top counterbore AND a side through-hole: both axes
        # rebuilt in one plan
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        drill = trimesh.creation.cylinder(radius=4.0, height=30.0,
                                          sections=SECTIONS)
        bore = trimesh.creation.cylinder(radius=6.0, height=4.0,
                                         sections=SECTIONS)
        bore.apply_translation([0.0, 0.0, 5.0])
        side = trimesh.creation.cylinder(radius=2.0, height=60.0,
                                         sections=SECTIONS)
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        side.apply_transform(rot)
        side.apply_translation([0.0, 8.0, -2.0])
        mesh = (plate.difference(drill).difference(bore)
                     .difference(side))
        _, feats, plan = _full(mesh)
        assert len(plan.holes) == 1            # the counterbore, top axis
        assert len(plan.cross_holes) == 1      # the side hole
        assert_geometry_match(mesh, plan)

    def test_blind_side_hole_becomes_a_cross_hole_op(self):
        # a blind side hole is now rebuilt as a depth-limited CrossHoleOp
        # (v0.14 left it unplanned; this FLIPS that contract)
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        side = trimesh.creation.cylinder(radius=3.0, height=30.0,
                                         sections=SECTIONS)
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        side.apply_transform(rot)
        side.apply_translation([-15.0, 0.0, 0.0])   # enters -x face, blind
        mesh = plate.difference(side)
        _, feats, plan = _full(mesh)
        holes = feats.by_kind("hole")
        assert len(holes) == 1
        assert holes[0].params["through"] is False
        assert len(plan.cross_holes) == 1
        ch = plan.cross_holes[0]
        assert ch.through is False
        assert np.isclose(ch.diameter, 6.0, atol=0.02)
        # enters at x=-20; drill (h30 centred at x=-15) floors at x=0 -> 20
        assert np.isclose(ch.depth, 20.0, atol=0.1)
        # entry direction points INTO the part (+x from the -x wall)
        assert float(np.asarray(ch.entry_direction) @ [1.0, 0, 0]) > 0.99
        assert plan.unplanned == []

    def test_blind_side_hole_roundtrip(self):
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        side = trimesh.creation.cylinder(radius=3.0, height=30.0,
                                         sections=SECTIONS)
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        side.apply_transform(rot)
        side.apply_translation([-15.0, 0.0, 0.0])
        mesh = plate.difference(side)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_through_and_blind_side_holes_together(self):
        plate = trimesh.creation.box(extents=[60.0, 30.0, 10.0])
        through = trimesh.creation.cylinder(radius=2.0, height=60.0,
                                            sections=SECTIONS)
        through.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        through.apply_translation([0.0, 8.0, 0.0])           # y=8, through x
        blind = trimesh.creation.cylinder(radius=3.0, height=30.0,
                                          sections=SECTIONS)
        blind.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        blind.apply_translation([-20.0, -8.0, 0.0])          # enters -x, blind
        mesh = plate.difference(through).difference(blind)
        _, _, plan = _full(mesh)
        assert len(plan.cross_holes) == 2
        assert sum(c.through for c in plan.cross_holes) == 1
        assert sum(not c.through for c in plan.cross_holes) == 1
        assert plan.unplanned == []

    def test_through_and_blind_side_roundtrip(self):
        plate = trimesh.creation.box(extents=[60.0, 30.0, 10.0])
        through = trimesh.creation.cylinder(radius=2.0, height=60.0,
                                            sections=SECTIONS)
        through.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        through.apply_translation([0.0, 8.0, 0.0])
        blind = trimesh.creation.cylinder(radius=3.0, height=30.0,
                                          sections=SECTIONS)
        blind.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        blind.apply_translation([-20.0, -8.0, 0.0])
        mesh = plate.difference(through).difference(blind)
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_blind_side_hole_from_plus_x(self):
        # entering the +x face: direction must point -x (inward)
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        side = trimesh.creation.cylinder(radius=3.0, height=30.0,
                                         sections=SECTIONS)
        side.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        side.apply_translation([15.0, 0.0, 0.0])             # enters +x face
        mesh = plate.difference(side)
        _, _, plan = _full(mesh)
        assert len(plan.cross_holes) == 1
        ch = plan.cross_holes[0]
        assert ch.through is False
        assert float(np.asarray(ch.entry_direction) @ [1.0, 0, 0]) < -0.99

    def test_blind_side_hole_on_y_face(self):
        # a blind hole entering the -y face (axis y), inward = +y
        plate = trimesh.creation.box(extents=[40.0, 40.0, 10.0])
        side = trimesh.creation.cylinder(radius=3.0, height=25.0,
                                         sections=SECTIONS)
        side.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        side.apply_translation([0.0, -20.0 + 12.5, 0.0])
        mesh = plate.difference(side)
        _, _, plan = _full(mesh)
        assert len(plan.cross_holes) == 1
        ch = plan.cross_holes[0]
        assert ch.through is False
        assert float(np.asarray(ch.entry_direction) @ [0, 1.0, 0]) > 0.99

    def test_blind_side_hole_rotated_roundtrip(self):
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        side = trimesh.creation.cylinder(radius=3.0, height=25.0,
                                         sections=SECTIONS)
        side.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        side.apply_translation([-20.0 + 12.5, 0.0, 0.0])
        mesh = plate.difference(side).copy()
        mesh.apply_transform(ROT)
        _, _, plan = _full(mesh)
        assert any(not c.through for c in plan.cross_holes)
        assert_geometry_match(mesh, plan)

    def test_two_blind_side_holes_both_planned(self):
        plate = trimesh.creation.box(extents=[40.0, 40.0, 10.0])
        for cy in (-10.0, 10.0):
            d = trimesh.creation.cylinder(radius=3.0, height=25.0,
                                          sections=SECTIONS)
            d.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            d.apply_translation([-20.0 + 12.5, cy, 0.0])
            plate = plate.difference(d)
        _, _, plan = _full(plate)
        # both blind side holes reconstructed (one op with two positions, or
        # two ops); either way, all blind and nothing dropped
        total = sum(len(c.positions3d) for c in plan.cross_holes
                    if not c.through)
        assert total == 2
        assert plan.unplanned == []
