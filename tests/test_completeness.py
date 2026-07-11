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

from .test_adversarial import assert_geometry_match, side_drilled_plate

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

    def test_blind_side_holes_stay_unplanned(self):
        # honesty: only THROUGH cross-axis holes are rebuilt in v0.14
        plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
        side = trimesh.creation.cylinder(radius=3.0, height=30.0,
                                         sections=SECTIONS)
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        side.apply_transform(rot)
        side.apply_translation([-15.0, 0.0, 0.0])   # enters -x face, blind
        mesh = plate.difference(side)
        _, feats, plan = _full(mesh)
        if feats.by_kind("hole"):              # recognized as a blind hole
            assert plan.cross_holes == []
            assert plan.unplanned              # reported, not silent
