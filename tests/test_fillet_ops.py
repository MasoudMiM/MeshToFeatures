# SPDX-License-Identifier: LGPL-2.1-or-later
"""v0.12 tests (horizontal fillets), written before the implementation.

A fillet whose axis is perpendicular to the extrusion direction cannot be
absorbed into the base sketch; it must be applied to EDGES of the rebuilt
sharp solid. The plan therefore carries FilletOps describing the sharp
edge each fillet replaces:

    sharp edge line = fillet axis line +- r * (n_A + n_B)

(n_A, n_B outward unit normals of the two blended planes; '+' for convex).
The executor finds matching straight edges on the built body with a PURE
matcher tested here, and the round-trip helper applies convex fillets
with booleans so the whole plan is geometry-verified headlessly.
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.history import FilletOp, fillet_edge_matches, plan_history
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

from .test_adversarial import assert_geometry_match
from .test_robustness import rounded_slab

pytest.importorskip("manifold3d")

SECTIONS = 64
Z = np.array([0.0, 0.0, 1.0])
Y = np.array([0.0, 1.0, 0.0])


def rounded_top_plate():
    """40 x 30 x 10 plate, both long TOP edges filleted r=3 (axis = y):
    the canonical horizontal-fillet part."""
    plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])   # z in [-5, 5]
    out = plate
    for sx in (1.0, -1.0):
        corner = trimesh.creation.box(extents=[4.0, 40.0, 4.0])
        corner.apply_translation([sx * 19.0, 0.0, 4.0])        # x 17..21, z 2..6
        cyl = trimesh.creation.cylinder(radius=3.0, height=40.0,
                                        sections=SECTIONS)
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cyl.apply_transform(rot)                               # axis = y
        cyl.apply_translation([sx * 17.0, 0.0, 2.0])
        out = out.difference(corner.difference(cyl))
    return out


def _plan(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    return report, feats, plan_history(report, feats, detect_patterns(feats),
                                       patches)


class TestFilletFeatures:
    def test_two_horizontal_fillets_recognized(self):
        _, feats, _ = _plan(rounded_top_plate())
        fs = feats.by_kind("fillet")
        assert len(fs) == 2
        for f in fs:
            assert f.params["radius"] == 3.0
            assert f.params["convex"] is True
            assert abs(np.asarray(f.params["axis"]) @ Y) > 0.999


class TestFilletPlan:
    def test_plan_carries_two_fillet_ops(self):
        _, _, plan = _plan(rounded_top_plate())
        assert len(plan.fillets) == 2
        for op in plan.fillets:
            assert op.radius == 3.0
            assert abs(op.direction @ Y) > 0.999

    def test_sharp_edge_reconstructed_at_original_corner(self):
        _, _, plan = _plan(rounded_top_plate())
        # the sharp edges are the lines x = +-20, z = +5 (top corners)
        xs = sorted(round(float(op.edge_start[0]), 6) for op in plan.fillets)
        assert xs == [-20.0, 20.0]
        for op in plan.fillets:
            assert np.isclose(op.edge_start[2], 5.0, atol=1e-6)
            assert np.isclose(op.edge_end[2], 5.0, atol=1e-6)
            span = np.linalg.norm(op.edge_end - op.edge_start)
            assert np.isclose(span, 30.0, atol=0.02)

    def test_base_is_full_sized_sharp_solid(self):
        _, _, plan = _plan(rounded_top_plate())
        assert np.isclose(plan.base.length, 10.0)
        # bottom face is unfilleted: profile must be the full 40 x 30 rect
        # (the frame's in-plane axes are arbitrary: compare sorted extents)
        pts = np.vstack([p.sample() for p in plan.base.profile])
        ext = sorted(np.round(pts.max(axis=0) - pts.min(axis=0), 6))
        assert ext == [30.0, 40.0]

    def test_vertical_fillets_still_absorbed_not_edge_ops(self):
        _, _, plan = _plan(rounded_slab())
        assert plan.fillets == []
        assert plan.absorbed_features >= 4


class TestEdgeMatcher:
    def _op(self):
        return FilletOp(radius=3.0,
                        edge_start=np.array([20.0, -15.0, 5.0]),
                        edge_end=np.array([20.0, 15.0, 5.0]),
                        direction=Y.copy(),
                        n_a=np.array([1.0, 0.0, 0.0]),
                        n_b=np.array([0.0, 0.0, 1.0]))

    def test_matching_edge(self):
        op = self._op()
        assert fillet_edge_matches(op, np.array([20.0, -15.0, 5.0]),
                                   np.array([20.0, 15.0, 5.0]), tol=0.05)

    def test_direction_mismatch_rejected(self):
        op = self._op()
        assert not fillet_edge_matches(op, np.array([20.0, -15.0, 5.0]),
                                       np.array([20.0, -15.0, -5.0]), tol=0.05)

    def test_parallel_but_offset_edge_rejected(self):
        op = self._op()  # the bottom edge x=20, z=-5: parallel, wrong line
        assert not fillet_edge_matches(op, np.array([20.0, -15.0, -5.0]),
                                       np.array([20.0, 15.0, -5.0]), tol=0.05)

    def test_short_sub_edge_on_the_line_accepted(self):
        # boolean rebuilds may split an edge; partial coverage still matches
        op = self._op()
        assert fillet_edge_matches(op, np.array([20.0, -5.0, 5.0]),
                                   np.array([20.0, 10.0, 5.0]), tol=0.05)


class TestRoundTrip:
    def test_filleted_plate_geometry_matches(self):
        mesh = rounded_top_plate()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)
