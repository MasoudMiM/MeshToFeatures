# SPDX-License-Identifier: LGPL-2.1-or-later
"""History-planner tests (v0.8), written before the implementation.

The plan must be *executable*: the strongest test rebuilds the solid from
the plan with boolean operations and demands the volume match the input
mesh within 1%. Structure tests pin the semantic content (base length,
op counts, parameters); sketch-conversion tests pin the loop -> lines/arcs
decomposition that makes emitted sketches editable.
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.history import (SketchArc, SketchCircle, SketchLine,
                               loop_to_sketch, plan_history)
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

from .test_composites import (counterbored_plate, drilled_plate,
                              pocketed_plate, stepped_shaft)
from .test_patterns import bolt_circle_plate
from .test_robustness import rounded_slab

pytest.importorskip("manifold3d")

Z = np.array([0.0, 0.0, 1.0])


def _plan(mesh):
    report = snap_report(reconstruct(mesh)).report
    feats = detect_features(report, plan_patches(report))
    return plan_history(report, feats, detect_patterns(feats))


# ------------------------------------------------------- loop_to_sketch

class TestLoopToSketch:
    def test_rectangle_is_four_lines(self):
        loop = np.array([[0, 0], [10, 0], [10, 6], [0, 6.0]])
        prims = loop_to_sketch(loop)
        assert all(isinstance(p, SketchLine) for p in prims)
        assert len(prims) == 4

    def test_rectangle_start_vertex_irrelevant(self):
        loop = np.array([[0, 0], [10, 0], [10, 6], [0, 6.0]])
        # start mid-edge: insert a collinear vertex and rotate
        loop2 = np.vstack([[5.0, 0.0], loop[1:], loop[0]])
        assert len(loop_to_sketch(loop2)) == 4

    def test_polygonal_circle_is_one_circle(self):
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        loop = np.column_stack([3 * np.cos(t), 3 * np.sin(t)])
        prims = loop_to_sketch(loop)
        assert len(prims) == 1
        assert isinstance(prims[0], SketchCircle)
        assert np.isclose(prims[0].radius, 3.0, atol=0.01)

    def test_rounded_rectangle_lines_and_arcs(self):
        # 20 x 10 with r=2 corners, arcs as 12-gon segments
        prims = loop_to_sketch(_rounded_rect_loop(20, 10, 2.0, 12))
        lines = [p for p in prims if isinstance(p, SketchLine)]
        arcs = [p for p in prims if isinstance(p, SketchArc)]
        assert len(lines) == 4
        assert len(arcs) == 4
        for a in arcs:
            assert np.isclose(a.radius, 2.0, atol=0.02)

    def test_sketch_is_closed_chain(self):
        prims = loop_to_sketch(_rounded_rect_loop(20, 10, 2.0, 12))
        for a, b in zip(prims, prims[1:] + prims[:1]):
            assert np.allclose(a.end, b.start, atol=1e-9)


def _rounded_rect_loop(w, h, r, n):
    pts = []
    cx, cy = w / 2 - r, h / 2 - r
    for sx, sy, a0 in ((cx, cy, 0.0), (-cx, cy, 90.0),
                       (-cx, -cy, 180.0), (cx, -cy, 270.0)):
        ang = np.deg2rad(np.linspace(a0, a0 + 90, n, endpoint=False))
        pts += [(sx + r * np.cos(a), sy + r * np.sin(a)) for a in ang]
    return np.array(pts)


# ------------------------------------------------------- plan structure

class TestPlanStructure:
    def test_drilled_plate_plan(self):
        plan = _plan(drilled_plate())
        assert np.isclose(plan.base.length, 5.0)
        assert np.allclose(np.abs(plan.frame_z @ Z), 1.0)
        assert len(plan.holes) == 1
        h = plan.holes[0]
        assert h.diameter == 8.0
        assert h.through is True
        assert len(h.positions) == 1

    def test_counterbore_plan(self):
        plan = _plan(counterbored_plate())
        assert len(plan.holes) == 1
        h = plan.holes[0]
        assert h.diameter == 8.0
        assert h.counterbore_diameter == 12.0
        assert np.isclose(h.counterbore_depth, 2.0)

    def test_bolt_circle_is_one_multi_position_op(self):
        plan = _plan(bolt_circle_plate())
        multi = [h for h in plan.holes if len(h.positions) == 6]
        assert len(multi) == 1
        single = [h for h in plan.holes if len(h.positions) == 1]
        assert len(single) == 1 and single[0].diameter == 10.0

    def test_pocket_plan(self):
        plan = _plan(pocketed_plate())
        assert len(plan.pockets) == 1
        assert np.isclose(plan.pockets[0].depth, 4.0)

    def test_boss_plan(self):
        plan = _plan(stepped_shaft())
        assert len(plan.pads) == 1
        assert np.isclose(plan.pads[0].length, 15.0)
        assert np.isclose(plan.base.length, 20.0)

    def test_slab_fillets_absorbed_into_profile(self):
        plan = _plan(rounded_slab())
        arcs = [p for p in plan.base.profile if isinstance(p, SketchArc)]
        assert len(arcs) == 4
        assert plan.absorbed_features >= 4  # the four fillet features


# --------------------------------------------------- executable semantics

def _rebuild_volume(plan) -> float:
    """Execute the plan with trimesh booleans and return the volume.

    Conventions (must match the planner): frame z up, bottom plane at
    z=0, base occupies [0, L]; pads stack on the base top; pockets,
    counterbores, and blind holes cut downward from the top face;
    through holes pierce everything.
    """
    from shapely.geometry import Polygon

    def poly_of(profile):
        pts = []
        for p in profile:
            pts.extend(p.sample())
        return Polygon(pts)

    def extrude(profile, z0, height):
        m = trimesh.creation.extrude_polygon(poly_of(profile), height=height)
        m.apply_translation([0, 0, z0])
        return m

    L = plan.base.length
    solid = extrude(plan.base.profile, 0.0, L)
    top = L
    for pad in plan.pads:
        solid = solid.union(extrude(pad.profile, L, pad.length))
        top = max(top, L + pad.length)
    for pk in plan.pockets:
        solid = solid.difference(extrude(pk.profile, L - pk.depth,
                                         pk.depth + 1.0))
    for h in plan.holes:
        for (x, y) in h.positions:
            if h.through:
                cyl = trimesh.creation.cylinder(radius=h.diameter / 2,
                                                height=top + 4.0, sections=96)
                cyl.apply_translation([x, y, top / 2])
            else:
                cyl = trimesh.creation.cylinder(radius=h.diameter / 2,
                                                height=h.depth + 1.0, sections=96)
                cyl.apply_translation([x, y, L - h.depth + (h.depth + 1.0) / 2])
            solid = solid.difference(cyl)
            if h.counterbore_diameter:
                cb = trimesh.creation.cylinder(
                    radius=h.counterbore_diameter / 2,
                    height=h.counterbore_depth + 1.0, sections=96)
                cb.apply_translation(
                    [x, y, L - h.counterbore_depth
                     + (h.counterbore_depth + 1.0) / 2])
                solid = solid.difference(cb)
    return float(solid.volume)


class TestRoundTrip:
    @pytest.mark.parametrize("fixture", [
        drilled_plate, counterbored_plate, pocketed_plate,
        stepped_shaft, rounded_slab, bolt_circle_plate,
    ])
    def test_rebuilt_volume_matches_mesh(self, fixture):
        mesh = fixture()
        plan = _plan(mesh)
        rebuilt = _rebuild_volume(plan)
        assert abs(rebuilt - mesh.volume) / mesh.volume < 0.01, \
            f"{fixture.__name__}: rebuilt {rebuilt:.2f} vs mesh {mesh.volume:.2f}"
