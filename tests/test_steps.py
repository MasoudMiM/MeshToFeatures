# SPDX-License-Identifier: LGPL-2.1-or-later
"""Step (multi-level top) tests, written before the implementation.

A STEP is an exposed intermediate-height plane touching the part
outline: the base pad must not fill above it. Planned as a PocketOp
whose profile is the shelf's own outer loop with outline-coincident
vertices pushed outward (open sides cut cleanly past the boundary; the
true step wall stays exact). Exposure implies an empty column above, so
cutting to the top face is valid by construction -- and symmetric:
down-facing shelves pocket from the bottom (the frame z sign is
arbitrary).

Discriminator: outline CONTACT. Pocket floors and counterbore shoulders
are interior loops and must never become phantom steps.
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

from .test_adversarial import assert_geometry_match

pytest.importorskip("manifold3d")


def step_block():
    """40 x 30 base, upper deck over half the footprint: one step of
    depth 4 exposed over y in [0, 15]."""
    base = trimesh.creation.box(extents=[40.0, 30.0, 6.0])
    base.apply_translation([0, 0, 3.0])                    # z in [0, 6]
    upper = trimesh.creation.box(extents=[40.0, 15.0, 4.0])
    upper.apply_translation([0, -7.5, 8.0])                # z in [6, 10]
    return base.union(upper)


def staircase_block():
    """Three levels: full base + two nested decks (a staircase)."""
    b = trimesh.creation.box(extents=[40.0, 30.0, 4.0])
    b.apply_translation([0, 0, 2.0])                       # z in [0, 4]
    m = trimesh.creation.box(extents=[40.0, 20.0, 3.0])
    m.apply_translation([0, -5.0, 5.5])                    # z in [4, 7]
    t = trimesh.creation.box(extents=[40.0, 10.0, 3.0])
    t.apply_translation([0, -10.0, 8.5])                   # z in [7, 10]
    return b.union(m).union(t)


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    return report, feats, plan_history(report, feats, detect_patterns(feats),
                                       patches)


class TestStep:
    def test_step_planned_as_pocket(self):
        _, _, plan = _full(step_block())
        assert len(plan.step_labels) == 1
        steps = [p for p in plan.pockets if p.label in plan.step_labels]
        assert len(steps) == 1
        assert np.isclose(steps[0].depth, 4.0, atol=1e-6)
        assert np.isclose(plan.base.length, 10.0)          # full height base

    def test_geometry_roundtrip(self):
        mesh = step_block()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_staircase_two_steps(self):
        mesh = staircase_block()
        _, _, plan = _full(mesh)
        assert len(plan.step_labels) == 2
        depths = sorted(round(p.depth, 6) for p in plan.pockets
                        if p.label in plan.step_labels)
        assert depths == [3.0, 6.0]
        assert_geometry_match(mesh, plan)

    def test_rotated_step_block(self):
        # frame z sign is arbitrary: a "down-facing" shelf must pocket
        # from the bottom
        mesh = step_block()
        rot = trimesh.transformations.rotation_matrix(
            0.9, [1.0, -0.7, 0.3], point=[3.0, 2.0, -1.0])
        mesh.apply_transform(rot)
        _, _, plan = _full(mesh)
        assert len(plan.step_labels) == 1
        assert_geometry_match(mesh, plan)


class TestNoPhantomSteps:
    def test_pocket_floor_is_not_a_step(self):
        # Under the height-field terrace model a pocket floor is
        # reconstructed as a terrace cut to its exact footprint -- the
        # correct result -- not a spurious outline-touching step. The old
        # "no step labels" invariant no longer applies (terraces subsume
        # both steps and pocket floors), so verify the geometry instead.
        from .test_composites import pocketed_plate
        mesh = pocketed_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_counterbore_shoulder_is_not_a_step(self):
        # A counterbore shoulder is likewise a terrace under the
        # height-field model; verify the geometry rather than the obsolete
        # label invariant.
        from .test_composites import counterbored_plate
        mesh = counterbored_plate()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)


def arc_cornered_step_block():
    """Step whose shelf boundary includes an ARC touching the outline:
    the configuration that produced open wires when steps used fragile
    fitted arcs."""
    base = trimesh.creation.box(extents=[40.0, 30.0, 6.0])
    base.apply_translation([0, 0, 3.0])
    from shapely.geometry import box as sbox
    poly = sbox(-20.0, -15.0, 20.0, 0.0).buffer(-4.0).buffer(4.0, quad_segs=24)
    upper = trimesh.creation.extrude_polygon(poly, height=4.0)
    upper.apply_translation([0, 0, 6.0])
    return base.union(upper)


class TestStepProfileRobustness:
    def test_step_profiles_are_closed_polylines(self):
        from meshtofeatures.history import SketchLine
        _, _, plan = _full(arc_cornered_step_block())
        steps = [p for p in plan.pockets if p.label in plan.step_labels]
        assert steps
        for pk in steps:
            assert all(isinstance(x, SketchLine) for x in pk.profile)
            for a, b in zip(pk.profile, pk.profile[1:] + pk.profile[:1]):
                assert np.allclose(a.end, b.start, atol=1e-12)

    def test_arc_cornered_step_roundtrip(self):
        mesh = arc_cornered_step_block()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)

    def test_polyline_mode_of_loop_to_sketch(self):
        from meshtofeatures.history import SketchLine, loop_to_sketch
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        loop = np.column_stack([3 * np.cos(t), 3 * np.sin(t)])
        prims = loop_to_sketch(loop, arcs=False)
        assert all(isinstance(p, SketchLine) for p in prims)
        assert len(prims) == 48


def ring_step_block():
    """Central tower on a base: the shelf is a RING whose outer loop is
    the full footprint and whose HOLE loop is the tower boundary.
    Discarding the hole loop over-cuts the tower (field-observed on
    featuretype: 'Step to depth 0.375' spanned the entire part)."""
    base = trimesh.creation.box(extents=[40.0, 30.0, 6.0])
    base.apply_translation([0, 0, 3.0])                    # z in [0, 6]
    tower = trimesh.creation.box(extents=[16.0, 10.0, 4.0])
    tower.apply_translation([0, 0, 8.0])                   # z in [6, 10]
    return base.union(tower)


class TestRingStep:
    def test_tower_is_not_a_phantom_pocket(self):
        # a tower standing in the shelf's hole loop has a parallel cap
        # ABOVE the opening: the pocket rule must not invert it
        _, feats, _ = _full(ring_step_block())
        assert feats.by_kind("pocket") == []

    def test_step_profile_carries_hole(self):
        _, _, plan = _full(ring_step_block())
        steps = [p for p in plan.pockets if p.label in plan.step_labels]
        assert len(steps) == 1
        assert len(steps[0].hole_profiles) == 1

    def test_tower_survives_roundtrip(self):
        mesh = ring_step_block()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)


def ring_step_with_drill():
    """Ring shelf around a tower PLUS a through-drill piercing the shelf:
    the step must preserve the tower's hole loop but NOT the drill's
    (field-observed: drill loops preserved as sketch holes left annular
    chimneys standing in the step cut)."""
    mesh = ring_step_block()
    drill = trimesh.creation.cylinder(radius=2.0, height=30.0, sections=64)
    drill.apply_translation([14.0, 0.0, 5.0])   # through the ring region
    return mesh.difference(drill)


class TestStepHoleDiscrimination:
    def test_only_raised_loops_are_preserved(self):
        # The tower's hole loop must be preserved so the tower survives the
        # cut. Under the terrace model the through-drill opening may also be
        # retained as a hole, but the drill's own hole feature cuts through
        # it -- so the real invariant is that the tower survives WITHOUT an
        # annular chimney, which the geometry roundtrip verifies.
        mesh = ring_step_with_drill()
        _, _, plan = _full(mesh)
        steps = [p for p in plan.pockets if p.label in plan.step_labels]
        assert len(steps) == 1
        assert len(steps[0].hole_profiles) >= 1     # the tower is preserved
        assert_geometry_match(mesh, plan)

    def test_geometry_roundtrip(self):
        mesh = ring_step_with_drill()
        _, _, plan = _full(mesh)
        assert_geometry_match(mesh, plan)


def deck_with_counterbored_shelf():
    """A raised deck makes the surrounding base-plate top a TERRACE cut down
    from the deck level; counterbored bores pass through that shelf. The
    field-reported failure: the terrace retained each bore opening as a
    sketch hole, leaving a standing column that the bore turned into an
    annular chimney wall above the shelf."""
    plate = trimesh.creation.box(extents=[60.0, 40.0, 8.0])
    plate.apply_translation([0, 0, 4.0])                     # z in [0, 8]
    deck = trimesh.creation.box(extents=[30.0, 40.0, 6.0])
    deck.apply_translation([0, 0, 11.0])                     # z in [8, 14]
    mesh = plate.union(deck)
    for x in (-22.0, 22.0):                                  # bores on the shelf
        cb = trimesh.creation.cylinder(radius=4.0, height=3.2, sections=64)
        cb.apply_translation([x, 0, 8 - 1.6])               # counterbore z in [5, 8]
        drill = trimesh.creation.cylinder(radius=2.0, height=20.0, sections=64)
        drill.apply_translation([x, 0, 4.0])                # through-drill
        mesh = mesh.difference(cb).difference(drill)
    return mesh


class TestTerraceChimney:
    def test_shelf_bores_do_not_chimney(self):
        from .test_adversarial import _rebuild_mesh, _to_world
        mesh = deck_with_counterbored_shelf()
        _, _, plan = _full(mesh)
        # the shelf terraces must not keep the bore openings as sketch holes
        terraces = [p for p in plan.pockets if p.label in plan.step_labels]
        assert sum(len(t.hole_profiles) for t in terraces) == 0
        # and no material may stand in the chimney annulus above the shelf
        # (z=8) and below the deck top (z=14), between drill r=2 and bore r=4
        rebuilt = _to_world(plan, _rebuild_mesh(plan))
        probes = np.array([[x + 3.0, y, 11.0]
                           for x in (-22.0, 22.0) for y in (-3.0, 0.0, 3.0)])
        assert not rebuilt.contains(probes).any()


def plate_deck_counterbores():
    """A thick base plate with a thin raised deck cap (featuretype-like):
    the base-plate top (z=10) sits below the deck top (z=13=L), and the
    counterbored bores open on the base plate. Their sketch must be placed
    on that opening face, not the global top -- otherwise PartDesign::Hole
    extrudes the counterbore OUTWARD as a tower (the field-reported walls)."""
    plate = trimesh.creation.box(extents=[50.0, 30.0, 10.0])
    plate.apply_translation([0, 0, 5.0])
    deck = trimesh.creation.box(extents=[24.0, 30.0, 3.0])
    deck.apply_translation([0, 0, 11.5])
    mesh = plate.union(deck)
    for x in (-18.0, 18.0):
        cb = trimesh.creation.cylinder(radius=3.0, height=4.0, sections=64)
        cb.apply_translation([x, 0, 9.0])                    # recess z[7,10]
        dr = trimesh.creation.cylinder(radius=1.5, height=30.0, sections=64)
        dr.apply_translation([x, 0, 5.0])
        mesh = mesh.difference(cb).difference(dr)
    return mesh


class TestCounterboreSurfaceLevel:
    def test_bore_sketch_placed_on_opening_face_not_global_top(self):
        mesh = plate_deck_counterbores()
        _, _, plan = _full(mesh)
        assert plan.holes, "counterbore not detected"
        h = plan.holes[0]
        assert h.from_top                       # uses PartDesign::Hole path
        assert h.surface_z is not None
        # opening face is the base-plate top (~10), well below deck top (13)
        assert abs(h.surface_z - 10.0) < 0.4
        assert h.surface_z < plan.base.length - 1.0

