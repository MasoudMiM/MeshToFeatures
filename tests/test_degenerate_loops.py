# SPDX-License-Identifier: LGPL-2.1-or-later
"""Degenerate-loop guard: the planner must never emit a pocket/step/pad
profile that cannot form a closed sketch wire.

Field STLs with noisy or subdivided faces (20mm-xyz-cube, origin_inside)
segment into sliver planes whose boundary loop collapses to one or two
points. Emitting a pocket from such a loop built a <3-point wire that
crashed the boolean rebuild ("linearring requires 4 coordinates") and
would crash the FreeCAD sketch. `_valid_loop` gates every emission site
so a sliver drops one feature instead of blanking the part.
"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from meshtofeatures.history import _valid_loop            # noqa: E402
from .test_adversarial import _plan, _rebuild_mesh, _to_world   # noqa: E402


class TestValidLoop:
    def test_rejects_too_few_vertices(self):
        assert not _valid_loop(np.array([[0.0, 0.0], [1.0, 0.0]]), 1e-3)
        assert not _valid_loop(np.zeros((1, 2)), 1e-3)

    def test_rejects_duplicate_vertices(self):
        # three rows but only two distinct points
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        assert not _valid_loop(loop, 1e-3)

    def test_rejects_collinear_zero_area(self):
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        assert not _valid_loop(loop, 1e-3)

    def test_accepts_triangle(self):
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert _valid_loop(loop, 1e-3)

    def test_accepts_square(self):
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert _valid_loop(loop, 1e-3)

    def test_area_threshold_scales_with_tol(self):
        # a 4-point loop enclosing near-zero area is rejected
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1e-6], [0.0, 1e-6]])
        assert not _valid_loop(loop, 1e-2)


def _stepped_block():
    """Base with a raised deck over half the top (produces a step)."""
    base = trimesh.creation.box(extents=[40, 30, 6])
    base.apply_translation([0, 0, 3])
    deck = trimesh.creation.box(extents=[40, 14, 4])
    deck.apply_translation([0, -8, 8])
    return base.union(deck)


def _pocketed_block():
    base = trimesh.creation.box(extents=[40, 30, 10])
    base.apply_translation([0, 0, 5])
    cut = trimesh.creation.box(extents=[16, 10, 4])
    cut.apply_translation([0, 0, 10])
    return base.difference(cut)


class TestNoDegenerateProfiles:
    """Invariant across the emission sites: every profile a plan carries
    has at least three primitives and rebuilds without crashing."""

    @pytest.mark.parametrize("factory", [_stepped_block, _pocketed_block])
    def test_profiles_have_three_or_more_primitives(self, factory):
        _, _, plan = _plan(factory())
        for pk in plan.pockets:
            assert len(pk.profile) >= 3, f"degenerate pocket: {pk.label}"
        for pad in plan.pads:
            assert len(pad.profile) >= 1  # circles are one primitive
        # and the whole thing rebuilds
        reb = _to_world(plan, _rebuild_mesh(plan))
        assert reb.volume > 0
