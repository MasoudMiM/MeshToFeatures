# SPDX-License-Identifier: LGPL-2.1-or-later
"""Graceful over-complexity handling.

A faceted, terraced, or organic export can have planar facets yet no
sensible single-axis prismatic reconstruction -- the field's box.STL
segments into ~400 planes at ~15 levels/direction and yields 97 phantom
"steps" (a nonsense body the executor can only limp through via
per-feature rollback). Rather than emit that, `plan_history` declines
with a clear message once the step count exceeds any plausible prismatic
part, exactly like the existing "no planar surfaces" / "degenerate base
extent" declines. Normal stepped parts (a handful of levels) are
untouched.
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.history import plan_history, MAX_STEP_LEVELS   # noqa: E402
from .test_adversarial import _plan                                # noqa: E402


def stair(n):
    """n stacked slabs shrinking from one side, rising in z (wide in y so
    the extrusion axis stays z) -> n-1 outline-touching step levels."""
    m = None
    for i in range(n):
        w = 100.0 - 1.6 * i
        slab = trimesh.creation.box(extents=[w, 200.0, 1.5])
        slab.apply_translation([-50.0 + w / 2.0, 0.0, 0.75 + 1.5 * i])
        m = slab if m is None else m.union(slab)
    return m


class TestOverComplexity:
    def test_threshold_is_generous(self):
        # no ordinary prismatic part has this many distinct step levels
        assert MAX_STEP_LEVELS >= 16

    def test_over_complex_is_declined(self):
        mesh = stair(MAX_STEP_LEVELS + 12)          # ~ MAX+11 steps
        with pytest.raises(ValueError, match="complex|prismatic|level"):
            _plan(mesh)

    def test_modest_steps_not_declined(self):
        mesh = stair(20)                            # 19 real steps
        _, _, plan = _plan(mesh)
        assert 0 < len(plan.step_labels) <= MAX_STEP_LEVELS
