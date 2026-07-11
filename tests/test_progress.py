# SPDX-License-Identifier: LGPL-2.1-or-later
"""Progress-callback tests (v0.10), written before the implementation.

The GUI needs stage + fraction feedback to run the pipeline off the main
thread without a frozen window. Contract:

* ``progress(stage: str, fraction: float)`` is optional and None-safe,
* fractions are monotonically nondecreasing within [0, 1] -- even though
  the refinement queue GROWS while being drained (a naive
  processed/total would go backwards when a failed segment spawns
  children),
* the final call reports 1.0,
* stages are human-readable non-empty strings,
* results are identical with and without a callback.
"""

import numpy as np
import trimesh

from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report


class Recorder:
    def __init__(self):
        self.calls: list[tuple[str, float]] = []

    def __call__(self, stage: str, fraction: float) -> None:
        self.calls.append((stage, float(fraction)))


def _mesh():
    return trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)


class TestReconstructProgress:
    def test_callback_receives_monotonic_fractions(self):
        rec = Recorder()
        reconstruct(_mesh(), progress=rec)
        fracs = [f for _, f in rec.calls]
        assert len(fracs) >= 3
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert all(b >= a for a, b in zip(fracs, fracs[1:]))
        assert fracs[-1] == 1.0

    def test_stages_are_meaningful(self):
        rec = Recorder()
        reconstruct(_mesh(), progress=rec)
        assert all(isinstance(s, str) and s for s, _ in rec.calls)

    def test_monotonic_even_when_refinement_spawns_children(self):
        # the rounded slab triggers the curvature second pass: the work
        # queue grows mid-run, the naive fraction would regress
        from .test_robustness import rounded_slab
        rec = Recorder()
        reconstruct(rounded_slab(), progress=rec)
        fracs = [f for _, f in rec.calls]
        assert all(b >= a for a, b in zip(fracs, fracs[1:]))
        assert fracs[-1] == 1.0

    def test_results_identical_with_and_without_callback(self):
        mesh = _mesh()
        a = reconstruct(mesh)
        b = reconstruct(mesh, progress=Recorder())
        assert a.kinds() == b.kinds()
        assert np.isclose(a.coverage, b.coverage)

    def test_none_is_default_and_safe(self):
        rep = reconstruct(_mesh(), progress=None)
        assert rep.kinds() == ["cylinder", "plane", "plane"]


class TestSnapProgress:
    def test_snap_report_progress(self):
        rec = Recorder()
        rep = reconstruct(_mesh())
        snap_report(rep, progress=rec)
        fracs = [f for _, f in rec.calls]
        assert fracs and fracs[-1] == 1.0
        assert all(b >= a for a, b in zip(fracs, fracs[1:]))
