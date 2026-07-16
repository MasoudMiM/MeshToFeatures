# SPDX-License-Identifier: LGPL-2.1-or-later
"""Robustness tests on real STL parts pulled from the trimesh model repo.

These are integration smoke tests: a real machined part must flow through
the whole pipeline without error, and the new feature passes (cone fitting,
countersink detection, blind cross-hole planning) must not false-positive on
geometry that has none of those features. Network-guarded: skipped offline.
"""

import io
import urllib.request

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.history import plan_history
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

_BASE = "https://raw.githubusercontent.com/mikedh/trimesh/main/models/"


def _load(name):
    try:
        with urllib.request.urlopen(_BASE + name, timeout=20) as r:
            data = r.read()
    except Exception as exc:                       # noqa: BLE001 - offline CI
        pytest.skip(f"could not fetch {name}: {exc}")
    ext = name.rsplit(".", 1)[-1].lower()
    mesh = trimesh.load(io.BytesIO(data), file_type=ext)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


class TestFeatureTypeStl:
    """featuretype.STL: a prismatic machined part (holes, counterbores,
    fillets, pockets) -- but NO conical features."""

    def test_pipeline_runs_and_covers(self):
        mesh = _load("featuretype.STL")
        report, feats, plan = _full(mesh)
        assert report.coverage > 0.95
        # a plan is produced with real features and nothing silently dropped
        assert (plan.holes or plan.pockets or plan.cross_holes)
        assert plan.unplanned == []

    def test_no_spurious_countersinks(self):
        # the part has no cones; the countersink pass must not invent any
        mesh = _load("featuretype.STL")
        report, feats, _ = _full(mesh)
        assert not any(s.fit.primitive.kind == "cone"
                       for s in report.surfaces)
        assert not any(h.params.get("countersink")
                       for h in feats.by_kind("hole"))

    def test_no_spurious_blind_cross_holes(self):
        # every planned cross-hole must be geometrically justified: a blind
        # one must carry a positive depth and a unit entry direction
        mesh = _load("featuretype.STL")
        _, _, plan = _full(mesh)
        for ch in plan.cross_holes:
            if not ch.through:
                assert ch.depth is not None and ch.depth > 0
                d = np.asarray(ch.entry_direction, dtype=float)
                assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-6)
