# SPDX-License-Identifier: LGPL-2.1-or-later
"""Issue #5 regression: the Thingiverse corner bracket failed with
"history plan failed: degenerate base extent".

The part's flat faces (base, recess floor, walls, top) are chained into one
large segment by gently faceted transition blends, so the curvature split
cannot separate them and only a single plane of the dominant normal cluster
is recognized -- a zero-thickness (degenerate) base extent. The fix peels
exactly-planar sub-regions out of such a blob (:func:`split_by_planes`),
recovering the parallel flats at their distinct offsets.

The fixture STL is vendored from Thingiverse thing:2419704, "Simple Corner
Bracket for 2020 Aluminum Profile" by Aynareth,
https://www.thingiverse.com/thing:2419704, licensed CC-BY 4.0
(https://creativecommons.org/licenses/by/4.0/); the file is unmodified
apart from being stored in this repository. The test skips if the file is
absent.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from freecad.meshtofeatures_wb.core.emission import plan_patches
from freecad.meshtofeatures_wb.core.features import detect_features
from freecad.meshtofeatures_wb.core.history import plan_history
from freecad.meshtofeatures_wb.core.patterns import detect_patterns
from freecad.meshtofeatures_wb.core.pipeline import reconstruct
from freecad.meshtofeatures_wb.core.snapping import snap_report

_STL = Path(__file__).parent / "fixtures" / "Simple_Corner_Bracket.stl"


def _load():
    if not _STL.exists():
        pytest.skip(f"fixture not present: {_STL}")
    return trimesh.load(str(_STL), force="mesh")


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


class TestCornerBracket:
    def test_plan_succeeds_no_degenerate_extent(self):
        # The reported failure: plan_history raised "degenerate base extent".
        # With the fix the parallel flats are recovered and the base extent
        # spans the part's true height.
        mesh = _load()
        _, _, plan = _full(mesh)
        assert plan.base is not None
        assert plan.base.length > 0.0
        assert np.isclose(plan.base.length, 20.0, atol=0.5)

    def test_flats_recovered_not_one_blob(self):
        # The flat faces must be recognized as distinct planes at several
        # offsets along the base axis, not swallowed by one blob. Pre-fix,
        # only a single plane of the dominant cluster was recognized.
        mesh = _load()
        report, _, _ = _full(mesh)
        assert report.coverage > 0.6
        from freecad.meshtofeatures_wb.core.primitives import Plane
        offsets = {round(float(s.fit.primitive.point @ s.fit.primitive.normal), 1)
                   for s in report.surfaces
                   if isinstance(s.fit.primitive, Plane)
                   and abs(float(s.fit.primitive.normal[2])) > 0.9}
        assert len(offsets) >= 3, f"expected several +Z flats, got {offsets}"

    def test_features_detected(self):
        # The countersunk through hole and the four wall (cross-axis) holes
        # are detected, and nothing is silently dropped.
        mesh = _load()
        _, feats, plan = _full(mesh)
        csink = [h for h in plan.holes if h.countersink_diameter]
        assert len(csink) == 1
        assert len(plan.cross_holes) == 4
        assert plan.unplanned == []

    def test_blend_features_recovered(self):
        # The rim blend bands are peeled as straight-spine channels and at
        # least one survives all the way into the build plan as a fillet or
        # chamfer operation (pre-fix: the bands stayed unrecognized and the
        # rebuild kept every sharp corner, ~25% over-volume).
        mesh = _load()
        _, feats, plan = _full(mesh)
        assert feats.by_kind("fillet") or feats.by_kind("chamfer"), \
            "no blend feature recognized on the bracket"
        assert (len(plan.fillets) + len(plan.chamfers)
                + plan.absorbed_features) >= 1


class TestDegenerateFilletFrames:
    """Issue #8: the diagonal band fillets' neighbour search could find
    two segments of the same face (n_a == n_b) or a face that only
    touches the band at its end -- a frame under which the executor's
    orthonormalisation collapses and the blend is undressable. Such
    blends must never be emitted as FilletOps; they are reported as
    undressable instead of silently absent."""

    def test_no_planned_fillet_has_degenerate_frame(self):
        mesh = _load()
        _, _, plan = _full(mesh)
        for op in plan.fillets:
            d = np.asarray(op.direction, dtype=float)
            d = d / np.linalg.norm(d)
            cross = np.cross(np.asarray(op.n_a, dtype=float),
                             np.asarray(op.n_b, dtype=float))
            assert abs(float(cross @ d)) > 0.1, \
                f"undressable frame planned for {op.label}: " \
                f"(n_a x n_b) . d = {float(cross @ d):.4f}"

    def test_diagonal_band_fillets_not_silently_dropped(self):
        # Every detected fillet feature must be accounted for: planned
        # (dressable frame), undressable (degenerate frame), absorbed
        # (vertical, in the base profile), or consumed by a pattern --
        # never silently unplanned.
        mesh = _load()
        _, feats, plan = _full(mesh)
        n = len(feats.by_kind("fillet"))
        assert n >= 4, "expected the bracket's diagonal band fillets"
        accounted = len(plan.fillets) + len(plan.undressable)
        assert accounted <= n
        assert n - accounted <= plan.absorbed_features
        assert plan.unplanned == []
        assert len(plan.fillets) >= 1, "all fillets rejected"

    def test_undressable_reported_in_result_lines(self, monkeypatch):
        import sys
        import types
        # ui.py imports FreeCAD/FreeCADGui/PySide at module level;
        # result_lines itself is pure, so bare stubs suffice
        monkeypatch.setitem(sys.modules, "FreeCAD", types.ModuleType("FreeCAD"))
        monkeypatch.setitem(sys.modules, "FreeCADGui",
                            types.ModuleType("FreeCADGui"))
        pyside = types.ModuleType("PySide")
        qcore = types.ModuleType("PySide.QtCore")
        qcore.QObject = object          # _Runner subclasses it at import
        qcore.QTimer = object
        pyside.QtCore = qcore
        pyside.QtWidgets = types.ModuleType("PySide.QtWidgets")
        monkeypatch.setitem(sys.modules, "PySide", pyside)
        monkeypatch.setitem(sys.modules, "PySide.QtCore", qcore)
        monkeypatch.setitem(sys.modules, "PySide.QtWidgets",
                            pyside.QtWidgets)
        from freecad.meshtofeatures_wb.ui import result_lines
        mesh = _load()
        report = snap_report(reconstruct(mesh)).report
        patches = plan_patches(report)
        feats = detect_features(report, patches)
        plan = plan_history(report, feats, detect_patterns(feats), patches)
        if not plan.undressable:
            pytest.skip("no undressable blends in this environment")
        res = {"report": report, "actions": [], "patches": patches,
               "features": feats, "patterns": detect_patterns(feats),
               "plan": plan, "plan_error": None}
        lines = result_lines(res)
        assert any("UNDRASSABLE" in ln for ln in lines), lines
