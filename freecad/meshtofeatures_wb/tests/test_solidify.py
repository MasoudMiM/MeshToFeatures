# SPDX-License-Identifier: LGPL-2.1-or-later
"""Deviation correction (solidify): headless plan execution plus the
patch fallback for geometry no analytic feature captured.

The contract: ``plan_to_mesh`` executes the plan with mesh booleans in
world coordinates; ``plan_corrections`` reports the OVER/UNDER components
of rebuild-vs-mesh; ``apply_corrections`` intersects with the source mesh
(all OVER removed at once) and unions the UNDER patches back. On the
issue-#5 corner bracket -- whose freeform diagonal band and base scoops
no primitive fits -- this closes the rebuild from +25% to within ~1% of
the mesh volume.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from freecad.meshtofeatures_wb.core.emission import plan_patches          # noqa: E402
from freecad.meshtofeatures_wb.core.features import detect_features       # noqa: E402
from freecad.meshtofeatures_wb.core.history import plan_history           # noqa: E402
from freecad.meshtofeatures_wb.core.patterns import detect_patterns       # noqa: E402
from freecad.meshtofeatures_wb.core.pipeline import reconstruct           # noqa: E402
from freecad.meshtofeatures_wb.core.snapping import snap_report           # noqa: E402
from freecad.meshtofeatures_wb.core.solidify import (                     # noqa: E402
    apply_corrections, plan_corrections, plan_to_mesh, _frustum)

_STL = Path(__file__).parent / "fixtures" / "Simple_Corner_Bracket.stl"


def _full(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    plan = plan_history(report, feats, detect_patterns(feats), patches)
    return report, feats, plan


class TestFrustum:
    def test_watertight_and_volume(self):
        # cylinder-ish frustum: exact volume pi r^2 h
        m = _frustum(1.0, 1.0, 0.0, 2.0, sections=128)
        assert m.is_watertight
        assert m.volume == pytest.approx(2.0 * np.pi, rel=1e-3)

    def test_apex_ends(self):
        for r_lo, r_hi in ((0.0, 3.0), (3.0, 0.0)):
            m = _frustum(r_lo, r_hi, 0.0, 2.0, sections=64)
            assert m.is_watertight
            # cone volume = (1/3) pi r^2 h
            assert abs(m.volume) == pytest.approx(
                np.pi * 9.0 * 2.0 / 3.0, rel=1e-2)


@pytest.mark.skipif(not _STL.exists(), reason="fixture not present")
class TestBracketCorrection:
    def test_corrections_close_the_volume_gap(self):
        mesh = trimesh.load(str(_STL), force="mesh")
        _, _, plan = _full(mesh)
        solid = plan_to_mesh(plan)
        assert solid is not None and solid.is_watertight
        # pre-correction: the parametric rebuild should not be exact
        # (the bracket has freeform features no analytic primitive captures)
        assert abs(solid.volume / mesh.volume - 1.0) > 0.001, \
            f"rebuild {solid.volume:.1f} too close to mesh {mesh.volume:.1f}"
        corrs = plan_corrections(plan, mesh)
        assert corrs, "expected deviation components on the bracket"
        assert any(c.kind == "cut" for c in corrs)
        assert any(c.kind == "add" for c in corrs)
        assert all(c.mesh.is_watertight for c in corrs)
        corrected = apply_corrections(solid, mesh, corrs)
        ratio = corrected.volume / mesh.volume
        assert abs(ratio - 1.0) < 0.02, \
            f"corrected {corrected.volume:.1f} vs mesh {mesh.volume:.1f}"

    def test_plan_carries_corrections_field(self):
        mesh = trimesh.load(str(_STL), force="mesh")
        _, _, plan = _full(mesh)
        assert hasattr(plan, "corrections")
        assert plan.corrections == []          # filled by the caller


class TestCompletePartNeedsNoCorrection:
    def drilled_plate(self):
        plate = trimesh.creation.box(extents=[60, 40, 10])
        hole = trimesh.creation.cylinder(radius=4.0, height=30.0,
                                         sections=64)
        return plate.difference(hole)

    def test_no_significant_components(self):
        mesh = self.drilled_plate()
        _, _, plan = _full(mesh)
        solid = plan_to_mesh(plan)
        assert solid is not None
        corrs = plan_corrections(plan, mesh)
        total = sum(c.volume for c in corrs)
        assert total < 0.02 * mesh.volume, \
            f"a fully captured part should need ~no patches, got {total:.1f}"
        corrected = apply_corrections(solid, mesh, corrs)
        assert abs(corrected.volume / mesh.volume - 1.0) < 0.01
