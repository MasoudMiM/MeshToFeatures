# SPDX-License-Identifier: LGPL-2.1-or-later
"""Per-feature deviation report (phase 3): pure-headless tests.

Covers the report plumbing (``core.status``): the ledger's merge rules, the
correction-to-feature attribution, the row builder over real plans (corpus
plate + the issue-#5 corner bracket), and the report-view formatting. No
FreeCAD, no executor -- build.py wires this same module in and feeds it the
runtime ledger.
"""

import numpy as np
import pytest
import trimesh

from freecad.meshtofeatures_wb.core.emission import plan_patches
from freecad.meshtofeatures_wb.core.features import detect_features
from freecad.meshtofeatures_wb.core.history import plan_history
from freecad.meshtofeatures_wb.core.patterns import detect_patterns
from freecad.meshtofeatures_wb.core.pipeline import reconstruct
from freecad.meshtofeatures_wb.core.snapping import snap_report
from freecad.meshtofeatures_wb.core.solidify import CorrectionOp
from freecad.meshtofeatures_wb.core.status import (
    ExecutionLedger,
    attribute_corrections,
    feature_report,
    format_feature_report,
    op_sites,
    plan_ops,
)

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _plan(name):
    mesh = trimesh.load(str(FIXTURES / name), force="mesh")
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    return plan_history(report, feats, detect_patterns(feats), patches)


@pytest.fixture(scope="module")
def cb_plan():
    return _plan("corpus_plate_counterbored.stl")


@pytest.fixture(scope="module")
def bracket_plan():
    return _plan("Simple_Corner_Bracket.stl")


def _world(plan, x, y, z):
    return (np.asarray(plan.frame_origin, float)
            + x * np.asarray(plan.frame_x, float)
            + y * np.asarray(plan.frame_y, float)
            + z * np.asarray(plan.frame_z, float))


class TestLedger:
    def test_default_rebuilt(self):
        led = ExecutionLedger()
        assert led.get("pocket", 3) == ("rebuilt", "")

    def test_records_win_by_severity(self):
        led = ExecutionLedger()
        led.record("hole", 0, "fallback", "pocket cut")
        led.record("hole", 0, "deferred")
        assert led.get("hole", 0)[0] == "deferred"
        led.record("hole", 0, "recovered", "retry")
        assert led.get("hole", 0)[0] == "deferred"
        led.record("hole", 0, "skipped", "failed to compute")
        assert led.get("hole", 0)[0] == "skipped"

    def test_details_merge_on_equal_severity(self):
        led = ExecutionLedger()
        led.record("hole", 1, "skipped", "cone site A")
        led.record("hole", 1, "skipped", "cone site B")
        status, detail = led.get("hole", 1)
        assert status == "skipped"
        assert "site A" in detail and "site B" in detail

    def test_unknown_status_rejected(self):
        led = ExecutionLedger()
        with pytest.raises(ValueError):
            led.record("pocket", 0, "exploded")


class TestRowBuilding:
    def test_one_row_per_planned_op(self, cb_plan):
        rows, _ = feature_report(cb_plan)
        n_planned = len(list(plan_ops(cb_plan)))
        assert len(rows) == n_planned + len(cb_plan.unplanned) \
            + len(cb_plan.undressable)
        executed = [r for r in rows
                    if r.kind in ("hole", "pocket", "pad", "base",
                                  "cross-hole", "cone", "sphere", "torus")]
        assert all(r.status == "rebuilt" for r in executed)

    def test_sites_align_with_ops(self, cb_plan, bracket_plan):
        for plan in (cb_plan, bracket_plan):
            sites = op_sites(plan)
            assert len(sites) == len(list(plan_ops(plan)))
            assert all(len(s) >= 1 for s in sites)

    def test_ledger_status_flows_to_rows(self, cb_plan):
        led = ExecutionLedger()
        led.record("hole", 0, "skipped", "failed to compute")
        led.record("hole", 1, "geometric", "mesh-fit edge")
        rows, _ = feature_report(cb_plan, led)
        holes = [r for r in rows if r.kind == "hole"]
        assert holes[0].status == "skipped"
        assert "failed" in holes[0].detail
        assert holes[1].status == "geometric"

    def test_sub_op_failure_cannot_mask_parent_deferral(self, cb_plan):
        """Review finding 1: a countersink skip must NOT overwrite the
        hole's deferred status -- the bore may be fully recovered while the
        cone is gone; the loss belongs in the detail, loud but honest."""
        led = ExecutionLedger()
        led.record("hole", 0, "deferred", "rebuilt at the end of the chain")
        led.record("countersink", 0, "skipped", "failed to compute")
        led.record("counterbore", 0, "skipped", "failed to compute")
        rows, _ = feature_report(cb_plan, led)
        hole0 = [r for r in rows if r.kind == "hole"][0]
        assert hole0.status == "deferred"
        assert "SKIPPED" in hole0.detail and "[countersink]" in hole0.detail
        assert "[counterbore]" in hole0.detail
        # the other hole is untouched
        other = [r for r in rows if r.kind == "hole"][1]
        assert other.status == "rebuilt" and other.detail == ""

    def test_parent_status_unchanged_by_sub_rollup(self, cb_plan):
        led = ExecutionLedger()
        led.record("countersink", 1, "skipped", "degenerate cone")
        rows, _ = feature_report(cb_plan, led)
        hole1 = [r for r in rows if r.kind == "hole"][1]
        assert hole1.status == "rebuilt"
        assert "degenerate cone" in hole1.detail

    def test_unplanned_rows_render(self, cb_plan):
        cb_plan.unplanned.append("test slot")
        try:
            rows, _ = feature_report(cb_plan)
            unpl = [r for r in rows if r.status == "unplanned"]
            assert any("test slot" == r.label for r in unpl)
        finally:
            cb_plan.unplanned.remove("test slot")


class TestAttribution:
    def test_component_lands_on_nearest_feature(self, cb_plan):
        h = cb_plan.holes[0]
        z = h.surface_z if h.surface_z is not None else cb_plan.base.length
        c = _world(cb_plan, h.positions[0][0], h.positions[0][1],
                   float(z) - 0.5)
        comp = trimesh.creation.box(extents=[0.9] * 3)
        comp.apply_translation(c)
        cb_plan.corrections = [CorrectionOp(kind="add", mesh=comp,
                                            volume=1.234)]
        try:
            dev, (un_u, un_o) = attribute_corrections(cb_plan)
            ops = list(plan_ops(cb_plan))
            hole_idx = [i for i, (k, _, _) in enumerate(ops)
                        if k == "hole"][0]
            assert dev[hole_idx] == pytest.approx(1.234)
            assert sum(abs(d) for i, d in enumerate(dev) if i != hole_idx) \
                == pytest.approx(0.0)
            assert un_u == 0.0 and un_o == 0.0
        finally:
            cb_plan.corrections = []

    def test_far_component_is_freeform(self, cb_plan):
        c = _world(cb_plan, 0.0, 0.0, cb_plan.base.length + 5000.0)
        comp = trimesh.creation.box(extents=[0.1] * 3)
        comp.apply_translation(c)
        cb_plan.corrections = [CorrectionOp(kind="cut", mesh=comp,
                                            volume=5.5)]
        try:
            dev, (un_u, un_o) = attribute_corrections(cb_plan)
            assert sum(abs(d) for d in dev) == pytest.approx(0.0)
            assert un_o == pytest.approx(5.5) and un_u == 0.0
        finally:
            cb_plan.corrections = []

    def test_sign_convention(self, cb_plan):
        h = cb_plan.holes[0]
        z = h.surface_z if h.surface_z is not None else cb_plan.base.length
        c = _world(cb_plan, h.positions[0][0], h.positions[0][1],
                   float(z) - 0.5)
        over = trimesh.creation.box(extents=[0.9] * 3)
        over.apply_translation(c)
        cb_plan.corrections = [CorrectionOp(kind="cut", mesh=over,
                                            volume=2.0)]
        try:
            dev, _ = attribute_corrections(cb_plan)
            ops = list(plan_ops(cb_plan))
            hole_idx = [i for i, (k, _, _) in enumerate(ops)
                        if k == "hole"][0]
            assert dev[hole_idx] == pytest.approx(-2.0)
        finally:
            cb_plan.corrections = []


class TestFormat:
    def test_table_lines(self, bracket_plan):
        led = ExecutionLedger()
        led.record("pocket", 0, "skipped", "failed to compute")
        rows, (un_u, un_o) = feature_report(bracket_plan, led)
        lines = format_feature_report(rows, src_volume=20000.0,
                                      body_volume=19800.0,
                                      corrected_volume=20010.0,
                                      unattributed=(1.5, 0.7),
                                      name="Rebuilt_bracket")
        text = "\n".join(lines)
        assert "FEATURE REPORT" in text and "body 99.00%" in text
        assert "corrected 100.05%" in text
        assert "SKIPPED" in text
        assert "freeform deviation" in text
        assert len(lines) >= len(rows) + 1
        # every executed (planned) feature prints exactly one line: no
        # silent drops -- that was the whole point of phase 3
        for r in rows:
            if r.kind in ("hole", "pocket", "pad", "fillet", "chamfer"):
                assert r.label in text or r.label == ""
