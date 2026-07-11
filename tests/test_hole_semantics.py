# SPDX-License-Identifier: LGPL-2.1-or-later
"""v0.11 tests, written before the implementation.

(d) PartDesign Hole emission splits into a PURE property mapping
    (HoleOp -> FreeCAD Hole feature properties) tested here, and a thin
    executor verified by the smoke test. The mapping must encode:
    through vs blind, counterbore, and side-ness (Reversed).

Bonus: STANDARD-SIZE identification -- a 6.6 mm hole is an M6 clearance
    hole and a 5.0 mm hole is an M6 tap drill; recovering the standard is
    recovering intent. Must not hallucinate on non-standard sizes.
"""

import numpy as np
import pytest

from meshtofeatures.history import HoleOp, hole_op_properties
from meshtofeatures.standards import identify_metric


class TestHolePropertyMapping:
    def test_through_hole(self):
        op = HoleOp(diameter=8.0, through=True, depth=5.0, positions=[(0, 0)])
        p = hole_op_properties(op)
        assert p["Diameter"] == 8.0
        assert p["DepthType"] == "ThroughAll"
        assert p["HoleCutType"] == "None"
        assert p["Reversed"] is False

    def test_blind_hole(self):
        op = HoleOp(diameter=6.0, through=False, depth=4.0, positions=[(0, 0)])
        p = hole_op_properties(op)
        assert p["DepthType"] == "Dimension"
        assert p["Depth"] == 4.0
        # our recognized blind holes are flat-bottomed cylinders
        assert p["DrillPoint"] == "Flat"

    def test_counterbore(self):
        op = HoleOp(diameter=8.0, through=True, depth=5.0, positions=[(0, 0)],
                    counterbore_diameter=12.0, counterbore_depth=2.0)
        p = hole_op_properties(op)
        assert p["HoleCutType"] == "Counterbore"
        assert p["HoleCutDiameter"] == 12.0
        assert p["HoleCutDepth"] == 2.0

    def test_bottom_side_sets_reversed(self):
        op = HoleOp(diameter=8.0, through=False, depth=3.0,
                    positions=[(0, 0)], from_top=False)
        assert hole_op_properties(op)["Reversed"] is True

    def test_plain_hole_has_no_cut_leakage(self):
        op = HoleOp(diameter=8.0, through=True, depth=5.0, positions=[(0, 0)])
        p = hole_op_properties(op)
        assert "HoleCutDiameter" not in p and "HoleCutDepth" not in p


class TestStandardIdentification:
    @pytest.mark.parametrize("d,expected", [
        (6.6, "M6 clearance"),
        (9.0, "M8 clearance"),
        (11.0, "M10 clearance"),
        (5.0, "M6 tap drill"),
        (6.8, "M8 tap drill"),
        (8.5, "M10 tap drill"),
    ])
    def test_exact_table_values(self, d, expected):
        assert identify_metric(d) == expected

    def test_tolerance_window(self):
        # real STL data lands near, not on, the table value
        assert identify_metric(6.62) == "M6 clearance"
        assert identify_metric(9.128, rtol=0.02) == "M8 clearance"

    def test_non_standard_returns_none(self):
        assert identify_metric(7.77) is None
        assert identify_metric(0.4) is None

    def test_no_cross_family_confusion(self):
        # 3.4 (M3 clearance) vs 3.3 (M4 tap): distinct at default tolerance
        assert identify_metric(3.4) == "M3 clearance"
        assert identify_metric(3.3) == "M4 tap drill"

    def test_annotation_reaches_feature_descriptions(self):
        import trimesh
        pytest.importorskip("manifold3d")
        from meshtofeatures.emission import plan_patches
        from meshtofeatures.features import detect_features
        from meshtofeatures.pipeline import reconstruct
        from meshtofeatures.snapping import snap_report
        # radius 4.5 -> d9.0 = M8 clearance; 4.5 sits ON the snap grid so
        # snapping cannot nudge it out of the standards window. (A d6.6
        # drill CAN be snapped to 6.5 -- grid boundary -- and then miss
        # the window: standards-aware snapping is a noted future step.)
        plate = trimesh.creation.box(extents=[40.0, 30.0, 5.0])
        drill = trimesh.creation.cylinder(radius=4.5, height=20.0, sections=64)
        drill.apply_translation([10.0, 5.0, 0.0])
        mesh = plate.difference(drill)
        report = snap_report(reconstruct(mesh)).report
        feats = detect_features(report, plan_patches(report))
        h = feats.by_kind("hole")[0]
        assert h.params.get("standard") == "M8 clearance"
        assert "M8 clearance" in h.description
