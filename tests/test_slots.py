# SPDX-License-Identifier: LGPL-2.1-or-later
"""v0.9 tests, written before the implementation.

(a) SLOTS: a stadium-shaped opening (2 parallel lines + 2 semicircular
    arcs of equal radius) with two half-cylinder end walls. Through and
    blind variants, with world-frame geometry round-trips. Slot ends must
    be claimed as slot surfaces, not mislabelled as fillets.

(b) RELATIVE-TOLERANCE SPEC KEYS: two 9.12812 vs 9.12811 holes are the
    same drill; fixed decimal rounding split them (observed on
    1002_tray_bottom.STL). Spec grouping must use relative tolerance --
    and still keep genuinely different sizes apart.
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import Feature, FeatureReport, detect_features
from meshtofeatures.history import plan_history
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

from .test_adversarial import assert_geometry_match

pytest.importorskip("manifold3d")

SECTIONS = 64


def _slot_tool(length, width, height):
    """Stadium prism: overall `length` x `width`, semicircular ends."""
    r = width / 2.0
    a = (length - width) / 2.0
    parts = [trimesh.creation.box(extents=[2 * a, width, height])]
    for sx in (-a, a):
        c = trimesh.creation.cylinder(radius=r, height=height, sections=SECTIONS)
        c.apply_translation([sx, 0, 0])
        parts.append(c)
    tool = parts[0]
    for p in parts[1:]:
        tool = tool.union(p)
    return tool


def through_slot_plate():
    plate = trimesh.creation.box(extents=[60.0, 40.0, 6.0])
    tool = _slot_tool(30.0, 8.0, 20.0)
    tool.apply_translation([5.0, -4.0, 0.0])
    return plate.difference(tool)


def blind_slot_plate():
    plate = trimesh.creation.box(extents=[60.0, 40.0, 10.0])  # z in [-5, 5]
    tool = _slot_tool(30.0, 8.0, 10.0)
    tool.apply_translation([5.0, -4.0, 6.0])                  # z in [1, 11]
    return plate.difference(tool)                              # depth 4


def _detect(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    return report, patches, feats


class TestThroughSlot:
    def test_feature(self):
        _, _, feats = _detect(through_slot_plate())
        slots = feats.by_kind("slot")
        assert len(slots) == 1
        p = slots[0].params
        assert np.isclose(p["width"], 8.0, atol=0.02)
        assert np.isclose(p["length"], 30.0, atol=0.05)
        assert p["through"] is True

    def test_ends_not_mislabelled_as_fillets(self):
        _, _, feats = _detect(through_slot_plate())
        assert feats.by_kind("fillet") == []

    def test_geometry_roundtrip(self):
        mesh = through_slot_plate()
        report, patches, feats = _detect(mesh)
        plan = plan_history(report, feats, detect_patterns(feats), patches)
        assert plan.pockets and plan.pockets[0].through
        assert_geometry_match(mesh, plan)


class TestBlindSlot:
    def test_feature(self):
        _, _, feats = _detect(blind_slot_plate())
        slots = feats.by_kind("slot")
        assert len(slots) == 1
        p = slots[0].params
        assert p["through"] is False
        assert np.isclose(p["depth"], 4.0, atol=0.02)

    def test_geometry_roundtrip(self):
        mesh = blind_slot_plate()
        report, patches, feats = _detect(mesh)
        plan = plan_history(report, feats, detect_patterns(feats), patches)
        assert_geometry_match(mesh, plan)


# ------------------------------------------------------- (b) spec keys

def _fake_hole(diameter, depth=5.0, through=True, pos=(0.0, 0.0, 0.0)):
    return Feature(kind="hole", surface_indices=[],
                   params={"diameter": diameter, "depth": depth,
                           "through": through, "position": list(pos),
                           "axis": [0.0, 0.0, 1.0]},
                   description=f"Hole d{diameter:g}")


class TestSpecTolerance:
    def test_near_identical_diameters_group(self):
        # the tray failure: 9.12812 vs 9.12811 split by decimal rounding
        feats = FeatureReport(features=[
            _fake_hole(9.12812, pos=(0, 0, 0)),
            _fake_hole(9.12811, pos=(20, 0, 0)),
            _fake_hole(9.12812, pos=(40, 0, 0)),
        ])
        pats = detect_patterns(feats)
        assert len(pats.patterns) == 1
        assert pats.patterns[0].params["pattern"] == "linear"

    def test_genuinely_different_diameters_stay_apart(self):
        feats = FeatureReport(features=[
            _fake_hole(8.0, pos=(0, 0, 0)),
            _fake_hole(8.0, pos=(20, 0, 0)),
            _fake_hole(8.0, pos=(40, 0, 0)),
            _fake_hole(6.0, pos=(0, 15, 0)),
            _fake_hole(6.0, pos=(20, 15, 0)),
            _fake_hole(6.0, pos=(40, 15, 0)),
        ])
        pats = detect_patterns(feats)
        assert len(pats.patterns) == 2

    def test_mixed_through_flags_stay_apart(self):
        feats = FeatureReport(features=[
            _fake_hole(8.0, through=True, pos=(0, 0, 0)),
            _fake_hole(8.0, through=True, pos=(20, 0, 0)),
            _fake_hole(8.0, through=False, depth=3.0, pos=(40, 0, 0)),
        ])
        pats = detect_patterns(feats)
        assert pats.patterns == []  # only 2 compatible members: below minimum
