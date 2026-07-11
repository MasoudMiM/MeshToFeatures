# SPDX-License-Identifier: LGPL-2.1-or-later
"""Pattern-detection tests (v0.7), written before the implementation.

Designers create hole PATTERNS, not individual holes; recovering the
pattern is recovering intent. Fixtures with exact ground truth:

* bolt circle -- 6 x d8 on BCD 60 (+ a d10 centre hole that must stay
  separate: different spec)
* linear row  -- 5 x d8, pitch 15
* grid        -- 3 x 4 of d6, pitches 12 x 10
* negatives   -- a lone hole patterns with nothing; unequal spacing and
  mixed specs must not be forced into patterns
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

pytest.importorskip("manifold3d")

SECTIONS = 48


def _plate_with_holes(centers, radii, extents=(100.0, 100.0, 5.0)):
    plate = trimesh.creation.box(extents=list(extents))
    for (x, y), r in zip(centers, radii):
        d = trimesh.creation.cylinder(radius=r, height=20.0, sections=SECTIONS)
        d.apply_translation([x, y, 0.0])
        plate = plate.difference(d)
    return plate


def _detect(mesh):
    report = snap_report(reconstruct(mesh)).report
    feats = detect_features(report, plan_patches(report))
    return feats, detect_patterns(feats)


def bolt_circle_plate():
    ang = np.deg2rad(np.arange(6) * 60.0)
    centers = [(30 * np.cos(a), 30 * np.sin(a)) for a in ang]
    return _plate_with_holes(centers + [(0.0, 0.0)], [4.0] * 6 + [5.0])


class TestCircularPattern:
    def test_bolt_circle_detected(self):
        feats, pats = _detect(bolt_circle_plate())
        circ = [p for p in pats.patterns if p.params["pattern"] == "circular"]
        assert len(circ) == 1
        p = circ[0].params
        assert p["count"] == 6
        assert np.isclose(p["bolt_circle_diameter"], 60.0, atol=0.05)
        assert np.allclose(p["center"][:2], [0.0, 0.0], atol=0.05)

    def test_centre_hole_stays_single(self):
        feats, pats = _detect(bolt_circle_plate())
        # the d10 centre hole has a different spec: ungrouped
        assert len(pats.ungrouped) == 1
        assert pats.ungrouped[0].params["diameter"] == 10.0

    def test_pattern_provenance_is_union_of_members(self):
        # planes are shared interfaces (ownership doctrine): the union is
        # deduplicated, and only defining cylinders must be unique
        from meshtofeatures.primitives import Cylinder
        report = snap_report(reconstruct(bolt_circle_plate())).report
        feats = detect_features(report, plan_patches(report))
        pats = detect_patterns(feats)
        p = [q for q in pats.patterns if q.params["pattern"] == "circular"][0]
        member_surfaces = {i for m in p.members for i in m.surface_indices}
        assert set(p.surface_indices) == member_surfaces
        assert len(p.surface_indices) == len(set(p.surface_indices))
        cyls = [i for m in p.members for i in m.surface_indices
                if isinstance(report.surfaces[i].fit.primitive, Cylinder)]
        assert len(cyls) == len(set(cyls))


class TestLinearPattern:
    def test_row_detected(self):
        centers = [(-30 + 15 * i, 0.0) for i in range(5)]
        feats, pats = _detect(_plate_with_holes(centers, [4.0] * 5))
        lin = [p for p in pats.patterns if p.params["pattern"] == "linear"]
        assert len(lin) == 1
        p = lin[0].params
        assert p["count"] == 5
        assert np.isclose(p["pitch"], 15.0, atol=0.02)

    def test_unequal_spacing_rejected(self):
        centers = [(-30, 0.0), (-10, 0.0), (14, 0.0)]  # 20 vs 24 spacing
        feats, pats = _detect(_plate_with_holes(centers, [4.0] * 3))
        assert pats.patterns == []
        assert len(pats.ungrouped) == 3


class TestGridPattern:
    def test_grid_detected(self):
        centers = [(-12 + 12 * i, -10 + 10 * j)
                   for i in range(3) for j in range(4)]
        feats, pats = _detect(_plate_with_holes(centers, [3.0] * 12))
        grids = [p for p in pats.patterns if p.params["pattern"] == "grid"]
        assert len(grids) == 1
        p = grids[0].params
        assert p["count"] == 12
        assert sorted(np.round(p["shape"])) == [3, 4]
        assert sorted(np.round(p["pitches"], 6)) == [10.0, 12.0]


class TestSpecSeparation:
    def test_two_specs_two_patterns(self):
        row1 = [(-20 + 20 * i, 15.0) for i in range(3)]
        row2 = [(-20 + 20 * i, -15.0) for i in range(3)]
        mesh = _plate_with_holes(row1 + row2, [4.0] * 3 + [3.0] * 3)
        feats, pats = _detect(mesh)
        lin = [p for p in pats.patterns if p.params["pattern"] == "linear"]
        assert len(lin) == 2
        assert sorted(p.params["member"]["diameter"] for p in lin) == [6.0, 8.0]


class TestHonesty:
    def test_single_hole_no_pattern(self):
        feats, pats = _detect(_plate_with_holes([(10.0, 5.0)], [4.0],
                                                extents=(40, 30, 5)))
        assert pats.patterns == []
        assert len(pats.ungrouped) == 1

    def test_descriptions(self):
        feats, pats = _detect(bolt_circle_plate())
        d = [q for q in pats.patterns][0].description
        assert "6" in d and "8" in d
