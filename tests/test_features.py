# SPDX-License-Identifier: LGPL-2.1-or-later
"""Feature-detection tests (v0.6), written before the implementation.

The composite fixtures each embody one canonical feature pattern:

* drilled plate     -> one THROUGH HOLE, d=8, depth 5, at (10, 5)
* counterbored hole -> one COUNTERBORED HOLE, d=8 through, cb d=12 x 2
* stepped shaft     -> one BOSS, d=12, h=15 (the small step standing on
                       the shoulder of the big shaft)
* rounded slab      -> four FILLETS, r=3, ~90 deg
* pocketed plate    -> one POCKET, 20 x 12 opening, depth 4

Rules must also *not* hallucinate: a sphere has no features, and every
feature must reference the surfaces it explains (provenance).
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

from .test_composites import (counterbored_plate, drilled_plate,
                              pocketed_plate, stepped_shaft)
from .test_robustness import rounded_slab

pytest.importorskip("manifold3d")


def _detect(mesh):
    report = snap_report(reconstruct(mesh)).report
    return report, detect_features(report, plan_patches(report))


class TestThroughHole:
    def test_single_through_hole(self):
        _, feats = _detect(drilled_plate())
        holes = feats.by_kind("hole")
        assert len(holes) == 1
        h = holes[0]
        assert h.params["diameter"] == 8.0
        assert h.params["through"] is True
        assert np.isclose(h.params["depth"], 5.0)

    def test_hole_position(self):
        _, feats = _detect(drilled_plate())
        pos = feats.by_kind("hole")[0].params["position"]
        assert np.allclose(np.asarray(pos)[:2], [10.0, 5.0])

    def test_provenance(self):
        report, feats = _detect(drilled_plate())  # noqa: F841
        h = feats.by_kind("hole")[0]
        kinds = {report.surfaces[i].kind for i in h.surface_indices}
        assert "cylinder" in kinds
        assert all(0 <= i < len(report.surfaces) for i in h.surface_indices)


class TestCounterbore:
    def test_detected_as_one_feature(self):
        _, feats = _detect(counterbored_plate())
        cbs = feats.by_kind("counterbore")
        assert len(cbs) == 1
        # the two cylinders must not ALSO be reported as plain holes
        assert feats.by_kind("hole") == []

    def test_parameters(self):
        _, feats = _detect(counterbored_plate())
        p = feats.by_kind("counterbore")[0].params
        assert p["diameter"] == 8.0
        assert p["counterbore_diameter"] == 12.0
        assert np.isclose(p["counterbore_depth"], 2.0)
        assert p["through"] is True


class TestBoss:
    def test_step_is_a_boss(self):
        _, feats = _detect(stepped_shaft())
        bosses = feats.by_kind("boss")
        assert len(bosses) == 1
        p = bosses[0].params
        assert p["diameter"] == 12.0
        assert np.isclose(p["height"], 15.0)

    def test_shaft_body_is_not_a_hole(self):
        _, feats = _detect(stepped_shaft())
        assert feats.by_kind("hole") == []


class TestFillet:
    def test_four_fillets(self):
        _, feats = _detect(rounded_slab())
        fillets = feats.by_kind("fillet")
        assert len(fillets) == 4
        for f in fillets:
            assert f.params["radius"] == 3.0
            assert np.isclose(f.params["arc_degrees"], 90.0, atol=10.0)


class TestPocket:
    def test_pocket_detected(self):
        _, feats = _detect(pocketed_plate())
        pockets = feats.by_kind("pocket")
        assert len(pockets) == 1
        p = pockets[0].params
        assert np.isclose(p["depth"], 4.0)
        assert sorted(np.round(p["opening_size"], 6)) == [12.0, 20.0]


class TestHonesty:
    def test_sphere_has_no_features(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
        _, feats = _detect(mesh)
        assert feats.features == []

    def test_defining_cylinders_assigned_at_most_once(self):
        # Ownership doctrine (amended after the hollow-boss bug): a
        # feature's defining CYLINDER is exclusive, but planes are shared
        # interfaces -- a top annulus is both a through-hole's opening
        # and a boss's cap.
        from meshtofeatures.primitives import Cylinder
        for mesh in (drilled_plate(), counterbored_plate(), stepped_shaft()):
            report, feats = _detect(mesh)
            cyl_used = [i for f in feats.features for i in f.surface_indices
                        if isinstance(report.surfaces[i].fit.primitive, Cylinder)]
            assert len(cyl_used) == len(set(cyl_used))

    def test_descriptions_are_human_readable(self):
        _, feats = _detect(counterbored_plate())
        d = feats.features[0].description
        assert "8" in d and "12" in d
