# SPDX-License-Identifier: LGPL-2.1-or-later
"""Refine-split threshold cap.

A large flat face can transition smoothly (chamfer/fillet) into a faceted
or curved perimeter, so region-growing at the feature-edge threshold
fuses them into one segment that fits no primitive and is dropped -- the
flat plane is lost (field-observed on counter.unitsmm.STL: 49% coverage,
one 4074-face blob holding a 66,000-area flat top; the fix lifts it to
100%). The refinement split (`split_by_curvature`) must run at a
threshold TIGHTER than the feature-edge threshold, or a tangent-merged
blob whose internal edges are all below the (possibly 60 deg) feature
threshold never breaks. `reconstruct` caps the refine-split threshold at
`refine_split_max` (default 30 deg).

counter's behaviour is emergent from the whole mesh (its global adaptive
threshold), so a small synthetic mesh cannot reproduce it; these tests
lock the cap MECHANISM (the threshold actually handed to the splitter)
and the flat-recovery capability. counter itself is covered by the
corpus stress run.
"""

import inspect

import numpy as np
import trimesh

import meshtofeatures.pipeline as pipeline
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.segmentation import adaptive_angle_threshold


def chamfered_prism(sides=12, r=50.0, h=20.0, cham=6.0):
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)

    def ring(rad, z):
        return np.c_[rad * np.cos(ang), rad * np.sin(ang), np.full(sides, z)]

    V = np.vstack([ring(r, 0.0), ring(r, h - cham), ring(r - cham, h),
                   [[0, 0, 0]], [[0, 0, h]]])
    bc, tc = 3 * sides, 3 * sides + 1
    F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [[bc, j, i], [i, j, sides + j], [i, sides + j, sides + i],
              [sides + i, sides + j, 2 * sides + j],
              [sides + i, 2 * sides + j, 2 * sides + i],
              [2 * sides + i, 2 * sides + j, tc]]
    m = trimesh.Trimesh(vertices=V, faces=np.array(F), process=True)
    m.fix_normals()
    return m


class TestRefineSplitCap:
    def test_parameter_exists(self):
        assert "refine_split_max" in inspect.signature(reconstruct).parameters

    def test_split_runs_below_feature_threshold(self, monkeypatch):
        mesh = chamfered_prism(sides=12)          # reaches refinement
        assert np.rad2deg(adaptive_angle_threshold(mesh)) > 40  # loose

        seen = []
        real = pipeline.split_by_curvature

        def spy(m_, f_, a_, **kw):
            seen.append(float(a_))
            return real(m_, f_, a_, **kw)

        monkeypatch.setattr(pipeline, "split_by_curvature", spy)
        reconstruct(mesh)
        assert seen, "refinement never ran"
        assert max(seen) <= np.deg2rad(30.0) + 1e-9, \
            "refine split was not capped below the feature threshold"

    def test_cap_never_loosens_a_fine_mesh(self, monkeypatch):
        mesh = chamfered_prism(sides=64)
        seen = []
        real = pipeline.split_by_curvature
        monkeypatch.setattr(
            pipeline, "split_by_curvature",
            lambda m_, f_, a_, **k: (seen.append(float(a_)),
                                     real(m_, f_, a_, **k))[1])
        reconstruct(mesh)
        thr = adaptive_angle_threshold(mesh)
        if seen:
            assert max(seen) <= min(thr, np.deg2rad(30.0)) + 1e-9


class TestFlatRecoveryCapability:
    def test_chamfered_part_recovers_both_flats(self):
        mesh = chamfered_prism(sides=36)
        report = reconstruct(mesh)
        big_flats = [s for s in report.surfaces
                     if s.fit.primitive.kind == "plane"
                     and abs(float(s.fit.primitive.normal[2])) > 0.99
                     and s.segment.area > 0.15 * mesh.area]
        assert len(big_flats) >= 2
        assert report.coverage > 0.9
