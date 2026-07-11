# SPDX-License-Identifier: LGPL-2.1-or-later
"""Frame-axis canonicalization.

`means[best]` inherits its sign from the tessellation's face-normal
winding, so the SAME part can reconstruct with frame_z = +axis from one
mesher and -axis from another (field-observed: Python/trimesh gave +z,
FreeCAD's mesh gave -z). The plan stays self-consistent either way, but
the FreeCAD executor's z_at/flip placement assumes a canonical
orientation and mis-places every from-bottom feature under the inverted
sign. `plan_history` points frame_z along the positive direction of its
dominant axis so the executor is never fed an inverted frame.
"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

import meshtofeatures.snapping as snapping
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.history import plan_history
from .test_adversarial import (_plan, _rebuild_mesh, _to_world,
                               assert_geometry_match)


def stepped_block():
    base = trimesh.creation.box(extents=[40, 30, 12])
    base.apply_translation([0, 0, 6])
    deck = trimesh.creation.box(extents=[40, 14, 6])
    deck.apply_translation([0, -8, 15])
    part = base.union(deck)
    drill = trimesh.creation.cylinder(radius=3, height=40, sections=48)
    drill.apply_translation([10, 5, 6])
    return part.difference(drill)


def _dominant_positive(z):
    return float(z[int(np.argmax(np.abs(z)))]) >= 0.0


class TestFrameCanonicalization:
    @pytest.mark.parametrize("factory", [stepped_block])
    def test_frame_is_canonical(self, factory):
        _, _, plan = _plan(factory())
        assert _dominant_positive(plan.frame_z)

    def test_inverted_mean_is_canonicalized(self, monkeypatch):
        """Force the dominant plane's mean normal to point the 'wrong'
        way (as a flipped-winding mesher would); the plan's frame_z must
        still come out canonical AND rebuild correctly."""
        mesh = stepped_block()
        report = snapping.snap_report(reconstruct(mesh)).report
        patches = plan_patches(report)
        feats = detect_features(report, patches)
        pats = detect_patterns(feats)

        real = snapping.cluster_directions

        def flipped(dirs, areas, tol):
            labels, means = real(dirs, areas, tol)
            return labels, -means            # invert every cluster mean

        monkeypatch.setattr(snapping, "cluster_directions", flipped)
        plan = plan_history(report, feats, pats, patches)

        # despite the inverted mean, the frame must be canonical ...
        assert _dominant_positive(plan.frame_z), \
            "inverted mean was not canonicalized"
        # ... and the geometry must still be correct
        reb = _to_world(plan, _rebuild_mesh(plan))
        assert len(reb.split(only_watertight=False)) == 1
        assert abs(reb.volume - mesh.volume) / mesh.volume < 0.05
