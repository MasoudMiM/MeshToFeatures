# SPDX-License-Identifier: LGPL-2.1-or-later
"""Blend-channel peeling (`split_by_channels`).

Tangent blends chain a fillet strip to its flat neighbours, and curvature
clustering chains adjacent strips of EQUAL radius into one blob (their
curvature proxy is identical), so neither the dihedral split nor the
curvature split separates them. The invariant that does is the spine
DIRECTION: a straight-spine channel (a constant-cross-section blend band)
has every face normal perpendicular to one common direction, and its
tessellation carries ruling edges parallel to it. These tests lock the
peel, the downstream fillet recovery, and the two false-positive classes
the validator must refuse (a cone is locally a channel; a planar patch has
normals on infinitely many great circles).
"""

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from freecad.meshtofeatures_wb.core.emission import plan_patches
from freecad.meshtofeatures_wb.core.features import detect_features
from freecad.meshtofeatures_wb.core.pipeline import reconstruct
from freecad.meshtofeatures_wb.core.segmentation import split_by_channels
from freecad.meshtofeatures_wb.core.snapping import snap_report


def filleted_prism(r=5.0, h=12.0, n=8):
    """A 40x30 bar extruded to height h with one vertical corner rounded
    by a radius-r fillet: the rounded strip is a straight-spine channel.
    The arc uses STL-grade tessellation (coarse enough that the first
    blend facet lifts off the tangent plane by more than any coplanarity
    tolerance, as real exports do)."""
    pts = [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0 - r)]
    cx, cy = 40.0 - r, 30.0 - r
    for a in np.linspace(0.0, np.pi / 2, n):
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    pts.append((0.0, 30.0))
    return trimesh.creation.extrude_polygon(Polygon(pts), height=h)


class TestSplitByChannels:
    def test_peels_the_fillet_strip_from_curved_junk(self):
        # Post-plane-peel remainder analogue: the fillet strip chained with
        # genuinely curved non-channel junk (a sphere patch, whose rulings
        # converge and whose cross-section is no circle). The peel must
        # take the strip and leave the junk.
        mesh = filleted_prism()
        strip_faces = np.flatnonzero(
            np.abs(mesh.face_normals[:, 2]) < 0.99)      # side + strip
        # drop the flat sides (2-face slivers): keep faces whose normal is
        # not axis-aligned -- the arc facets and the junk we add below
        arc_faces = np.flatnonzero(
            (np.abs(mesh.face_normals[:, 2]) < 0.99)
            & (np.abs(mesh.face_normals[:, 0]) > 0.05)
            & (np.abs(mesh.face_normals[:, 1]) > 0.05))
        assert len(arc_faces) >= 8
        # junk: a sphere cap somewhere else (merged in via a shared edge is
        # not required -- the peel works on the face set as given)
        sph = trimesh.creation.icosphere(subdivisions=2, radius=7.0)
        sph.apply_translation([80.0, 80.0, 6.0])
        cap = np.flatnonzero(sph.face_normals[:, 2] > 0.3)
        combo = trimesh.util.concatenate(
            mesh.submesh([arc_faces], append=True),
            sph.submesh([cap], append=True))
        faces = np.arange(len(combo.faces))
        channels, remainder = split_by_channels(combo, faces, rms_gate=0.05)
        assert len(channels) == 1
        strip = channels[0]
        assert strip.area == pytest.approx(5.0 * (np.pi / 2) * 12.0,
                                           rel=0.15)
        assert len(remainder) == len(cap)

    def test_refuses_a_cone_wall(self):
        # A narrow cone sector is locally channel-like (zero curvature
        # along its generators) but its rulings CONVERGE at the apex; the
        # peel must refuse it and leave it to the cone fit.
        import importlib
        mod = importlib.import_module(
            "freecad.meshtofeatures_wb.tests.test_cone_pocket")
        mesh = mod.conical_pocket_plate(r_mouth=6.0, r_far=2.0, depth=4.0,
                                        t=10.0)
        faces = np.arange(len(mesh.faces))
        channels, remainder = split_by_channels(mesh, faces, rms_gate=0.05)
        assert channels == []
        assert len(remainder) == len(faces)

    def test_refuses_planar_faces(self):
        mesh = trimesh.creation.box(extents=[30, 20, 10])
        faces = np.arange(len(mesh.faces))
        channels, remainder = split_by_channels(mesh, faces, rms_gate=0.05)
        assert channels == []
        assert len(remainder) == len(faces)


class TestFilletRecoveryEndToEnd:
    def test_fillet_strip_recognized_and_planned(self):
        mesh = filleted_prism()
        report = snap_report(reconstruct(mesh)).report
        assert report.coverage > 0.99
        from freecad.meshtofeatures_wb.core.primitives import Cylinder
        cyls = [s for s in report.surfaces
                if isinstance(s.fit.primitive, Cylinder)]
        assert any(abs(s.fit.primitive.radius - 5.0) < 0.05 for s in cyls), \
            "the blend strip was not recovered as a cylinder"
        feats = detect_features(report, plan_patches(report))
        fillets = feats.by_kind("fillet")
        assert len(fillets) == 1
        assert fillets[0].params["radius"] == pytest.approx(5.0, abs=0.05)
