# SPDX-License-Identifier: LGPL-2.1-or-later
"""Base-face through-holes: a frame/ring/bracket whose base face has a
central opening.

The base profile was built from the base face's OUTER loop only, so the
base prism filled any hole in that face -- over-filling frames by exactly
the hole's volume (field-observed on idler_riser.STL: rebuilt 2x too big,
the over-fill == central-opening area x base length). A hole that is a
genuine through-opening (present at BOTH z-ends, not a blind-pocket
mouth) must be carried on the base profile so the base is a ring.

A round or rectangular opening is already recovered as a hole/pocket
feature; an odd polygon (octagon) is not, which is exactly the case the
base profile must handle itself.
"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from .test_adversarial import _plan, assert_geometry_match, ROT   # noqa: E402


def frame_octagon():
    """40x40x10 plate with an octagonal through-opening (radius 12): the
    opening is not recovered as a hole/pocket feature, so the base
    profile must carry it or the ring fills solid."""
    base = trimesh.creation.box(extents=[40, 40, 10])
    base.apply_translation([0, 0, 5])
    hole = trimesh.creation.cylinder(radius=12, height=20, sections=8)
    hole.apply_translation([0, 0, 5])
    return base.difference(hole)


def blind_pocket_plate():
    """A blind pocket (NOT through): the base must stay solid -- its base
    face's pocket mouth must not be mistaken for a through-hole."""
    base = trimesh.creation.box(extents=[40, 40, 10])
    base.apply_translation([0, 0, 5])
    cut = trimesh.creation.box(extents=[16, 12, 4])
    cut.apply_translation([0, 0, 9])            # opens at top only (z[7,10])
    return base.difference(cut)


class TestBaseThroughHole:
    def test_base_carries_through_hole(self):
        mesh = frame_octagon()
        _, _, plan = _plan(mesh)
        assert getattr(plan.base, "hole_profiles", []), \
            "frame base must carry its through-opening"

    def test_frame_roundtrip(self):
        mesh = frame_octagon()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)

    def test_frame_roundtrip_rotated(self):
        mesh = frame_octagon()
        mesh.apply_transform(ROT)
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)

    def test_blind_pocket_base_stays_solid(self):
        mesh = blind_pocket_plate()
        _, _, plan = _plan(mesh)
        # the blind pocket mouth must NOT be carried as a base hole
        assert not getattr(plan.base, "hole_profiles", []), \
            "a blind pocket must not open the base"
        assert_geometry_match(mesh, plan)


def counterbored_plate():
    """A solid plate with two counterbored THROUGH-holes. The holes are
    recovered as hole/counterbore FEATURES, so they must NOT be carved
    into the base profile (else the base pre-punches them, the counterbore
    pockets cut already-open space, and the sinks are lost -- field-
    observed on featuretype: 8 counterbored holes pre-holed the base)."""
    base = trimesh.creation.box(extents=[40, 30, 10])
    base.apply_translation([0, 0, 5])
    for cx in (-10, 10):
        d = trimesh.creation.cylinder(radius=2.0, height=20, sections=48)
        d.apply_translation([cx, 0, 5])
        cb = trimesh.creation.cylinder(radius=4.0, height=3, sections=48)
        cb.apply_translation([cx, 0, 8.5])
        base = base.difference(d).difference(cb)
    return base


class TestDrilledHolesNotInBase:
    def test_counterbored_holes_stay_features(self):
        mesh = counterbored_plate()
        _, _, plan = _plan(mesh)
        assert not plan.base.hole_profiles, \
            "drilled counterbored holes must be features, not base openings"

    def test_counterbored_plate_roundtrip(self):
        mesh = counterbored_plate()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)
