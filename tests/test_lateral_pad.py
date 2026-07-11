# SPDX-License-Identifier: LGPL-2.1-or-later
"""Lateral pads: material protruding horizontally beyond the base footprint.

Every pad the planner emitted before v0.15.6 was a VERTICAL extrusion --
a boss cap or a whole prism grown along z. A mounting flange, side rail,
or gusseted bracket protrudes *sideways* off a wall in a limited z-band,
and had no representation at all: the base footprint comes from one
horizontal plane's outline, so anything sticking out past that outline
was silently dropped (7-8% of volume on the field's featuretype.STL, an
exact-`depth` p99 plateau on the mesh->rebuilt surface distance).

These fixtures pin the two shapes that matter:

* a rectangular flange (flat underside) -- the common mounting lug,
* a gusseted bracket (45-degree ramped underside reaching the base) --
  the featuretype.STL shape.

Both are extruded along the wall they hang off; the pad profile lives in
the plane perpendicular to that axis. Verification is the adversarial
two-way world-frame surface match, including an arbitrarily rotated
copy (a lateral pad on a rotated part exercises the general basis).
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.history import PadOp

pytest.importorskip("manifold3d")

from .test_adversarial import (_plan, assert_geometry_match, ROT,   # noqa: E402
                               _rebuild_mesh, _to_world)


# ----------------------------------------------------------------- fixtures

def flanged_plate():
    """Base 40x30x10 (z in [0,10]) with a rectangular flange hanging off
    the -y wall: full width, y in [-20,-15], z in [3,8]."""
    base = trimesh.creation.box(extents=[40, 30, 10])
    base.apply_translation([0, 0, 5])                     # z in [0,10]
    flange = trimesh.creation.box(extents=[40, 5, 5])
    flange.apply_translation([0, -17.5, 5.5])             # y[-20,-15] z[3,8]
    return base.union(flange)


def gusseted_bracket():
    """Base 40x30x10 (z in [0,10]) with a gusseted bracket off the -y
    wall: trapezoid cross-section A(-15,0) B(-18,3) C(-18,8) D(-15,8),
    extruded full width -- the featuretype.STL flange shape (45-deg ramp
    underside reaching the base bottom)."""
    base = trimesh.creation.box(extents=[40, 30, 10])
    base.apply_translation([0, 0, 5])
    prof = [(-15.0, 0.0), (-18.0, 3.0), (-18.0, 8.0), (-15.0, 8.0)]
    pts = []
    for x in (-20.0, 20.0):
        for (y, z) in prof:
            pts.append((x, y, z))
    bracket = trimesh.Trimesh(vertices=np.array(pts)).convex_hull
    return base.union(bracket)


def two_lug_plate():
    """Base 40x30x10 (z in [0,10]) with TWO separate rectangular lugs on
    the -y wall: x[-18,-8] and x[8,18], both y[-20,-15] z[3,8], with an
    empty gap x[-8,8] between them. Must become TWO lateral pads, not one
    bridging the gap."""
    base = trimesh.creation.box(extents=[40, 30, 10])
    base.apply_translation([0, 0, 5])
    a = trimesh.creation.box(extents=[10, 5, 5])
    a.apply_translation([-13, -17.5, 5.5])                # x[-18,-8]
    b = trimesh.creation.box(extents=[10, 5, 5])
    b.apply_translation([13, -17.5, 5.5])                 # x[8,18]
    return base.union(a).union(b)


def horizontal_boss():
    """Base 40x30x10 (z in [0,10]) with a horizontal cylindrical boss (a
    peg) protruding off the -y wall: radius 3, axis along -y, y[-21,-15],
    centered x=0 z=5. Its outer wall is CURVED, so the planar-wall anchor
    misses it (design note 36 handles it via the cylinder anchor)."""
    base = trimesh.creation.box(extents=[40, 30, 10])
    base.apply_translation([0, 0, 5])
    cyl = trimesh.creation.cylinder(radius=3, height=6, sections=64)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]))                            # axis -> y
    cyl.apply_translation([0, -18, 5])                    # y[-21,-15]
    return base.union(cyl)


def rounded_rail():
    """Base 40x30x10 with a half-round rail along the -y wall: a cylinder
    radius 3, axis along x (parallel to the wall), centered on the wall
    y=-15 at z=5, so the outer half protrudes. Curved outer wall whose
    axis is parallel to the wall (vs the boss's perpendicular axis)."""
    base = trimesh.creation.box(extents=[40, 30, 10])
    base.apply_translation([0, 0, 5])
    cyl = trimesh.creation.cylinder(radius=3, height=40, sections=64)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [0, 1, 0]))                            # axis -> x
    cyl.apply_translation([0, -15, 5])                    # centered on wall
    return base.union(cyl)


# -------------------------------------------------------------------- tests

class TestRectangularFlange:
    def test_a_lateral_pad_is_emitted(self):
        mesh = flanged_plate()
        _, _, plan = _plan(mesh)
        lateral = [p for p in plan.pads if getattr(p, "axis", None) is not None]
        assert len(lateral) == 1, f"expected one lateral pad, got {plan.pads}"
        assert plan.unplanned == []

    def test_geometry_roundtrip(self):
        mesh = flanged_plate()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)

    def test_geometry_roundtrip_rotated(self):
        mesh = flanged_plate()
        mesh.apply_transform(ROT)
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)


class TestGussetedBracket:
    def test_a_lateral_pad_is_emitted(self):
        mesh = gusseted_bracket()
        _, _, plan = _plan(mesh)
        lateral = [p for p in plan.pads if getattr(p, "axis", None) is not None]
        assert len(lateral) == 1

    def test_geometry_roundtrip(self):
        mesh = gusseted_bracket()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)

    def test_ramp_slope_survives_burying(self):
        """The bury step (extending the pad's inner edge into the base so
        OCC fuses it) must not PIVOT a slanted underside: sliding the
        ramp's inner endpoint along u changes the ramp's slope and lifts
        the pad's underside off the true surface (field-observed on
        featuretype.STL: the bottom bevel started 0.07 early at slope 0.88
        instead of 1.0 -- a visible step at the bottom). The true ramp runs
        from (y=-15, z=0) to (y=-18, z=3); both endpoints must survive as
        profile vertices, and the rebuilt underside must sit on the ramp."""
        from meshtofeatures.history import lateral_pad_world_frame
        mesh = gusseted_bracket()
        _, _, plan = _plan(mesh)
        pad = [p for p in plan.pads if getattr(p, "axis", None) is not None][0]
        origin, u, v, a = (np.asarray(q, float)
                           for q in lateral_pad_world_frame(plan, pad))
        world = [origin + float(pr.start[0]) * u + float(pr.start[1]) * v
                 for pr in pad.profile]
        diag = float(np.linalg.norm(mesh.extents))
        tol = 1e-3 * diag

        def has_vertex(y, z):
            return any(abs(w[1] - y) <= 2 * tol and abs(w[2] - z) <= 2 * tol
                       for w in world)

        assert has_vertex(-15.0, 0.0), \
            f"ramp inner endpoint (-15, 0) missing from profile: " \
            f"{[(round(float(w[1]), 3), round(float(w[2]), 3)) for w in world]}"
        assert has_vertex(-18.0, 3.0), "ramp outer endpoint (-18, 3) missing"
        # the rebuilt solid's underside at the ramp midpoint must be ON the
        # ramp (z = -(y + 15)), not lifted by a pivoted profile edge
        reb = _to_world(plan, _rebuild_mesh(plan))
        loc, _, _ = reb.ray.intersects_location(
            np.array([[0.0, -16.5, -5.0]]), np.array([[0.0, 0.0, 1.0]]),
            multiple_hits=False)
        assert len(loc), "no underside hit at ramp midpoint"
        assert abs(float(loc[0][2]) - 1.5) <= 3 * tol, \
            f"underside at y=-16.5 is z={loc[0][2]:.3f}, expected 1.5 (ramp)"


class TestMultipleFlangesOneSide:
    """Two separate protrusions sharing an outward direction must become
    two pads, not one convex hull bridging the empty gap between them."""

    def test_two_lugs_two_pads(self):
        mesh = two_lug_plate()
        _, _, plan = _plan(mesh)
        lateral = [p for p in plan.pads if getattr(p, "axis", None) is not None]
        assert len(lateral) == 2, f"expected two lateral pads, got {len(lateral)}"
        assert plan.unplanned == []

    def test_geometry_roundtrip(self):
        mesh = two_lug_plate()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)

    def test_geometry_roundtrip_rotated(self):
        mesh = two_lug_plate()
        mesh.apply_transform(ROT)
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)


class TestCurvedProtrusions:
    """Curved outer walls: a horizontal cylindrical boss (axis perp to the
    wall) and a rounded rail (axis parallel to the wall) both become a
    lateral pad with a circular profile extruded along the cylinder axis."""

    def _circle_pad(self, plan):
        from meshtofeatures.history import SketchCircle
        lat = [p for p in plan.pads if getattr(p, "axis", None) is not None]
        assert len(lat) == 1, f"expected one cylinder pad, got {len(lat)}"
        assert all(isinstance(pr, SketchCircle) for pr in lat[0].profile), \
            "cylinder pad profile must be a circle"
        return lat[0]

    def test_boss_circular_pad(self):
        mesh = horizontal_boss()
        _, _, plan = _plan(mesh)
        self._circle_pad(plan)
        assert plan.unplanned == []

    def test_boss_roundtrip(self):
        mesh = horizontal_boss()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)

    def test_boss_roundtrip_rotated(self):
        mesh = horizontal_boss()
        mesh.apply_transform(ROT)
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)

    def test_rail_circular_pad(self):
        mesh = rounded_rail()
        _, _, plan = _plan(mesh)
        self._circle_pad(plan)
        assert plan.unplanned == []

    def test_rail_roundtrip(self):
        mesh = rounded_rail()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)


class TestCylinderConvexity:
    """A drilled HOLE (concave) must never be emitted as a protruding
    boss, even when an incomplete base outline makes it look like it
    reaches past the footprint (field-observed on angle_block.STL)."""

    def test_convexity_sign(self):
        from meshtofeatures.history import _cylinder_convexity
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts = np.column_stack([np.zeros_like(theta),
                               3.0 * np.cos(theta), 3.0 * np.sin(theta)])
        radial = np.column_stack([np.zeros_like(theta),
                                  np.cos(theta), np.sin(theta)])
        axis = np.array([1.0, 0.0, 0.0])
        origin = np.zeros(3)
        assert _cylinder_convexity(pts, radial, axis, origin) > 0.9   # boss
        assert _cylinder_convexity(pts, -radial, axis, origin) < -0.9  # hole

    def test_horizontal_hole_is_not_a_boss(self):
        a = trimesh.creation.box(extents=[40, 10, 10])
        a.apply_translation([0, -10, 5])
        b = trimesh.creation.box(extents=[10, 30, 10])
        b.apply_translation([-15, 0, 5])
        drill = trimesh.creation.cylinder(radius=3, height=14, sections=48)
        drill.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
        drill.apply_translation([-15, 0, 5])
        mesh = a.union(b).difference(drill)
        _, _, plan = _plan(mesh)
        circ = [p for p in plan.pads
                if getattr(p, "axis", None) is not None
                and type(p.profile[0]).__name__ == "SketchCircle"]
        assert circ == [], "a horizontal hole was misread as a boss"


class TestLateralPadWorldFrame:
    """The FreeCAD executor builds in world coordinates from the pad's
    PLAN-frame basis via lateral_pad_world_frame. It must compose to
    exactly the plan-frame construction followed by the frame placement
    (the transform the verified container rebuild uses), so the FreeCAD
    body and the round-trip agree."""

    def _frame_M(self, plan):
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = plan.frame_x, plan.frame_y, plan.frame_z
        M[:3, 3] = plan.frame_origin
        return M

    def test_composition_identity(self):
        from meshtofeatures.history import lateral_pad_world_frame
        for fixture in (flanged_plate, gusseted_bracket):
            mesh = fixture()
            mesh.apply_transform(ROT)                 # general (rotated) frame
            _, _, plan = _plan(mesh)
            pad = [p for p in plan.pads
                   if getattr(p, "axis", None) is not None][0]
            origin, u, v, axis = lateral_pad_world_frame(plan, pad)
            # world basis is orthonormal
            for w in (u, v, axis):
                assert np.isclose(np.linalg.norm(w), 1.0)
            assert abs(float(u @ v)) < 1e-9
            assert abs(float(u @ axis)) < 1e-9
            assert abs(float(v @ axis)) < 1e-9
            # a profile point maps identically both ways
            M = self._frame_M(plan)
            for (uc, vc) in [(0.0, 0.0), (1.3, -2.1), (5.0, 4.0)]:
                plan_pt = pad.plane_origin + uc * pad.plane_u + vc * pad.plane_v
                world_via_frame = (M[:3, :3] @ plan_pt) + M[:3, 3]
                world_direct = origin + uc * u + vc * v
                assert np.allclose(world_via_frame, world_direct, atol=1e-9)
            # extrusion direction agrees with the frame-mapped plan axis
            assert np.allclose(axis, M[:3, :3] @ pad.axis, atol=1e-9)


class TestLateralPadFusesToBase:
    """Regression for the field-observed floating flange: the manifold
    round-trip fuses any overlap, but OCC's boolean leaves the pad a
    separate solid when the overlap is a thin sliver. The pad must bury a
    ROBUST fraction of itself in the base so the FreeCAD fuse connects."""

    @pytest.mark.parametrize("factory", [flanged_plate, gusseted_bracket])
    def test_pad_overlaps_base_robustly(self, factory):
        from shapely.geometry import Polygon
        mesh = factory()
        _, _, plan = _plan(mesh)
        pad = next(p for p in plan.pads
                   if getattr(p, "axis", None) is not None)

        def ring(prims):
            pts = []
            for pr in prims:
                pts.extend(np.array(pr.sample())[:, :2])
            return pts

        base = trimesh.creation.extrude_polygon(
            Polygon(ring(plan.base.profile)), height=plan.base.length)
        padm = trimesh.creation.extrude_polygon(
            Polygon([(float(a[0]), float(a[1])) for a in ring(pad.profile)]),
            height=pad.length)
        T = np.eye(4)
        T[:3, 0], T[:3, 1] = pad.plane_u, pad.plane_v
        T[:3, 2], T[:3, 3] = pad.axis, pad.plane_origin
        padm.apply_transform(T)
        inter = base.intersection(padm)
        vol = inter.volume if inter is not None and hasattr(inter, "volume") else 0.0
        # a thin sliver overlap (the old 0.05*depth) is a few % of the pad;
        # require a solid fraction so OCC fuses reliably
        assert vol > 0.1 * padm.volume, \
            f"pad overlaps base by only {vol / padm.volume:.1%} (thin sliver)"

    @pytest.mark.parametrize("factory", [flanged_plate, gusseted_bracket,
                                         two_lug_plate, horizontal_boss,
                                         rounded_rail])
    def test_rebuild_is_single_body(self, factory):
        # a lateral pad must fuse into ONE connected solid, not float off
        _, _, plan = _plan(factory())
        reb = _to_world(plan, _rebuild_mesh(plan))
        assert len(reb.split(only_watertight=False)) == 1


class TestLateralPadRightHanded:
    """The pad basis (plane_u, plane_v, axis) must be RIGHT-handed:
    FreeCAD's App.Placement cannot represent a reflection, so a
    left-handed frame silently mis-places the pad and the flange floats
    off the body (field-observed on featuretype in FreeCAD; the manifold
    round-trip applies the full matrix and never saw it)."""

    @pytest.mark.parametrize("factory", [flanged_plate, gusseted_bracket,
                                         two_lug_plate, horizontal_boss,
                                         rounded_rail])
    def test_basis_is_right_handed(self, factory):
        _, _, plan = _plan(factory())
        pads = [p for p in plan.pads if getattr(p, "axis", None) is not None]
        assert pads
        for pad in pads:
            det = float(np.linalg.det(
                np.column_stack([pad.plane_u, pad.plane_v, pad.axis])))
            assert det > 0.5, f"left-handed pad basis (det={det:.2f})"
