# SPDX-License-Identifier: LGPL-2.1-or-later
"""Adversarial tests: hunting bugs in the untested seams.

Every prior fixture cuts from the top, is axis-aligned, and every hole is
through. Volume round-trips cannot see "right material removed from the
wrong place", so this suite upgrades verification to a two-way surface
distance match in WORLD coordinates, then attacks:

* blind holes (never fixtured before),
* a counterbore machined from the BOTTOM face (side-ness),
* an arbitrarily rotated part end-to-end through the history planner,
* a hollow boss (boss and hole sharing an axis),
* a side-drilled hole (must land in `unplanned` gracefully, not crash).
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.history import SketchCircle, plan_history
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

pytest.importorskip("manifold3d")

SECTIONS = 64


def _plan(mesh):
    report = snap_report(reconstruct(mesh)).report
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    return report, feats, plan_history(report, feats, detect_patterns(feats),
                                       patches)


# ----------------------------------------------------------- round trip 2.0

def _rebuild_mesh(plan) -> trimesh.Trimesh:
    """Execute the plan with booleans, in the PLAN frame."""
    from shapely.geometry import Polygon

    def _ring(prims):
        pts = []
        for p in prims:
            pts.extend(p.sample())
        return pts

    def extrude(profile, z0, height, holes=()):
        # extrude_polygon's interior-ring handling proved unreliable
        # (holes silently dropped): punch holes with explicit booleans
        m = trimesh.creation.extrude_polygon(Polygon(_ring(profile)),
                                             height=height)
        m.apply_translation([0, 0, z0])
        for h in holes:
            hm = trimesh.creation.extrude_polygon(Polygon(_ring(h)),
                                                  height=height + 2.0)
            hm.apply_translation([0, 0, z0 - 1.0])
            m = m.difference(hm)
        return m

    def extrude_lateral(pad):
        # profile lives in (u, v); extrude along `axis` by pad.length.
        # trimesh extrudes the 2D polygon along local +z, so map local
        # (x, y, z) -> (plane_u, plane_v, axis) anchored at plane_origin.
        poly = Polygon([(float(p[0]), float(p[1])) for p in _ring(pad.profile)])
        m = trimesh.creation.extrude_polygon(poly, height=pad.length)
        T = np.eye(4)
        T[:3, 0] = pad.plane_u
        T[:3, 1] = pad.plane_v
        T[:3, 2] = pad.axis
        T[:3, 3] = pad.plane_origin
        m.apply_transform(T)
        return m

    L = plan.base.length
    solid = extrude(plan.base.profile, 0.0, L,
                    holes=getattr(plan.base, "hole_profiles", ()))
    top = L
    for pad in plan.pads:
        if getattr(pad, "axis", None) is not None:
            solid = solid.union(extrude_lateral(pad))
        elif getattr(pad, "from_top", True):
            solid = solid.union(extrude(pad.profile, L, pad.length))
            top = max(top, L + pad.length)
        else:
            solid = solid.union(extrude(pad.profile, -pad.length, pad.length))
    for pk in plan.pockets:
        hp = getattr(pk, "hole_profiles", ())
        if getattr(pk, "through", False):
            solid = solid.difference(
                extrude(pk.profile, -2.0, top + 4.0, holes=hp))
        else:
            z0 = (L - pk.depth) if getattr(pk, "from_top", True) else -1.0
            solid = solid.difference(
                extrude(pk.profile, z0, pk.depth + 1.0, holes=hp))
    for h in plan.holes:
        from_top = getattr(h, "from_top", True)
        # The executor cuts a below-top counterbore from the GLOBAL TOP down
        # through its floor (drill ThroughAll + counterbore length extended by
        # cb_extra = L - surface_z); the shoulder terraces that used to form
        # the recess in this proxy are now dropped as redundant, so mirror the
        # executor here: cut the counterbore from L down to (surface_z -
        # counterbore_depth). manifold3d fuses this cleanly (the old artifact
        # was from cutting AT surface_z, double-cutting a terrace face).
        sz = getattr(h, "surface_z", None)
        sz = float(sz) if sz is not None else (L if from_top else 0.0)
        cb_extra = (L - sz) if from_top else 0.0
        for (x, y) in h.positions:
            if h.through:
                below = sum(p.length for p in plan.pads
                            if not getattr(p, "from_top", True))
                cyl = trimesh.creation.cylinder(radius=h.diameter / 2,
                                                height=top + below + 8.0,
                                                sections=SECTIONS)
                cyl.apply_translation([x, y, (top - below) / 2])
            else:
                dh = h.depth + cb_extra + 1.0
                cyl = trimesh.creation.cylinder(radius=h.diameter / 2,
                                                height=dh, sections=SECTIONS)
                floor = (L - (h.depth + cb_extra)) if from_top \
                    else (h.depth + cb_extra)
                zc = floor + dh / 2 if from_top else floor - dh / 2
                cyl.apply_translation([x, y, zc])
            solid = solid.difference(cyl)
            if h.counterbore_diameter:
                ch = h.counterbore_depth + cb_extra + 1.0
                cb = trimesh.creation.cylinder(
                    radius=h.counterbore_diameter / 2,
                    height=ch, sections=SECTIONS)
                floor = (L - (h.counterbore_depth + cb_extra)) if from_top \
                    else (h.counterbore_depth + cb_extra)
                zc = floor + ch / 2 if from_top else floor - ch / 2
                cb.apply_translation([x, y, zc])
                solid = solid.difference(cb)
            if getattr(h, "countersink_diameter", None):
                # revolve the conical cavity: drill radius at the throat out
                # to the countersink radius at the mouth face (surface_z).
                from shapely.geometry import Polygon as _Poly
                dr = h.diameter / 2.0
                cr = h.countersink_diameter / 2.0
                ha = np.deg2rad((h.countersink_angle or 90.0) / 2.0)
                run = (cr - dr) / np.tan(ha)
                mouth = sz if from_top else 0.0
                if from_top:
                    throat, over = mouth - run, mouth + 1.0
                    prof = [(0.0, throat), (dr, throat), (cr, mouth),
                            (cr, over), (0.0, over)]
                else:
                    throat, over = mouth + run, mouth - 1.0
                    # reverse winding so the revolve faces outward (a volume)
                    prof = [(0.0, over), (cr, over), (cr, mouth),
                            (dr, throat), (0.0, throat)]
                tool = trimesh.creation.revolve(
                    _Poly(prof).exterior.coords, sections=SECTIONS)
                tool.apply_translation([x, y, 0.0])
                solid = solid.difference(tool)
    # apply fillets (convex, ~perpendicular blends) via corner-tool cuts;
    # FilletOps live in WORLD coordinates, this solid is in the PLAN
    # frame -- transform each op's geometry into the plan frame first
    Minv = np.eye(4)
    Minv[:3, 0], Minv[:3, 1], Minv[:3, 2] = plan.frame_x, plan.frame_y, plan.frame_z
    Minv[:3, 3] = plan.frame_origin
    Minv = np.linalg.inv(Minv)

    def loc(v):
        return (Minv[:3, :3] @ np.asarray(v, dtype=float)) + Minv[:3, 3]

    def vec(v):
        return Minv[:3, :3] @ np.asarray(v, dtype=float)

    diag_pf = float(np.linalg.norm(solid.bounds[1] - solid.bounds[0]))
    for ch in getattr(plan, "cross_holes", []):
        axis = vec(ch.axis)
        if getattr(ch, "through", True):
            cyl = trimesh.creation.cylinder(radius=ch.diameter / 2,
                                            height=3.0 * diag_pf, sections=96)
            cyl.apply_transform(
                trimesh.geometry.align_vectors([0, 0, 1.0], axis))
            for pos in ch.positions3d:
                c = cyl.copy()
                c.apply_translation(loc(pos))
                solid = solid.difference(c)
        else:
            # blind: a depth-limited bore from each ENTRY point along the
            # inward direction, with a small outward overhang for a clean cut
            direction = vec(ch.entry_direction)
            pad = 0.02 * diag_pf
            length = float(ch.depth) + pad
            cyl = trimesh.creation.cylinder(radius=ch.diameter / 2,
                                            height=length, sections=96)
            cyl.apply_transform(
                trimesh.geometry.align_vectors([0, 0, 1.0], direction))
            for pos in ch.positions3d:
                c = cyl.copy()
                # cylinder is centred on its axis; shift so it spans from
                # (entry - pad) to (entry + depth) along the inward direction
                centre = loc(pos) + (length / 2.0 - pad) * direction
                c.apply_translation(centre)
                solid = solid.difference(c)

    for co in getattr(plan, "chamfers", []):
        e0, e1 = loc(co.edge_start), loc(co.edge_end)
        d = vec(co.direction)
        na, nb = vec(co.n_a), vec(co.n_b)
        s = co.size
        m = 0.5 * s
        span_d = e1 - e0
        pts = []
        for base in (e0 - 0.05 * span_d, e1 + 0.05 * span_d):
            for a2, b2 in ((m, m), (0.0, -s), (-s, 0.0)):
                pts.append(base + a2 * na + b2 * nb)
        solid = solid.difference(
            trimesh.Trimesh(vertices=np.array(pts)).convex_hull)

    for fo in getattr(plan, "fillets", []):
        e0, e1 = loc(fo.edge_start), loc(fo.edge_end)
        d = vec(fo.direction)
        na, nb = vec(fo.n_a), vec(fo.n_b)
        span = float(np.linalg.norm(e1 - e0))
        mid = 0.5 * (e0 + e1)
        r = fo.radius
        # paddings must be RELATIVE: fit noise in the edge estimate scales
        # with the part, and an absolute 1-unit outward overhang left
        # sharp-corner slivers uncut at 25x scale. Outward overhang cannot
        # over-cut (it is outside the part); the inward faces stay exactly
        # at r (tangent planes).
        out = 0.5 * r
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = d, na, nb
        T[:3, 3] = mid + ((out - r) / 2.0) * (na + nb)
        box = trimesh.creation.box(
            extents=[1.1 * span, r + out, r + out], transform=T)
        cyl = trimesh.creation.cylinder(radius=r, height=1.1 * span,
                                        sections=96)
        A = np.eye(4)
        A[:3, 0], A[:3, 1], A[:3, 2] = na, nb, d   # cylinder axis -> d
        A[:3, 3] = mid - r * (na + nb)
        cyl.apply_transform(A)
        if fo.convex:
            solid = solid.difference(box.difference(cyl))
        else:
            # concave: ADD the wedge filler between the corner and the
            # fillet cylinder; box spans corner -> +r along both (void-
            # pointing) normals, padded only on the material side, exact
            # span (over-length would create floating junk when unioning)
            T2 = np.eye(4)
            T2[:3, 0], T2[:3, 1], T2[:3, 2] = d, na, nb
            pad = 0.5 * r
            T2[:3, 3] = mid + ((r - pad) / 2.0) * (na + nb)
            box2 = trimesh.creation.box(
                extents=[span, r + pad, r + pad], transform=T2)
            A2 = np.eye(4)
            A2[:3, 0], A2[:3, 1], A2[:3, 2] = na, nb, d
            A2[:3, 3] = mid + r * (na + nb)
            cyl2 = trimesh.creation.cylinder(radius=r, height=span,
                                             sections=96)
            cyl2.apply_transform(A2)
            solid = solid.union(box2.difference(cyl2))
    return solid


def _to_world(plan, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = plan.frame_x, plan.frame_y, plan.frame_z
    M[:3, 3] = plan.frame_origin
    out = mesh.copy()
    out.apply_transform(M)
    return out


def assert_geometry_match(mesh: trimesh.Trimesh, plan, rel_tol=0.01, seed=0):
    """Two-way p99 surface distance in world coordinates."""
    rebuilt = _to_world(plan, _rebuild_mesh(plan))
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    tol = rel_tol * diag
    rng = np.random.default_rng(seed)
    for a, b, label in ((mesh, rebuilt, "mesh->rebuilt"),
                        (rebuilt, mesh, "rebuilt->mesh")):
        pts, _ = trimesh.sample.sample_surface(a, 2500, seed=rng.integers(1 << 30))
        _, d, _ = trimesh.proximity.closest_point(b, pts)
        p99 = float(np.percentile(d, 99))
        assert p99 < tol, f"{label}: p99 dist {p99:.4f} > {tol:.4f}"
    # volume as a sanity companion
    assert abs(rebuilt.volume - mesh.volume) / mesh.volume < 0.01


# ----------------------------------------------------------------- fixtures

def blind_hole_plate():
    plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])   # z in [-5, 5]
    drill = trimesh.creation.cylinder(radius=4.0, height=12.0, sections=SECTIONS)
    drill.apply_translation([8.0, -6.0, 5.0])                  # z in [-1, 11]
    return plate.difference(drill)                             # blind, depth 6


def bottom_counterbore_plate():
    plate = trimesh.creation.box(extents=[40.0, 30.0, 5.0])    # z in [-2.5, 2.5]
    drill = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=SECTIONS)
    bore = trimesh.creation.cylinder(radius=6.0, height=4.0, sections=SECTIONS)
    bore.apply_translation([0.0, 0.0, -2.5])                   # z in [-4.5, -0.5]
    return plate.difference(drill).difference(bore)            # cb on BOTTOM


def hollow_boss():
    base = trimesh.creation.cylinder(radius=10.0, height=8.0, sections=SECTIONS)
    base.apply_translation([0, 0, 4.0])                        # z in [0, 8]
    boss = trimesh.creation.cylinder(radius=6.0, height=10.0, sections=SECTIONS)
    boss.apply_translation([0, 0, 13.0])                       # z in [8, 18]
    hole = trimesh.creation.cylinder(radius=3.0, height=40.0, sections=SECTIONS)
    return base.union(boss).difference(hole)


def side_drilled_plate():
    plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
    drill = trimesh.creation.cylinder(radius=3.0, height=60.0, sections=SECTIONS)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    drill.apply_transform(rot)                                 # axis = x
    return plate.difference(drill)


ROT = trimesh.transformations.rotation_matrix(0.71, [1.0, 2.0, 0.4],
                                              point=[5.0, -3.0, 2.0])


# -------------------------------------------------------------------- tests

class TestBlindHole:
    def test_feature(self):
        report, feats, _ = _plan(blind_hole_plate())
        holes = feats.by_kind("hole")
        assert len(holes) == 1
        assert holes[0].params["through"] is False
        assert np.isclose(holes[0].params["depth"], 6.0)

    def test_geometry_roundtrip(self):
        mesh = blind_hole_plate()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)


class TestBottomCounterbore:
    def test_feature(self):
        report, feats, _ = _plan(bottom_counterbore_plate())
        cbs = feats.by_kind("counterbore")
        assert len(cbs) == 1
        assert cbs[0].params["counterbore_diameter"] == 12.0

    def test_geometry_roundtrip(self):
        mesh = bottom_counterbore_plate()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)


class TestRotatedPart:
    def test_geometry_roundtrip_in_world(self):
        from .test_composites import counterbored_plate
        mesh = counterbored_plate()
        mesh.apply_transform(ROT)
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)


class TestHollowBoss:
    def test_geometry_roundtrip(self):
        mesh = hollow_boss()
        _, _, plan = _plan(mesh)
        assert_geometry_match(mesh, plan)


class TestSideHole:
    def test_becomes_a_cross_hole_op(self):
        # contract FLIPPED in v0.14: side through-holes are now rebuilt
        # (see test_completeness); here we just pin that nothing is
        # silently dropped
        mesh = side_drilled_plate()
        _, feats, plan = _plan(mesh)
        assert feats.by_kind("hole")
        assert plan.cross_holes and plan.unplanned == []
