# SPDX-License-Identifier: LGPL-2.1-or-later
"""Rule-based feature inference: from recognized surfaces to named
manufacturing features.

The recognition pipeline delivers *surfaces*; designers think in
*features*. This module bridges them with explicit geometric rules:

* a HOLE is a concave full-revolution cylinder whose ends are explained
  by surrounding planes (hole loops on the cylinder) or a floor disk,
* a COUNTERBORE is a coaxial stack of two holes joined by an annular
  shoulder,
* a BOSS is the convex mirror image: a cylinder standing on a surrounding
  plane, capped by a disk,
* a SLOT is a stadium-shaped opening (two parallel lines + two
  semicircular arcs of equal radius) with two concave half-cylinder end
  walls; slots are claimed BEFORE fillets so the end walls are not
  mislabelled as blends,
* a FILLET is a partial-revolution cylinder (blends have no full turn),
* a POCKET is a non-circular opening loop in a plane with perpendicular
  walls and a parallel floor.

Concavity is decided by the mesh itself: outward face normals of a hole
wall point *toward* the axis, of a boss/shaft *away* from it. Every
feature records the surface indices it explains (provenance), and each
surface is consumed by at most one feature; whatever no rule explains is
left unassigned rather than guessed -- same honesty contract as the rest
of the pipeline.

This layer is deliberately rule-based: it doubles as the label generator
for the learned history-inference models planned later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .emission import PatchSpec, plan_patches
from .pipeline import ReconstructionReport
from .primitives import Cylinder, Plane

__all__ = ["Feature", "FeatureReport", "detect_features"]


@dataclass
class Feature:
    kind: str                      # 'hole' | 'counterbore' | 'boss' | 'fillet' | 'pocket'
    surface_indices: list[int]     # indices into report.surfaces (provenance)
    params: dict
    description: str


@dataclass
class FeatureReport:
    features: list[Feature] = field(default_factory=list)
    unassigned: list[int] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Feature]:
        return [f for f in self.features if f.kind == kind]


# --------------------------------------------------------------------------
# geometric predicates
# --------------------------------------------------------------------------

def _diag(report: ReconstructionReport) -> float:
    pts = np.vstack([s.segment.points for s in report.surfaces])
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _is_concave_cylinder(surf) -> bool:
    """Outward face normals of a hole wall point toward the axis."""
    cyl: Cylinder = surf.fit.primitive
    k = len(surf.segment.face_indices)
    centroids = surf.segment.samples[:k]          # samples start with centroids
    v = centroids - cyl.point
    radial = v - np.outer(v @ cyl.axis, cyl.axis)
    n = np.linalg.norm(radial, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    radial /= n
    return float(np.mean(np.einsum("ij,ij->i",
                                   surf.segment.face_normals, radial))) < 0.0


def _loop_on_cylinder(loop: np.ndarray, cyl: Cylinder, tol: float) -> bool:
    return bool(np.all(np.abs(cyl.radial_distance(loop) - cyl.radius) < tol))


def _axial(pts: np.ndarray, cyl: Cylinder) -> np.ndarray:
    return (pts - cyl.point) @ cyl.axis


def _split_loops(segment) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """(outer boundary loop, interior hole loops) of a segment, outer
    identified by the largest enclosed area."""
    loops = segment.boundary_loops
    if not loops:
        return None, []
    areas = []
    for lp in loops:
        c = lp.mean(axis=0)
        areas.append(0.5 * float(np.linalg.norm(
            np.cross(lp - c, np.roll(lp, -1, axis=0) - c).sum(axis=0))))
    k = int(np.argmax(areas))
    return loops[k], [lp for j, lp in enumerate(loops) if j != k]


def _plane_hole_loops_on(surf_plane, cyl: Cylinder, tol: float) -> list[float]:
    """Axial positions of this plane's *interior* hole loops lying on
    ``cyl``. The outer boundary is deliberately excluded: a shoulder or
    cap whose rim sits on a cylinder is an end DISK for that cylinder,
    not a surrounding opening -- conflating the two turns base bodies
    into phantom bosses."""
    _, holes = _split_loops(surf_plane.segment)
    return [float(np.mean(_axial(lp, cyl))) for lp in holes
            if _loop_on_cylinder(lp, cyl, tol)]


def _outer_loop_is_disk_on(surf_plane, cyl: Cylinder, tol: float) -> bool:
    """True when the plane's *outer* boundary lies on the cylinder (a cap
    or floor disk)."""
    outer, _ = _split_loops(surf_plane.segment)
    return outer is not None and _loop_on_cylinder(outer, cyl, tol)


def _is_concave_cone(surf) -> bool:
    """Outward face normals of a conical HOLE wall point toward the axis
    (a countersink), of a conical stud point away from it."""
    from .primitives import Cone
    cone: Cone = surf.fit.primitive
    k = len(surf.segment.face_indices)
    centroids = surf.segment.samples[:k]
    v = centroids - cone.apex
    h = v @ cone.axis
    radial = v - np.outer(h, cone.axis)
    n = np.linalg.norm(radial, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    radial /= n
    return float(np.mean(np.einsum("ij,ij->i",
                                   surf.segment.face_normals, radial))) < 0.0


def _plane_hole_loops_on_cone(surf_plane, cone, tol: float) -> list:
    """This plane's *interior* hole loops whose points lie on ``cone`` (the
    conical mouth's wide rim opening at a face)."""
    _, holes = _split_loops(surf_plane.segment)
    return [lp for lp in holes
            if bool(np.all(cone.distance(lp) < tol))]


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------



def _stadium(loop2d: np.ndarray) -> dict | None:
    """Recognize a stadium (obround) loop: 2 parallel lines + 2 equal
    semicircular arcs. Returns geometry or None."""
    from .history import SketchArc, SketchLine, loop_to_sketch  # lazy: no cycle
    prims = loop_to_sketch(loop2d)
    lines = [p for p in prims if isinstance(p, SketchLine)]
    arcs = [p for p in prims if isinstance(p, SketchArc)]
    if len(prims) != 4 or len(lines) != 2 or len(arcs) != 2:
        return None
    r1, r2 = arcs[0].radius, arcs[1].radius
    if abs(r1 - r2) > 0.02 * max(r1, r2):
        return None
    for a in arcs:
        if abs(abs(a.sweep) - np.pi) > np.deg2rad(20):
            return None
    d1 = lines[0].end - lines[0].start
    d2 = lines[1].end - lines[1].start
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) > 1e-3 * np.linalg.norm(d1) * np.linalg.norm(d2):
        return None
    r = 0.5 * (r1 + r2)
    c1, c2 = arcs[0].center, arcs[1].center
    return {"width": 2.0 * r, "radius": r,
            "length": float(np.linalg.norm(c2 - c1)) + 2.0 * r,
            "centers_2d": (c1, c2),
            "midpoint_2d": 0.5 * (c1 + c2)}

def detect_features(report: ReconstructionReport,
                    patches: list[PatchSpec] | None = None) -> FeatureReport:
    if not report.surfaces:
        return FeatureReport()
    if patches is None:
        patches = plan_patches(report)
    tol = max(1e-3 * _diag(report), 1e-9)

    planes = [i for i, s in enumerate(report.surfaces)
              if isinstance(s.fit.primitive, Plane)]
    cyls = [i for i, s in enumerate(report.surfaces)
            if isinstance(s.fit.primitive, Cylinder)]
    from .primitives import Cone
    cones = [i for i, s in enumerate(report.surfaces)
             if isinstance(s.fit.primitive, Cone)]

    out = FeatureReport()
    consumed: set[int] = set()

    # ---- classify cylinders ------------------------------------------------
    full_concave, full_convex, partial = [], [], []
    for i in cyls:
        if patches[i].full_u:
            (full_concave if _is_concave_cylinder(report.surfaces[i])
             else full_convex).append(i)
        else:
            partial.append(i)

    def _ends(i: int) -> dict:
        """End conditions of cylinder i: surrounding planes and floor disks."""
        cyl = report.surfaces[i].fit.primitive
        openings, floors = [], []
        for p in planes:
            sp = report.surfaces[p]
            if abs(float(sp.fit.primitive.normal @ cyl.axis)) < 0.999:
                continue
            for z in _plane_hole_loops_on(sp, cyl, tol):
                openings.append((p, z))
            if _outer_loop_is_disk_on(sp, cyl, tol):
                floors.append((p, float(np.mean(_axial(sp.segment.points, cyl)))))
        return {"openings": openings, "floors": floors}

    # ---- countersinks: a concave cone capping a coaxial drilled cylinder ---
    # A countersunk hole is the conical entry (a full-revolution concave
    # cone) sitting on the opening side of a drilled cylinder: the cone's
    # wide rim opens at a face, its narrow end matches the drill radius.
    # Claimed BEFORE the plain-hole pass so the drill is described as a
    # countersunk hole, not a bare hole with an unassigned cone. The drill
    # is then removed from the concave list so no coaxial stack re-claims it.
    from .standards import identify_metric
    cs_drills: set[int] = set()
    for ci in cones:
        scone = report.surfaces[ci]
        cone = scone.fit.primitive
        if not patches[ci].full_u or not _is_concave_cone(scone):
            continue
        ha = cone.half_angle
        h_lo, h_hi = patches[ci].v_range          # heights above apex
        r_narrow = h_lo * np.tan(ha)
        r_wide = h_hi * np.tan(ha)
        drill = None
        for di in full_concave:
            if di in consumed or di in cs_drills:
                continue
            dc = report.surfaces[di].fit.primitive
            if abs(float(dc.axis @ cone.axis)) < 1.0 - 1e-6:
                continue                           # not collinear direction
            off = dc.point - cone.apex
            perp = off - float(off @ cone.axis) * cone.axis
            if float(np.linalg.norm(perp)) > 5.0 * tol:
                continue                           # not the same axis line
            if abs(dc.radius - r_narrow) > tol + 0.06 * max(dc.radius, r_narrow):
                continue                           # radius mismatch at throat
            drill = di
            break
        if drill is None:
            continue
        dc = report.surfaces[drill].fit.primitive
        axis = cone.axis.copy()                    # apex -> opening (mouth)
        dh = (report.surfaces[drill].segment.points - cone.apex) @ axis
        mouth_h = float(h_hi)
        mouth = cone.apex + mouth_h * axis
        dinfo = _ends(drill)
        through = len(dinfo["openings"]) >= 1
        depth = mouth_h - float(dh.min())          # mouth -> floor / far face
        anchor = dc.point - float(dc.point @ dc.axis) * dc.axis
        used = [drill, ci]
        for p in planes:
            if _plane_hole_loops_on_cone(report.surfaces[p], cone, tol) \
                    and p not in used:
                used.append(p)
        for p, _z in dinfo["openings"]:
            if p not in used:
                used.append(p)
        if not through:
            for p, _z in dinfo["floors"][:1]:
                if p not in used:
                    used.append(p)
        consumed.add(drill)
        consumed.add(ci)
        cs_drills.add(drill)
        std = identify_metric(2 * dc.radius)
        out.features.append(Feature(
            kind="hole",
            surface_indices=used,
            params={
                "diameter": 2 * dc.radius,
                "depth": depth,
                "through": through,
                "position": anchor.tolist(),
                "axis": axis.tolist(),
                "standard": std,
                "countersink": True,
                "countersink_diameter": 2 * r_wide,
                "countersink_angle": float(np.rad2deg(2 * ha)),
                "mouth": mouth.tolist(),
            },
            description=(
                f"Countersunk hole d{2 * dc.radius:g} "
                f"{'through' if through else f'x {depth:g} blind'}, "
                f"csink d{2 * r_wide:g} x {np.rad2deg(2 * ha):.0f} deg"),
        ))
    full_concave = [i for i in full_concave if i not in cs_drills]

    # ---- coaxial stacks of concave cylinders -> holes / counterbores ------
    stacks: list[list[int]] = []
    for i in full_concave:
        ci = report.surfaces[i].fit.primitive
        placed = False
        for st in stacks:
            cj = report.surfaces[st[0]].fit.primitive
            if abs(float(ci.axis @ cj.axis)) > 1.0 - 1e-9 \
                    and cj.radial_distance([ci.point])[0] < tol:
                st.append(i)
                placed = True
                break
        if not placed:
            stacks.append([i])

    for st in stacks:
        st.sort(key=lambda i: report.surfaces[i].fit.primitive.radius)
        infos = {i: _ends(i) for i in st}
        spans = {i: patches[i].v_range for i in st}
        anchors = {}
        for i in st:
            c = report.surfaces[i].fit.primitive
            anchors[i] = c.point - (c.point @ c.axis) * c.axis

        if len(st) == 2:
            small, big = st
            r1 = report.surfaces[small].fit.primitive.radius
            r2 = report.surfaces[big].fit.primitive.radius
            # shoulder: an annular plane whose hole loop is on the small
            # cylinder AND whose outer loop is on the big one
            shoulder = None
            csmall = report.surfaces[small].fit.primitive
            cbig = report.surfaces[big].fit.primitive
            for p in planes:
                sp = report.surfaces[p]
                if _outer_loop_is_disk_on(sp, cbig, tol) \
                        and _plane_hole_loops_on(sp, csmall, tol):
                    shoulder = p
                    break
            if shoulder is not None:
                through = len(infos[small]["openings"]) >= 1
                depth_small = spans[small][1] - spans[small][0]
                depth_big = spans[big][1] - spans[big][0]
                used = []
                for u in ([small, big, shoulder]
                          + [p for p, _ in infos[small]["openings"]]
                          + [p for p, _ in infos[big]["openings"]]):
                    if u not in used:
                        used.append(u)
                consumed.update(u for u in used if u in (small, big))
                out.features.append(Feature(
                    kind="counterbore",
                    surface_indices=used,
                    params={
                        "diameter": 2 * r1,
                        "counterbore_diameter": 2 * r2,
                        "counterbore_depth": depth_big,
                        "depth": depth_small + depth_big,
                        "through": through,
                        "position": anchors[small].tolist(),
                        "axis": csmall.axis.tolist(),
                    },
                    description=(f"Counterbored hole d{2*r1:g}"
                                 f"{' through' if through else ''},"
                                 f" cb d{2*r2:g} x {depth_big:g}"),
                ))
                continue

        for i in st:  # plain holes (single, or stack without shoulder)
            if i in consumed:
                continue
            cyl = report.surfaces[i].fit.primitive
            info = infos[i]
            n_open, n_floor = len(info["openings"]), len(info["floors"])
            if n_open == 0 and n_floor == 0:
                continue  # a tube wall with unexplained ends: not a hole
            through = n_open >= 2
            depth = spans[i][1] - spans[i][0]
            used = []
            candidates = [i] + [p for p, _ in info["openings"]]
            if not through:
                candidates += [p for p, _ in info["floors"][:1]]
            for u in candidates:
                if u not in used:
                    used.append(u)
            consumed.add(i)
            from .standards import identify_metric
            std = identify_metric(2 * cyl.radius)
            out.features.append(Feature(
                kind="hole",
                surface_indices=used,
                params={
                    "diameter": 2 * cyl.radius,
                    "depth": depth,
                    "through": through,
                    "position": anchors[i].tolist(),
                    "axis": cyl.axis.tolist(),
                    "standard": std,
                },
                description=(f"Hole d{2*cyl.radius:g} "
                             f"{'through' if through else f'x {depth:g} blind'}"
                             + (f" ({std})" if std else "")),
            ))

    # ---- convex full cylinders -> bosses -----------------------------------
    for i in full_convex:
        if i in consumed:
            continue
        cyl = report.surfaces[i].fit.primitive
        info = _ends(i)
        cap = next((p for p, _ in info["floors"]), None)
        opening = next((p for p, _ in info["openings"]), None)
        if cap is None or opening is None:
            continue  # e.g. the base shaft of a stepped part: leave unassigned
        height = patches[i].v_range[1] - patches[i].v_range[0]
        used = [i, cap, opening]
        consumed.add(i)
        out.features.append(Feature(
            kind="boss",
            surface_indices=used,
            params={"diameter": 2 * cyl.radius, "height": height,
                    "position": (cyl.point - (cyl.point @ cyl.axis) * cyl.axis).tolist(),
                    "axis": cyl.axis.tolist()},
            description=f"Boss d{2*cyl.radius:g} x {height:g}",
        ))

    # ---- stadium openings -> slots (claimed before fillets!) ---------------
    for p in planes:
        sp = report.surfaces[p]
        n_top = sp.fit.primitive.normal
        for hole2d in patches[p].holes:
            st = _stadium(hole2d)
            if st is None:
                continue
            # end walls: concave partial cylinders of matching radius whose
            # axis anchors project onto the arc centers
            pf = patches[p]
            ends = []
            for c1 in st["centers_2d"]:
                c3 = pf.origin + c1[0] * pf.x_dir + c1[1] * pf.y_dir
                for ci in partial:
                    if ci in consumed:
                        continue
                    cyl = report.surfaces[ci].fit.primitive
                    if abs(float(cyl.axis @ n_top)) < 0.999:
                        continue
                    if abs(cyl.radius - st["radius"]) > tol + 0.02 * st["radius"]:
                        continue
                    if cyl.radial_distance([c3])[0] < 5 * tol \
                            and _is_concave_cylinder(report.surfaces[ci]):
                        ends.append(ci)
                        break
            if len(ends) != 2:
                continue
            span = patches[ends[0]].v_range
            depth = float(span[1] - span[0])
            # through: another plane carries a stadium with matching centers
            through = False
            for q in planes:
                if q == p:
                    continue
                sq = report.surfaces[q]
                if abs(float(sq.fit.primitive.normal @ n_top)) < 0.999:
                    continue
                for h2 in patches[q].holes:
                    st2 = _stadium(h2)
                    if st2 is None:
                        continue
                    qf = patches[q]
                    m3 = qf.origin + st2["midpoint_2d"][0] * qf.x_dir \
                        + st2["midpoint_2d"][1] * qf.y_dir
                    p3 = pf.origin + st["midpoint_2d"][0] * pf.x_dir \
                        + st["midpoint_2d"][1] * pf.y_dir
                    lateral = (m3 - p3) - ((m3 - p3) @ n_top) * n_top
                    if np.linalg.norm(lateral) < 5 * tol:
                        through = True
                        break
                if through:
                    break
            mid3 = pf.origin + st["midpoint_2d"][0] * pf.x_dir \
                + st["midpoint_2d"][1] * pf.y_dir
            used = [p] + ends
            consumed.update(ends)
            out.features.append(Feature(
                kind="slot", surface_indices=used,
                params={"width": st["width"], "length": st["length"],
                        "depth": depth, "through": through, "open": False,
                        "position": mid3.tolist(),
                        "axis": (n_top if n_top @ n_top else n_top).tolist()},
                description=(f"Slot {st['length']:g} x {st['width']:g}"
                             + (" through" if through
                                else f" x {depth:g} deep")),
            ))

    # ---- open-ended slots: semicircular notches in OUTER boundaries --------
    for p in planes:
        sp = report.surfaces[p]
        n_top = sp.fit.primitive.normal
        pf = patches[p]
        if pf.polygon is None or len(pf.polygon) < 8:
            continue
        from .history import SketchArc, SketchLine, loop_to_sketch
        prims = loop_to_sketch(pf.polygon)
        for ai, pr in enumerate(prims):
            if not isinstance(pr, SketchArc):
                continue
            if abs(abs(pr.sweep) - np.pi) > np.deg2rad(25):
                continue
            prev = prims[(ai - 1) % len(prims)]
            nxt = prims[(ai + 1) % len(prims)]
            if not (isinstance(prev, SketchLine) and isinstance(nxt, SketchLine)):
                continue
            d1 = prev.end - prev.start
            d2 = nxt.end - nxt.start
            if abs(d1[0] * d2[1] - d1[1] * d2[0]) \
                    > 1e-2 * np.linalg.norm(d1) * np.linalg.norm(d2):
                continue
            r = pr.radius
            c3 = pf.origin + pr.center[0] * pf.x_dir + pr.center[1] * pf.y_dir
            end = None
            for ci in partial:
                if ci in consumed:
                    continue
                cyl = report.surfaces[ci].fit.primitive
                if abs(float(cyl.axis @ n_top)) < 0.999:
                    continue
                if abs(cyl.radius - r) > tol + 0.02 * r:
                    continue
                if cyl.radial_distance([c3])[0] < 5 * tol \
                        and _is_concave_cylinder(report.surfaces[ci]):
                    end = ci
                    break
            if end is None:
                continue
            # slot direction: from the arc centre towards the open mouth
            mouth2 = 0.5 * (prev.start + nxt.end)
            dir2 = mouth2 - pr.center
            length = float(np.linalg.norm(dir2)) + r
            dir2 = dir2 / max(np.linalg.norm(dir2), 1e-12)
            dir3 = dir2[0] * pf.x_dir + dir2[1] * pf.y_dir
            span = patches[end].v_range
            depth = float(span[1] - span[0])
            # through: the opposite parallel plane carries a matching notch
            through = False
            for q in planes:
                if q == p:
                    continue
                sq = report.surfaces[q]
                if abs(float(sq.fit.primitive.normal @ n_top)) < 0.999:
                    continue
                lateral = (c3 - sq.fit.primitive.point)
                lateral = lateral - (lateral @ n_top) * n_top
                # matching end cylinder pierces both faces; cheap check:
                # the end cylinder's span reaches this plane
                off = float((sq.fit.primitive.point
                             - patches[end].origin) @ n_top)
                if -tol <= off * np.sign(off or 1.0) <= depth + tol:
                    through = True
                    break
            consumed.add(end)
            out.features.append(Feature(
                kind="slot", surface_indices=[p, end],
                params={"width": 2.0 * r, "length": length,
                        "depth": depth, "through": through, "open": True,
                        "position": c3.tolist(),
                        "direction": dir3.tolist(),
                        "axis": n_top.tolist()},
                description=(f"Open slot {length:g} x {2*r:g}"
                             + (" through" if through
                                else f" x {depth:g} deep")),
            ))

    # ---- partial cylinders -> fillets / blends -----------------------------
    for i in partial:
        if i in consumed:
            continue
        cyl = report.surfaces[i].fit.primitive
        u0, u1 = patches[i].u_range
        arc = float(np.rad2deg(u1 - u0))
        if arc > 200.0:
            continue  # more than a blend: leave unassigned
        consumed.add(i)
        out.features.append(Feature(
            kind="fillet",
            surface_indices=[i],
            params={"radius": cyl.radius, "arc_degrees": arc,
                    "axis": cyl.axis.tolist(),
                    "convex": not _is_concave_cylinder(report.surfaces[i])},
            description=f"Fillet r{cyl.radius:g} ({arc:.0f} deg)",
        ))

    # ---- edge chamfers: bisecting plane strips ------------------------------
    from .primitives import Plane as _Plane
    claimed_strips: set[int] = set()
    for p in planes:
        if p in claimed_strips:
            continue
        sp = report.surfaces[p]
        n_p = sp.fit.primitive.normal.copy()
        if float(n_p @ sp.segment.face_normals.mean(axis=0)) < 0:
            n_p = -n_p
        pkeys = _keys(sp.segment.points)
        pair = []
        for q in planes:
            if q == p:
                continue
            sq = report.surfaces[q]
            n_q = sq.fit.primitive.normal.copy()
            if float(n_q @ sq.segment.face_normals.mean(axis=0)) < 0:
                n_q = -n_q
            # a chamfer strip's normal sits ~45 deg from BOTH neighbours
            if not (0.60 < float(n_p @ n_q) < 0.80):
                continue
            if len(pkeys & _keys(sq.segment.points)) >= 2:
                pair.append((q, n_q))
            if len(pair) == 2:
                break
        if len(pair) != 2:
            continue
        (a, n_a), (b, n_b) = pair
        if abs(float(n_a @ n_b)) > 0.05:       # blended faces ~perpendicular
            continue
        # the strip-between-faces and face-between-strips configurations
        # are angularly IDENTICAL (a top face flanked by two chamfers
        # bisects them perfectly); the chamfer is the NARROW one
        if sp.segment.area >= min(report.surfaces[a].segment.area,
                                  report.surfaces[b].segment.area):
            continue
        bisector = n_a + n_b
        bisector = bisector / np.linalg.norm(bisector)
        if float(n_p @ bisector) < 0.995:      # strip must bisect the pair
            continue
        d = np.cross(n_a, n_b)
        d = d / np.linalg.norm(d)
        c = sp.segment.points.mean(axis=0)
        # sharp edge point: on both neighbour planes, at the strip's level
        pa = report.surfaces[a].fit.primitive
        pb = report.surfaces[b].fit.primitive
        x = np.linalg.solve(
            np.vstack([n_a, n_b, d]),
            np.array([float(n_a @ pa.point), float(n_b @ pb.point),
                      float(d @ c)]))
        leg = float(np.sqrt(2.0) * abs(n_p @ (c - x)))
        h = (sp.segment.points - x) @ d
        angle = np.rad2deg(np.arccos(np.clip(abs(float(n_p @ n_a)), 0, 1)))
        claimed_strips.add(p)
        out.features.append(Feature(
            kind="chamfer", surface_indices=[p, a, b],
            params={"size": leg, "angle_deg": float(angle),
                    "length": float(h.max() - h.min()),
                    "axis": d.tolist(),
                    "position": (x + 0.5 * (h.max() + h.min()) * d).tolist()},
            description=f"Chamfer {leg:g} x {leg:g} ({angle:.0f} deg)"))

    # ---- prismatic pockets --------------------------------------------------
    for p in planes:
        if p in consumed:
            continue
        sp = report.surfaces[p]
        n_top = sp.fit.primitive.normal
        for hole in patches[p].holes:
            # circular openings belong to hole features; skip loops that lie
            # on any cylinder
            loop3d = _loop3d(patches[p], hole)
            if any(_loop_on_cylinder(loop3d, report.surfaces[c].fit.primitive, tol)
                   for c in cyls):
                continue
            # walls: unconsumed planes perpendicular to the top sharing
            # loop vertices; floor: parallel plane adjacent to the walls
            keys = _keys(loop3d)
            walls = []
            for w in planes:
                if w == p or w in consumed:
                    continue
                sw = report.surfaces[w]
                if abs(float(sw.fit.primitive.normal @ n_top)) > 1e-6:
                    continue
                if len(keys & _keys(sw.segment.points)) >= 2:
                    walls.append(w)
            if len(walls) < 3:
                continue
            wall_keys = set().union(*(_keys(report.surfaces[w].segment.points)
                                      for w in walls))
            floor = None
            # outward orientation of the opening plane: a pocket's floor
            # must lie BELOW it (against the outward normal). Without the
            # sign check, a tower standing in a ring shelf's hole loop is
            # classified as an inverted pocket -- its cap is a parallel
            # "floor" ABOVE the opening -- and the phantom pocket razes
            # the tower (field-observed on featuretype's raised decks).
            n_out = n_top.copy()
            if float(n_out @ sp.segment.face_normals.mean(axis=0)) < 0:
                n_out = -n_out
            for f in planes:
                if f == p or f in consumed or f in walls:
                    continue
                sf = report.surfaces[f]
                if abs(float(sf.fit.primitive.normal @ n_top)) < 0.999:
                    continue
                if float((sf.fit.primitive.point
                          - sp.fit.primitive.point) @ n_out) >= -tol:
                    continue                    # sits above: not a floor
                if len(wall_keys & _keys(sf.segment.points)) >= 3:
                    floor = f
                    break
            if floor is None:
                continue
            depth = abs(float((report.surfaces[floor].fit.primitive.point
                               - sp.fit.primitive.point) @ n_top))
            size = [float(hole[:, 0].max() - hole[:, 0].min()),
                    float(hole[:, 1].max() - hole[:, 1].min())]
            used = [p, floor] + walls
            used = [u for u in used if u not in consumed]
            consumed.update(used)
            out.features.append(Feature(
                kind="pocket",
                surface_indices=used,
                params={"depth": depth, "opening_size": size},
                description=f"Pocket {size[0]:g} x {size[1]:g} x {depth:g} deep",
            ))
            break

    referenced = {i for feat in out.features for i in feat.surface_indices}
    out.unassigned = [i for i in range(len(report.surfaces))
                      if i not in referenced]
    return out


def _loop3d(patch: PatchSpec, loop2d: np.ndarray) -> np.ndarray:
    return (patch.origin
            + loop2d[:, :1] * patch.x_dir
            + loop2d[:, 1:] * patch.y_dir)


def _keys(pts: np.ndarray) -> set:
    return {tuple(np.round(p, 9)) for p in np.asarray(pts)}
