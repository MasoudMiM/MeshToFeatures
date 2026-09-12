# SPDX-License-Identifier: LGPL-2.1-or-later
"""Headless plan execution and deviation correction.

:func:`plan_to_mesh` executes a :class:`~.history.BuildPlan` with mesh
booleans -- the same conventions as the FreeCAD executor (frame embedded
in world space, base over ``[0, L]``, cuts from the opening face) -- and
returns the rebuilt solid in WORLD coordinates. :func:`plan_corrections`
compares it with the source mesh and returns the connected components of
the symmetric difference as watertight patch solids: the OVER components
(material the rebuild has that the part lacks) and the UNDER components
(material the part has that the rebuild lacks). Applying them closes the
residual gap left by freeform geometry no analytic feature captured
(transition bands, scoops, tapered blend walls) -- the hybrid-modelling
fallback: parametric features stay primary, patches cover only what
recognition could not explain.

Both functions need a boolean engine (``manifold3d``); without one they
degrade to ``None`` / ``[]`` and the pipeline behaves as before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

__all__ = ["CorrectionOp", "boolean_engine_available", "plan_to_mesh",
           "plan_corrections", "apply_corrections", "add_corrections"]


@dataclass
class CorrectionOp:
    """One connected component of the rebuild-vs-mesh deviation.

    ``kind`` is ``'cut'`` for material the rebuild must lose, ``'add'``
    for material it must gain; ``mesh`` is a watertight patch solid in
    world coordinates (the plan frame embedded in mesh space).
    """

    kind: str                    # 'cut' | 'add'
    mesh: trimesh.Trimesh
    volume: float


def boolean_engine_available() -> bool:
    """True when trimesh can run booleans (manifold3d installed)."""
    try:
        import manifold3d  # noqa: F401
        return True
    except Exception:                                  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# cutter primitives
# --------------------------------------------------------------------------

def _poly_of(profile, holes=()):
    from shapely.geometry import Polygon
    pts = []
    for p in profile:
        pts.extend(p.sample())
    rings = []
    for h in holes:
        hp = []
        for p in h:
            hp.extend(p.sample())
        rings.append(hp)
    return Polygon(pts, rings)


def _extrude(poly, z0, height):
    m = trimesh.creation.extrude_polygon(poly, height=height)
    m.apply_translation([0.0, 0.0, z0])
    return m


def _cutter_z(radius: float, z0: float, z1: float, sections: int = 96):
    cyl = trimesh.creation.cylinder(radius=radius, height=z1 - z0,
                                    sections=sections)
    cyl.apply_translation([0.0, 0.0, 0.5 * (z0 + z1)])
    return cyl


def _frustum(r_lo: float, r_hi: float, z_lo: float, z_hi: float,
             center=(0.0, 0.0), sections: int = 96) -> trimesh.Trimesh:
    """Watertight frustum of revolution about the z axis (apex allowed:
    a zero radius becomes a single vertex)."""
    ang = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    c = np.asarray(center, dtype=float)
    cos, sin = np.cos(ang), np.sin(ang)
    verts: list = []
    faces: list = []

    def ring(r, z):
        return np.column_stack([c[0] + r * cos, c[1] + r * sin,
                                np.full(sections, z)])

    lo_is_apex = r_lo <= 1e-12
    hi_is_apex = r_hi <= 1e-12
    if lo_is_apex:
        verts.append(np.array([[c[0], c[1], z_lo]]))       # index 0
    else:
        verts.append(ring(r_lo, z_lo))                     # 0..s-1
    if hi_is_apex:
        verts.append(np.array([[c[0], c[1], z_hi]]))
    else:
        verts.append(ring(r_hi, z_hi))
    lo_base = 0
    hi_base = sections if not lo_is_apex else 1

    if lo_is_apex and hi_is_apex:
        raise ValueError("degenerate frustum: both ends are apexes")
    n_verts = sum(len(v) for v in verts)
    if lo_is_apex:                                         # cone, apex below
        for i in range(sections):
            faces.append([0, hi_base + i, hi_base + (i + 1) % sections])
        verts.append(np.array([[c[0], c[1], z_hi]]))       # cap centre
        tc = n_verts
        for i in range(sections):
            faces.append([tc, hi_base + (i + 1) % sections, hi_base + i])
    elif hi_is_apex:                                       # cone, apex above
        apex = hi_base
        for i in range(sections):
            faces.append([apex, lo_base + (i + 1) % sections, lo_base + i])
        verts.append(np.array([[c[0], c[1], z_lo]]))       # cap centre
        bc = n_verts
        for i in range(sections):
            faces.append([bc, lo_base + i, lo_base + (i + 1) % sections])
    else:                                                  # side + two caps
        for i in range(sections):
            j = (i + 1) % sections
            a, b = lo_base + i, lo_base + j
            d, e = hi_base + i, hi_base + j
            faces.append([a, b, e])
            faces.append([a, e, d])
        verts.append(np.array([[c[0], c[1], z_lo], [c[0], c[1], z_hi]]))
        bc, tc = n_verts, n_verts + 1
        for i in range(sections):
            j = (i + 1) % sections
            faces.append([bc, lo_base + j, lo_base + i])
            faces.append([tc, hi_base + i, hi_base + j])
    m = trimesh.Trimesh(vertices=np.vstack(verts),
                        faces=np.array(faces), process=True)
    m.fix_normals()
    return m


def _frame_matrix(plan) -> np.ndarray:
    t = np.eye(4)
    t[:3, 0] = plan.frame_x
    t[:3, 1] = plan.frame_y
    t[:3, 2] = plan.frame_z
    t[:3, 3] = plan.frame_origin
    return t


def _edge_cutter(op, shape_poly, sections: int = 48):
    """Extrude ``shape_poly`` (2D in the edge-cross-section plane) along a
    FilletOp/ChamferOp's edge; returns the positioned solid."""
    d = np.asarray(op.direction, dtype=float)
    d = d / np.linalg.norm(d)
    na = np.asarray(op.n_a, dtype=float)
    nb = np.asarray(op.n_b, dtype=float)
    na = na - (na @ d) * d
    na /= np.linalg.norm(na)
    nb = nb - (nb @ d) * d
    nb = nb - (nb @ na) * na
    nb /= np.linalg.norm(nb)
    length = float(np.linalg.norm(np.asarray(op.edge_end)
                                  - np.asarray(op.edge_start)))
    m = trimesh.creation.extrude_polygon(shape_poly, height=length,
                                         sections=sections)
    mat = np.eye(4)
    mat[:3, 0] = na
    mat[:3, 1] = nb
    mat[:3, 2] = d
    mat[:3, 3] = op.edge_start
    m.apply_transform(mat)
    return m


def _apply_fillets(solid, plan):
    from .history import blend_corner_profile, chamfer_corner_profile
    for op in getattr(plan, "fillets", []):
        r = float(op.radius)
        poly = _poly_of(blend_corner_profile(r, op.convex))
        if op.convex:
            solid = solid.difference(_edge_cutter(op, poly))
        else:
            solid = solid.union(_edge_cutter(op, poly))
    for op in getattr(plan, "chamfers", []):
        s = float(op.size)
        convex = _chamfer_convexity(plan, op, solid)
        poly = _poly_of(chamfer_corner_profile(s, convex))
        if convex:
            solid = solid.difference(_edge_cutter(op, poly))
        else:
            solid = solid.union(_edge_cutter(op, poly))
    return solid


def _chamfer_convexity(plan, op, solid) -> bool:
    """Does the sharp solid occupy the (-n_a, -n_b) quadrant at the edge?
    Probe a point just inside each quadrant against the solid itself."""
    mid = 0.5 * (np.asarray(op.edge_start) + np.asarray(op.edge_end))
    na = np.asarray(op.n_a, dtype=float)
    nb = np.asarray(op.n_b, dtype=float)
    eps = 0.25 * float(op.size)
    convex_pt = (mid - eps * na - eps * nb).reshape(1, 3)
    try:
        return bool(solid.contains(convex_pt)[0])
    except Exception:                                      # noqa: BLE001
        return True


# --------------------------------------------------------------------------
# headless execution
# --------------------------------------------------------------------------

def plan_to_mesh(plan, sections: int = 96) -> trimesh.Trimesh | None:
    """Execute ``plan`` with mesh booleans; world-coordinate solid, or
    ``None`` when a boolean engine is missing or an op is unsupported."""
    if not boolean_engine_available():
        return None
    try:
        return _plan_to_mesh(plan, sections)
    except Exception:                                      # noqa: BLE001
        return None


def _plan_to_mesh(plan, sections: int) -> trimesh.Trimesh:
    L = float(plan.base.length)
    solid = _extrude(_poly_of(plan.base.profile,
                              getattr(plan.base, "hole_profiles", [])),
                     0.0, L)

    # pads (vertical and lateral). Gusset webs live inside a recess;
    # they must be built AFTER the pockets that carve the cavity,
    # otherwise the pocket booleans create ghost components.
    from .history import lateral_pad_world_frame
    gusset_pads = []
    for p in getattr(plan, "pads", []):
        if getattr(p, "label", "") == "Gusset web":
            gusset_pads.append(p)
            continue
        if getattr(p, "axis", None) is not None:
            origin, u, v, axis = lateral_pad_world_frame(plan, p)
            m = _extrude(_poly_of(p.profile), 0.0, float(p.length))
            mat = np.eye(4)
            mat[:3, 0] = u
            mat[:3, 1] = v
            mat[:3, 2] = axis
            mat[:3, 3] = origin
            m.apply_transform(mat)
            solid = solid.union(m)
            continue
        z0 = L if getattr(p, "from_top", True) else -float(p.length)
        solid = solid.union(_extrude(_poly_of(p.profile), z0,
                                      float(p.length)))

    # pockets
    for pk in getattr(plan, "pockets", []):
        # When the pocket has a mouth profile, the cavity changes with
        # height; use the LARGER (mouth) profile as the prismatic cutter
        # -- an overestimate of the true loft volume, but far closer to
        # the mesh than the floor-only profile (which under-cuts by the
        # boss area).  The headless correction pass closes the remaining
        # gap exactly.
        use = getattr(pk, "mouth_profile", None) or pk.profile
        if getattr(pk, "through", False):
            z0, h = -1.0, L + 2.0
        elif getattr(pk, "from_top", True):
            z0, h = L - float(pk.depth), float(pk.depth) + 1.0
        else:
            z0, h = -1.0, float(pk.depth) + 1.0
        cutter = _extrude(_poly_of(use, getattr(pk, "hole_profiles", [])),
                          z0, h)
        solid = solid.difference(cutter)

    # holes (drill + counterbore + countersink). Mirrors the executor's
    # placement branches: a hole opens at ``surface_z`` (or the part face)
    # and is cut from the mouth toward the part interior -- downward for
    # top-side holes AND for bottom-side holes whose mouth is an internal
    # face (a recess floor), upward for plain bottom-side holes.
    holes_list = getattr(plan, "holes", [])
    for h in holes_list:
        top = getattr(h, "from_top", True)
        surf_z = getattr(h, "surface_z", None)
        tol_z = 1e-3 * L if L else 1e-6
        opens_below = surf_z is not None and float(surf_z) < L - tol_z
        open_z = (float(surf_z) if surf_z is not None
                  else (L if top else 0.0))
        downward = top or (surf_z is not None and not top)
        cb_extra = (L - float(surf_z)) if (opens_below and top) else 0.0
        for (x, y) in h.positions:
            if h.through:
                if opens_below and top:
                    z0, z1 = -1.0, L + 1.0       # from the global top,
                elif downward and not (not top and surf_z is not None):
                    # Regular downward through cut
                    z0, z1 = -1.0, open_z + 0.5
                elif not top and surf_z is not None:
                    # from-bottom through-hole opening on an internal face:
                    # drill UP from the bottom face, not DOWN from the opening
                    z0, z1 = -1.0, L + 1.0       # through everything
                else:
                    z0, z1 = open_z - 0.5, L + 1.0
                solid = solid.difference(
                    _cutter_z(h.diameter / 2, z0, z1, sections))
            else:
                depth = float(h.depth)
                if downward:
                    z0, z1 = open_z - depth, open_z + 0.5
                else:
                    z0, z1 = open_z - 0.5, open_z + depth
                solid = solid.difference(
                    _cutter_z(h.diameter / 2, z0, z1, sections))
            if h.counterbore_diameter:
                cbd = float(h.counterbore_depth) + cb_extra
                if downward:
                    z0, z1 = open_z - cbd, open_z + 0.5
                else:
                    z0, z1 = open_z - 0.5, open_z + cbd
                solid = solid.difference(
                    _cutter_z(h.counterbore_diameter / 2, z0, z1,
                              sections))
            if h.countersink_diameter:
                half = np.deg2rad(float(h.countersink_angle or 90.0) / 2.0)
                r_mouth = float(h.countersink_diameter) / 2.0
                r_drill = float(h.diameter) / 2.0
                cs_depth = (r_mouth - r_drill) / np.tan(half)
                if downward:
                    fr = _frustum(r_drill, r_mouth, open_z - cs_depth,
                                  open_z + 0.3, center=(x, y),
                                  sections=sections)
                else:
                    fr = _frustum(r_mouth, r_drill, open_z - 0.3,
                                  open_z + cs_depth, center=(x, y),
                                  sections=sections)
                solid = solid.difference(fr)

    # conical pockets / studs
    for c in getattr(plan, "cones", []):
        face_z = (float(c.surface_z) if c.surface_z is not None
                  else (L if c.from_top else 0.0))
        depth = (face_z + 1.0 if c.from_top else L + 1.0 - face_z) \
            if getattr(c, "through", False) else float(c.depth)
        for (x, y) in c.positions:
            if c.from_top:
                fr = _frustum(c.r_far, c.r_mouth,
                              face_z - depth, face_z + 0.5,
                              center=(x, y), sections=sections)
            else:
                fr = _frustum(c.r_mouth, c.r_far,
                              face_z - 0.5, face_z + depth,
                              center=(x, y), sections=sections)
            solid = solid.union(fr) if c.additive else solid.difference(fr)

    # into world coordinates (the plan frame is embedded in mesh space)
    solid.apply_transform(_frame_matrix(plan))

    # gusset webs: lateral pads that live inside a recess cavity.
    # They must be built after the frame transform AND after the
    # pocket booleans (otherwise the pocket cutter creates ghosts).
    # lateral_pad_world_frame returns WORLD coordinates, so no further
    # transform is needed.
    for p in getattr(plan, "pads", []):
        if getattr(p, "label", "") != "Gusset web":
            continue
        origin, u, v, axis = lateral_pad_world_frame(plan, p)
        m = _extrude(_poly_of(p.profile), 0.0, float(p.length))
        mat = np.eye(4)
        mat[:3, 0] = u
        mat[:3, 1] = v
        mat[:3, 2] = axis
        mat[:3, 3] = origin
        m.apply_transform(mat)
        solid = solid.union(m)

    # cross-axis holes, fillets and chamfers are specified in world coords
    for ch in getattr(plan, "cross_holes", []):
        axis = np.asarray(ch.axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        diag = float(np.linalg.norm(solid.bounds[1] - solid.bounds[0]))
        reach = 1.5 * diag + 2.0
        r = float(ch.diameter) / 2.0
        if getattr(ch, "through", True):
            for pos in ch.positions3d:
                cyl = trimesh.creation.cylinder(radius=r, height=reach,
                                                sections=sections)
                zhat = np.array([0.0, 0.0, 1.0])
                rot = trimesh.geometry.align_vectors(zhat, axis)
                cyl.apply_transform(rot)
                cyl.apply_translation(np.asarray(pos, dtype=float))
                solid = solid.difference(cyl)
        else:
            entry = np.asarray(ch.entry_direction, dtype=float)
            depth = float(ch.depth)
            for pos in ch.positions3d:
                start = np.asarray(pos, dtype=float) - 0.5 * entry
                cyl = trimesh.creation.cylinder(radius=r, height=depth + 0.5,
                                                sections=sections)
                zhat = np.array([0.0, 0.0, 1.0])
                cyl.apply_transform(trimesh.geometry.align_vectors(
                    zhat, entry))
                cyl.apply_translation(start + entry * (depth + 0.5) / 2.0)
                solid = solid.difference(cyl)

    solid = _apply_fillets(solid, plan)
    return solid


# --------------------------------------------------------------------------
# deviation correction
# --------------------------------------------------------------------------

def plan_corrections(
    plan, mesh: trimesh.Trimesh, min_volume: float | None = None,
) -> list[CorrectionOp]:
    """Connected components of (rebuild \\ mesh) and (mesh \\ rebuild).

    ``kind='cut'`` components are the excess of the parametric rebuild
    (OVER -- material the part lacks); ``kind='add'`` the deficit (UNDER
    -- material the rebuild lacks). Both are returned: the cut components
    quantify and localize what the feature tree over-built, and the
    canonical application (:func:`apply_corrections`, mirrored by the
    FreeCAD executor) removes all OVER at once by intersecting with the
    source mesh and unions back the UNDER patches. ``[]`` when no boolean
    engine is available or the plan cannot be executed headlessly.
    """
    solid = plan_to_mesh(plan)
    if solid is None:
        return []
    if min_volume is None:
        min_volume = max(1e-4 * float(mesh.volume), 1e-6)
    out: list[CorrectionOp] = []
    try:
        over = solid.difference(mesh)
        under = mesh.difference(solid)
    except Exception:                                      # noqa: BLE001
        return []
    for kind, diff in (("cut", over), ("add", under)):
        if diff is None or len(diff.faces) == 0:
            continue
        for comp in diff.split():
            if comp.volume < min_volume:
                continue
            out.append(CorrectionOp(kind=kind, mesh=comp,
                                    volume=float(comp.volume)))
    out.sort(key=lambda c: -c.volume)
    return out


def apply_corrections(
    solid: trimesh.Trimesh, mesh: trimesh.Trimesh,
    corrections: list[CorrectionOp],
) -> trimesh.Trimesh:
    """Canonical application of :func:`plan_corrections` output.

    Intersect with the source mesh (removes every OVER component at once,
    without subtracting coplanar-bounded patch solids one by one -- that
    sequential subtraction jitters by percent-level volumes on shared
    faces) and union back the UNDER patches. The FreeCAD executor uses
    this exact headless result (OCC's own booleans between a PartDesign
    compound and the faceted mesh degenerate), converted to a terminal
    ``Part::Feature``.
    """
    try:
        out = solid.intersection(mesh)
    except Exception:                                      # noqa: BLE001
        out = solid
    for c in corrections:
        if c.kind != "add":
            continue
        try:
            out = out.union(c.mesh)
        except Exception:                                  # noqa: BLE001
            continue
    return out


def add_corrections(plan, mesh: trimesh.Trimesh) -> list[CorrectionOp]:
    """Compute deviation corrections and attach them to ``plan`` in place.

    Sets ``plan.corrections`` and ``plan.source_mesh`` so an executor can
    apply them; returns the list (empty when no boolean engine is
    available or the plan cannot be executed headlessly, in which case
    the plan is left untouched and behaves exactly as before).
    """
    corrs = plan_corrections(plan, mesh)
    if corrs:
        plan.corrections = corrs
        plan.source_mesh = mesh
    return corrs
