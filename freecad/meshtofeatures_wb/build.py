# SPDX-License-Identifier: LGPL-2.1-or-later
"""FreeCAD executor for BuildPlans: PartDesign Body with editable sketches.

Pure transliteration -- all geometric decisions were made and round-trip
tested in ``meshtofeatures.history``. Conventions mirror the plan: sketches
attach at frame z-offsets; pockets/blind holes cut from the top face;
through holes use ThroughAll.
"""

from __future__ import annotations

import math

import FreeCAD as App  # type: ignore
import Part  # type: ignore

from .core.history import (BuildPlan, SketchArc, SketchCircle,
                               SketchLine, blend_corner_profile,
                               chamfer_corner_profile, fillet_edge_matches,
                               hole_op_properties)
from .core.fitting import _axis_frame


def _placement(plan: BuildPlan, z_offset: float,
               flip: bool = False) -> "App.Placement":
    """Sketch placement at a frame z-offset. ``flip`` mirrors the plane
    (x, -y, -z): used for from-bottom operations so the DEFAULT feature
    direction is correct by construction -- direction booleans like
    Pocket.Reversed proved unreliable on FreeCAD 1.1's refactored
    extrude features (field-observed: bottom-side steps cut into air)."""
    o = plan.frame_origin + z_offset * plan.frame_z
    fy = -plan.frame_y if flip else plan.frame_y
    fz = -plan.frame_z if flip else plan.frame_z
    m = App.Matrix(
        float(plan.frame_x[0]), float(fy[0]), float(fz[0]), float(o[0]),
        float(plan.frame_x[1]), float(fy[1]), float(fz[1]), float(o[1]),
        float(plan.frame_x[2]), float(fy[2]), float(fz[2]), float(o[2]),
        0.0, 0.0, 0.0, 1.0)
    return App.Placement(m)


def _mirror_y(prims):
    """Mirror 2D profile primitives across the x-axis (for flipped
    sketch planes): y coordinates negate, arc sweeps flip sign."""
    import numpy as np
    from .core.history import SketchArc, SketchCircle, SketchLine
    out = []
    for p in prims:
        if isinstance(p, SketchLine):
            out.append(SketchLine(start=p.start * np.array([1.0, -1.0]),
                                  end=p.end * np.array([1.0, -1.0])))
        elif isinstance(p, SketchCircle):
            out.append(SketchCircle(center=p.center * np.array([1.0, -1.0]),
                                    radius=p.radius))
        elif isinstance(p, SketchArc):
            out.append(SketchArc(center=p.center * np.array([1.0, -1.0]),
                                 radius=p.radius,
                                 start=p.start * np.array([1.0, -1.0]),
                                 end=p.end * np.array([1.0, -1.0]),
                                 sweep=-p.sweep))
    return out


def _add_geometry(sk, prims) -> None:
    Z = App.Vector(0, 0, 1)
    for p in prims:
        if isinstance(p, SketchLine):
            sk.addGeometry(Part.LineSegment(
                App.Vector(float(p.start[0]), float(p.start[1]), 0),
                App.Vector(float(p.end[0]), float(p.end[1]), 0)), False)
        elif isinstance(p, SketchCircle):
            sk.addGeometry(Part.Circle(
                App.Vector(float(p.center[0]), float(p.center[1]), 0),
                Z, float(p.radius)), False)
        elif isinstance(p, SketchArc):
            # endpoints must be the EXACT shared chain points: fitted-arc
            # endpoints can sit ~1e-4 off the raw loop points, and
            # angle-parametrized arcs land ON the circle, opening the
            # wire (field-observed: "Wire is not closed"). Build through
            # three points (start, on-arc mid, end): endpoints close
            # exactly, the interior deviates by at most the fit residual.
            a0 = math.atan2(p.start[1] - p.center[1], p.start[0] - p.center[0])
            am = a0 + 0.5 * p.sweep
            v_s = App.Vector(float(p.start[0]), float(p.start[1]), 0)
            v_m = App.Vector(float(p.center[0] + p.radius * math.cos(am)),
                             float(p.center[1] + p.radius * math.sin(am)), 0)
            v_e = App.Vector(float(p.end[0]), float(p.end[1]), 0)
            try:
                sk.addGeometry(Part.ArcOfCircle(v_s, v_m, v_e), False)
            except Exception:  # noqa: BLE001 - fall back to angle form
                a1 = a0 + p.sweep
                if p.sweep < 0:
                    a0, a1 = a1, a0
                circle = Part.Circle(
                    App.Vector(float(p.center[0]), float(p.center[1]), 0),
                    Z, float(p.radius))
                sk.addGeometry(Part.ArcOfCircle(circle, a0, a1), False)


def _circle_sketch(doc, body, plan, z, entries, label, flip=False):
    sk = doc.addObject("Sketcher::SketchObject", label)
    body.addObject(sk)
    sk.Placement = _placement(plan, z, flip=flip)
    _add_geometry(sk, _mirror_y(entries) if flip else entries)
    return sk


def _point_placement(plan: BuildPlan, x: float, y: float,
                     z: float) -> "App.Placement":
    """World placement of a PRIMITIVE based at frame point (x, y, z) with
    its local +z along the frame z. Unlike ``_placement`` this carries the
    in-plane offset, because a primitive is positioned by its placement
    rather than by geometry inside a sketch."""
    o = (plan.frame_origin + x * plan.frame_x + y * plan.frame_y
         + z * plan.frame_z)
    m = App.Matrix(
        float(plan.frame_x[0]), float(plan.frame_y[0]),
        float(plan.frame_z[0]), float(o[0]),
        float(plan.frame_x[1]), float(plan.frame_y[1]),
        float(plan.frame_z[1]), float(o[1]),
        float(plan.frame_x[2]), float(plan.frame_y[2]),
        float(plan.frame_z[2]), float(o[2]),
        0.0, 0.0, 0.0, 1.0)
    return App.Placement(m)


def _add_cone(doc, body, plan, name, x, y, z_lo, r_lo, r_hi, height,
              additive=False, label="", deferred=None):
    """Cut (or add) a cone as a placed PartDesign primitive.

    ``Radius1`` is the radius at the BASE and ``Radius2`` at the top, so the
    caller passes the radii at the low-z and high-z ends and the primitive
    needs no rotation beyond the frame's. A primitive is used rather than a
    tapered pocket because the taper SIGN is a convention that cannot be
    verified without FreeCAD, and rather than a sketch-and-revolve because
    there is no profile wire to fail to close. Either radius may be 0 (a
    cone running to a point) but not both.
    """
    if height <= 0.0 or (r_lo <= 0.0 and r_hi <= 0.0):
        return None
    kind = "PartDesign::AdditiveCone" if additive \
        else "PartDesign::SubtractiveCone"
    try:
        op = doc.addObject(kind, name)
    except Exception as exc:                    # noqa: BLE001 - old FreeCAD
        App.Console.PrintError(
            f"[meshtofeatures] {label or name}: this FreeCAD has no "
            f"{kind} primitive ({exc}); the conical face is NOT rebuilt\n")
        return None
    body.addObject(op)
    op.Radius1 = float(r_lo)
    op.Radius2 = float(r_hi)
    op.Height = float(height)
    op.Placement = _point_placement(plan, float(x), float(y), float(z_lo))
    if label:
        op.Label = label
    _apply_refine(op)
    _rollback_if_broken(doc, body, op, deferred=deferred)
    return op


def _apply_refine(op):
    """Turn on Refine so booleans clean up residual sliver faces/edges
    (documented cause of thin 'sheet' walls rendered at curved cut
    boundaries in PartDesign). Guarded: not every feature exposes it."""
    try:
        if hasattr(op, "Refine"):
            op.Refine = True
    except Exception:
        pass


def _rollback_if_broken(doc, body, op, sketch=None, deferred=None) -> bool:
    """Recompute; if ``op`` failed, try to RECOVER it before giving up, and
    if it must be removed, say so LOUDLY. (Field lesson: the 8th counterbore
    shoulder terrace failed to compute on featuretype and was silently
    deleted, leaving a standing wall at that one bore for ten versions --
    a silent drop hides the root cause.) Recovery ladder:
      1. Refine off (Refine occasionally trips OCC on tricky shapes)
      2. jitter Length by a relative 1e-4 (sub-mesh-tolerance; dodges
         boolean tangency/coincidence flakes, the classic OCC failure)
      3. nudge the sketch in-plane by ~1e-5 (changes coincidence phase)
    Only if all rungs fail is the feature removed -- with an ERROR naming
    the feature, so the missing cut is visible instead of mysterious."""
    doc.recompute()

    def _is_broken():
        return "Invalid" in list(getattr(op, "State", [])) \
            or (hasattr(op, "isValid") and not op.isValid())

    if not _is_broken():
        return False

    # rung 1: Refine off (cheap, does not mutate geometry)
    if getattr(op, "Refine", False):
        try:
            op.Refine = False
            doc.recompute()
        except Exception:  # noqa: BLE001
            pass

    # DEFERRAL (cuts only), IMMEDIATELY and with the sketch PRISTINE: a
    # Pocket that fails here may compute cleanly at the END of the chain --
    # the failure can be an ORDER-dependent boolean flake tied to the
    # intermediate body state. Field-proven on featuretype: the 8th
    # counterbore shoulder terrace failed in place but computed 'Up-to-date'
    # when re-added at the end WITH ITS SKETCH UNTOUCHED; in-place jitters
    # never recovered it and only dirty the sketch the replay depends on, so
    # they are NOT attempted when deferral is available. Set subtraction
    # commutes ((X-A)-B == (X-B)-A), so deferring a subtractive feature
    # yields the IDENTICAL solid.
    if _is_broken() and deferred is not None and sketch is not None \
            and op.TypeId == "PartDesign::Pocket":
        spec = {"Label": op.Label}
        for prop in ("Type", "Length", "Length2", "Reversed", "Midplane"):
            if hasattr(op, prop):
                try:
                    val = getattr(op, prop)
                    spec[prop] = str(val) if prop == "Type" else val
                except Exception:  # noqa: BLE001
                    pass
        msg = (f"[meshtofeatures] feature '{op.Label}' failed in place; "
               f"DEFERRING it to the end of the chain\n")
        App.Console.PrintMessage(msg)     # message level: never filtered
        print(msg.strip())
        doc.removeObject(op.Name)
        doc.recompute()
        deferred.append((sketch, spec))
        return True

    # rung 2 (no-deferral contexts only): relative Length jitter
    if _is_broken() and hasattr(op, "Length") \
            and str(getattr(op, "Type", "")) == "Length":
        base = float(op.Length)
        for eps in (1e-4, -1e-4, 3e-4):
            try:
                op.Length = base * (1.0 + eps)
                doc.recompute()
            except Exception:  # noqa: BLE001
                continue
            if not _is_broken():
                break
        if _is_broken():
            try:
                op.Length = base
                doc.recompute()
            except Exception:  # noqa: BLE001
                pass

    # rung 3 (no-deferral contexts only): in-plane sketch nudge
    if _is_broken() and sketch is not None:
        try:
            pl = sketch.Placement
            b = pl.Base
            sketch.Placement = App.Placement(
                App.Vector(b.x + 1e-5, b.y + 1.3e-5, b.z), pl.Rotation)
            doc.recompute()
        except Exception:  # noqa: BLE001
            pass

    if not _is_broken():
        msg = (f"[meshtofeatures] feature '{op.Label}' initially failed but "
               f"was RECOVERED by an epsilon retry\n")
        App.Console.PrintMessage(msg)
        print(msg.strip())
        return False

    msg = (f"[meshtofeatures] feature '{op.Label}' FAILED to compute "
           f"(state={list(getattr(op, 'State', []))}) and was REMOVED -- "
           f"the rebuilt part is MISSING this cut\n")
    App.Console.PrintError(msg)
    App.Console.PrintMessage(msg)         # message level: never filtered
    print(msg.strip())
    doc.removeObject(op.Name)
    if sketch is not None:
        doc.removeObject(sketch.Name)
    doc.recompute()
    return True


def _lateral_pad(doc, body, plan, pad, k):
    """A lateral pad (design note 36): material protruding sideways off a
    wall. The sketch sits on a plane perpendicular to the pad axis --
    placement columns (u, v, axis) with the profile in (u, v) -- and the
    Pad extrudes ``length`` along the axis (the sketch normal). Direction
    is encoded in the placement (plane_origin anchors the min-axis end so
    the DEFAULT +normal extrusion is correct); no Reversed/Midplane, per
    the doctrine that direction booleans are unreliable on FreeCAD 1.1.
    """
    from .core.history import lateral_pad_world_frame
    origin, u, v, axis = lateral_pad_world_frame(plan, pad)
    m = App.Matrix(
        float(u[0]), float(v[0]), float(axis[0]), float(origin[0]),
        float(u[1]), float(v[1]), float(axis[1]), float(origin[1]),
        float(u[2]), float(v[2]), float(axis[2]), float(origin[2]),
        0.0, 0.0, 0.0, 1.0)
    s = doc.addObject("Sketcher::SketchObject", f"LateralPadProfile{k}")
    body.addObject(s)
    s.Placement = App.Placement(m)
    _add_geometry(s, pad.profile)
    s.Visibility = False
    op = doc.addObject("PartDesign::Pad", f"LateralPad{k}")
    body.addObject(op)
    op.Profile = s
    op.Length = float(pad.length)
    op.Label = pad.label or op.Name
    _apply_refine(op)
    _rollback_if_broken(doc, body, op, s)


def _geometric_blend(doc, body, blend, k, kind):
    """Dress a detected fillet whose sharp edge no parametric edge carries
    (issue #6's freeform-band class).

    The blend's sharp edge is reconstructed from the mesh's fits, so on a
    freeform band (a transition surface no analytic primitive captures)
    the parametric body can have NO edge at that position -- a
    PartDesign::Fillet dressup has nothing to attach to and would be
    skipped, losing the blend. Instead the same corner-tool cross-section
    the headless round-trip applies (:func:`blend_corner_profile`) is
    sketched on the plane through ``edge_start`` perpendicular to the
    edge direction and extruded along the detected span: a Pocket removes
    the convex sliver, a Pad fuses the concave quarter round. Direction
    is encoded in the placement (mirrored profile), not a Reversed
    boolean (field doctrine). The terminal deviation-correction pass
    reconciles any residual against the source mesh. Returns the created
    op (now the body tip), or None on any failure (caller reports).

    Chamfers stay out of scope: their convexity needs the headless solid
    probe, and the primary PartDesign::Chamfer path needs no edge-side
    decision.
    """
    import numpy as np
    d = np.asarray(blend.direction, dtype=float)
    nd = float(np.linalg.norm(d))
    na = np.asarray(blend.n_a, dtype=float)
    nb = np.asarray(blend.n_b, dtype=float)
    if nd < 1e-12:
        return None
    d = d / nd
    na = na - (na @ d) * d
    nna = float(np.linalg.norm(na))
    nb = nb - (nb @ d) * d
    nb = nb - (nb @ na) * na
    nnb = float(np.linalg.norm(nb))
    if nna < 1e-9 or nnb < 1e-9:
        return None
    na = na / nna
    nb = nb / nnb
    length = float(np.linalg.norm(np.asarray(blend.edge_end)
                                  - np.asarray(blend.edge_start)))
    if length <= 0.0:
        return None
    size = (float(getattr(blend, "radius", 0.0)) if kind == "fillet"
            else float(getattr(blend, "size", 0.0)))
    if size <= 0.0:
        return None
    convex = bool(getattr(blend, "convex", True))
    # bury only the concave fuse (legs overlap into material so OCC can
    # fuse); burying the convex CUTTER would shave the two faces.
    bury = 0.02 * size if not convex else 0.0
    profile = blend_corner_profile(size, convex, bury=bury) \
        if kind == "fillet" else chamfer_corner_profile(size, convex,
                                                       bury=bury)
    if not profile:
        return None
    # A sketch plane must be right-handed: columns (x, y, z) with x cross
    # y = z. For a Pad the sketch normal is the extrusion direction, so
    # (na, nb, d) works as-is. A Pocket extrudes OPPOSITE the sketch
    # normal, so the plane is flipped to (na, -nb, -d) -- which mirrors
    # the second in-plane axis -- and the profile's second coordinate is
    # mirrored with it, mapping to the identical 3D region.
    v = -nb if convex else nb
    z = -d if convex else d
    m = App.Matrix(
        float(na[0]), float(v[0]), float(z[0]), float(blend.edge_start[0]),
        float(na[1]), float(v[1]), float(z[1]), float(blend.edge_start[1]),
        float(na[2]), float(v[2]), float(z[2]), float(blend.edge_start[2]),
        0.0, 0.0, 0.0, 1.0)
    sk = doc.addObject("Sketcher::SketchObject", f"BlendProfile{k}")
    body.addObject(sk)
    sk.Placement = App.Placement(m)
    if convex:
        profile = _mirror_y(profile)          # v axis is -nb: mirror with it
    # drop zero-length segments (a bury of 0 emits them; they are polygon
    # bookkeeping, not geometry)
    for p in list(profile):
        if isinstance(p, SketchLine) and float(np.linalg.norm(
                np.asarray(p.end) - np.asarray(p.start))) < 1e-9:
            profile.remove(p)
    _add_geometry(sk, profile)
    sk.Visibility = False
    op = doc.addObject("PartDesign::Pocket" if convex
                       else "PartDesign::Pad", f"GeometricBlend{k}")
    body.addObject(op)
    op.Profile = sk
    op.Length = length
    kind_name = "fillet" if kind == "fillet" else "chamfer"
    op.Label = (f"{kind_name} (geometric) {blend.label or op.Name}").strip()
    _apply_refine(op)
    broken = _rollback_if_broken(doc, body, op, sk)
    if broken:
        return None
    msg = (f"[meshtofeatures] {kind_name} '{blend.label or op.Name}': no "
           f"parametric edge at the detected position; dressed "
           f"geometrically at the mesh-fit edge (freeform-band class)\n")
    App.Console.PrintMessage(msg)
    print(msg.strip())
    return op


def _shape_from_mesh(m):
    """A watertight trimesh -> Part solid, one planar face per triangle.

    Used for deviation-correction patches and the source-mesh intersection.
    Built via makePolygon/makeShell (the documented API): ``Part.Shape``
    fed a raw facet list crashes FreeCAD 1.1 outright. Raises on failure
    so the caller can degrade.
    """
    verts = m.vertices
    faces = []
    for tri in m.faces:
        a = App.Vector(float(verts[tri[0]][0]), float(verts[tri[0]][1]),
                       float(verts[tri[0]][2]))
        b = App.Vector(float(verts[tri[1]][0]), float(verts[tri[1]][1]),
                       float(verts[tri[1]][2]))
        c = App.Vector(float(verts[tri[2]][0]), float(verts[tri[2]][1]),
                       float(verts[tri[2]][2]))
        if a.isEqual(b, 1e-9) or b.isEqual(c, 1e-9) or a.isEqual(c, 1e-9):
            continue                       # degenerate facet (boolean debris)
        faces.append(Part.Face(Part.makePolygon([a, b, c, a])))
    sh = Part.makeShell(faces)
    try:
        out = Part.Solid(sh)
    except Exception:                                      # noqa: BLE001
        out = sh
    if not out.isValid():
        # boolean output carries micro-slivers that fail OCC's BRep check
        # without affecting the geometry; fix() heals them (volume-
        # preserving, verified on the issue-#5 bracket)
        try:
            out.fix(0.0, 0.1, 0.1)
        except Exception:                                  # noqa: BLE001
            pass
    return out


def _apply_corrections(doc, body, plan: BuildPlan, name: str):
    """Terminal deviation correction (hybrid-modelling fallback).

    Computed headlessly with manifold booleans (intersect the parametric
    rebuild with the source mesh, union the UNDER patches -- proven exact
    where OCC's booleans degenerate on faceted input), then installed as
    a single ``Part::Feature`` with the mesh-accurate shape. The body
    stays the editable parametric history and is never modified here.
    Degrades to a warning on any failure.
    """
    mesh = getattr(plan, "source_mesh", None)
    corrs = getattr(plan, "corrections", None) or []
    if not corrs or mesh is None:
        return None
    try:
        from .core.solidify import apply_corrections, plan_to_mesh
        solid = plan_to_mesh(plan)
        if solid is None:
            App.Console.PrintWarning(
                "[meshtofeatures] deviation correction skipped: plan not "
                "headlessly executable\n")
            return None
        corrected = apply_corrections(solid, mesh, corrs)
        shape = _shape_from_mesh(corrected)
        feat = doc.addObject("Part::Feature", name + "_Corrected")
        feat.Shape = shape
        feat.Label = name + " (corrected)"
        App.Console.PrintMessage(
            f"[meshtofeatures] deviation correction applied; "
            f"corrected volume {shape.Volume:.1f} "
            f"(mesh {mesh.volume:.1f})\n")
        return feat
    except Exception as exc:                               # noqa: BLE001
        App.Console.PrintWarning(
            f"[meshtofeatures] deviation correction failed: {exc}\n")
        return None


def build_body(doc, plan: BuildPlan, name: str = "Rebuilt"):
    """Create a PartDesign Body implementing ``plan``; returns the body."""
    body = doc.addObject("PartDesign::Body", name)
    deferred = []            # failed CUTS to retry at the end of the chain

    base_entries = list(plan.base.profile)
    for hp in getattr(plan.base, "hole_profiles", []):
        base_entries.extend(hp)     # frame/ring: inner wires punch the base
    sk = _circle_sketch(doc, body, plan, 0.0, base_entries, "BaseProfile")
    pad = doc.addObject("PartDesign::Pad", "BasePad")
    body.addObject(pad)
    pad.Profile = sk
    pad.Length = float(plan.base.length)
    _apply_refine(pad)
    sk.Visibility = False
    # PartDesign::Hole validates its base feature's shape EAGERLY at
    # creation (unlike Pocket): every chained feature must be recomputed
    # before the next one is added, or Hole fails with "Base feature's
    # TopoShape is invalid" (field-observed on FreeCAD 1.1.1)
    doc.recompute()

    L = float(plan.base.length)
    # Gusset webs live INSIDE a recess, so they must be added after the
    # pocket that carves that recess -- split them out of the main pad pass
    # and build them once the pockets are done.
    gusset_pads = []
    for k, p in enumerate(plan.pads):
        if getattr(p, "axis", None) is not None:
            if p.label == "Gusset web":
                gusset_pads.append((k, p))
                continue
            _lateral_pad(doc, body, plan, p, k)
            continue
        top = getattr(p, "from_top", True)
        s = _circle_sketch(doc, body, plan, L if top else 0.0, p.profile,
                           f"PadProfile{k}", flip=not top)
        op = doc.addObject("PartDesign::Pad", f"Pad{k}")
        body.addObject(op)
        op.Profile = s
        op.Length = float(p.length)
        op.Label = p.label or op.Name
        _apply_refine(op)
        s.Visibility = False
        _rollback_if_broken(doc, body, op, s)

    for k, p in enumerate(plan.pockets):
        top = getattr(p, "from_top", True)
        use = list(getattr(p, "mouth_profile", None) or p.profile)
        entries = use.copy()
        for hp in getattr(p, "hole_profiles", []):
            entries.extend(hp)
        try:
            s = _circle_sketch(doc, body, plan, L if top else 0.0, entries,
                               f"PocketProfile{k}", flip=not top)
        except Exception:                     # degenerate mouth profile fallback
            use, entries = list(p.profile), list(p.profile)
            for hp in getattr(p, "hole_profiles", []):
                entries.extend(hp)
            s = _circle_sketch(doc, body, plan, L if top else 0.0, entries,
                               f"PocketProfile{k}", flip=not top)
        op = doc.addObject("PartDesign::Pocket", f"Pocket{k}")
        body.addObject(op)
        op.Profile = s
        if getattr(p, "through", False):
            op.Type = "ThroughAll"
        else:
            op.Length = float(p.depth)
        op.Label = p.label or op.Name
        _apply_refine(op)
        s.Visibility = False
        _rollback_if_broken(doc, body, op, s, deferred=deferred)

    # gusset webs, now that the recess pockets exist to receive them
    for k, p in gusset_pads:
        _lateral_pad(doc, body, plan, p, k)

    for k, h in enumerate(plan.holes):
        import numpy as np
        circles = [SketchCircle(center=np.array(pos), radius=h.diameter / 2)
                   for pos in h.positions]
        top = getattr(h, "from_top", True)
        # A from-top bore that opens BELOW the global top (a counterbore on a
        # base plate under a raised deck) is cut from the GLOBAL TOP straight
        # down through its floor: drill ThroughAll from the top, counterbore
        # depth extended by the column height (L - surface_z). Placing the
        # sketch on the opening face (surface_z) instead makes PartDesign
        # extrude the bore UPWARD as a solid column (field-confirmed), so the
        # sketch stays at the top. Everything is a Pocket, so no tower.
        surf_z = getattr(h, "surface_z", None)
        _tol_z = 1e-3 * float(L) if L else 1e-6
        opens_below_top = surf_z is not None and float(surf_z) < L - _tol_z
        if opens_below_top and top:
            z_at = L
            cb_extra = L - float(surf_z)        # column height above opening
        elif surf_z is not None:
            z_at = float(surf_z)
            cb_extra = 0.0
        else:
            z_at = L if top else 0.0
            cb_extra = 0.0
        # A from-bottom THROUGH bore opening below the top: a flipped pocket
        # down from the opening face under-cuts (only the countersink depth
        # lands), so drill UP from the bottom face instead. Drilling up means
        # the sketch sits on the bottom face with its normal pointing DOWN
        # (flip=True): a PartDesign Pocket cuts opposite its sketch normal,
        # so flip=True extrudes the cut up into the part while flip=False
        # extrudes down into air and removes nothing (field-observed: the
        # issue-#5 bracket's bottom countersink stayed solid). The
        # countersink cone is placed at the opening face below.
        drill_z_at, drill_flip = z_at, not top
        if h.through and not top and surf_z is not None:
            drill_z_at, drill_flip = 0.0, True
        s = _circle_sketch(doc, body, plan, drill_z_at, circles,
                           f"HoleProfile{k}", flip=drill_flip)
        s.Visibility = False
        op = None
        try:
            if not top or opens_below_top:
                # PartDesign::Hole misbehaves whenever the opening face is not
                # the outermost top face (flipped bottom planes: holes as
                # threaded towers; below-top planes: bores as solid columns).
                # The pocket path -- validated by the smoke ring-step -- only
                # cuts material.
                raise RuntimeError("bore not on outer top face: use pocket "
                                   "path")
            # semantic PartDesign Hole: one editable feature carrying
            # diameter, depth mode, and counterbore parameters
            op = doc.addObject("PartDesign::Hole", f"Hole{k}")
            body.addObject(op)
            op.Profile = s
            props = hole_op_properties(h)
            props["Reversed"] = False   # side is encoded in the placement
            for name, value in props.items():
                if hasattr(op, name):
                    setattr(op, name, value)
            op.Label = h.label or op.Name
            _apply_refine(op)
            doc.recompute()
            if not op.isValid():
                raise RuntimeError("Hole feature did not compute validly")
        except Exception as exc:  # noqa: BLE001 - fall back to pocket cuts
            if "bottom-side" not in str(exc) and "outer top face" \
                    not in str(exc):
                App.Console.PrintWarning(
                    f"[meshtofeatures] Hole feature failed ({exc}); "
                    f"using pockets\n")
            # Remove ONLY a Hole object this iteration actually created.
            # (The old `'op' in dir()` test saw `op` leaked from the pocket
            # loop above and deleted the last terrace pocket whenever a
            # bottom-side hole took this fallback before its Hole existed --
            # field: issue #5's bracket lost its recess cut to its own
            # bottom-face countersink.)
            if op is not None:
                doc.removeObject(op.Name)
            op = doc.addObject("PartDesign::Pocket", f"Hole{k}")
            body.addObject(op)
            op.Profile = s
            if h.through and top:
                op.Type = "ThroughAll"
            elif h.through:
                # A from-bottom through-hole is a flipped sketch. Cut with
                # a Length through the base (like the from-bottom STEPS,
                # which are Length pockets and cut correctly) rather than
                # ThroughAll, whose direction handling on a flipped sketch
                # is less predictable. (The counterbored holes' real failure
                # was upstream -- the base pre-punched them; see note 37.)
                op.Length = float(L) * 1.1
            else:
                # blind bore: from z_at down by its depth, plus the column
                # height above the opening face when cut from the global top
                op.Length = float(h.depth) + cb_extra
            op.Label = h.label or op.Name
            _apply_refine(op)
            _rollback_if_broken(doc, body, op, s, deferred=deferred)
            if h.counterbore_diameter:
                # For a BELOW-TOP bore, grow the counterbore circle by the
                # SAME 1.5*tol buffer the terraces get. The reconstructed
                # island loop and the counterbore both snap to the identical
                # radius (featuretype: exactly 0.21875), so a circle cut at
                # that radius is RADIALLY COINCIDENT with the island and OCC
                # leaves an epsilon-thin standing band (the field-observed
                # striped half-cylinder) wherever the buffered shoulder
                # terrace -- which normally cleans that band -- fails to
                # compute. With the buffer, the circle alone consumes the
                # band, independent of the flaky terrace, and all recesses
                # get the SAME effective radius the buffered terraces already
                # give the others (consistency, within mesh tolerance, per
                # the v0.15.25 buffer doctrine). Top-face bores keep the
                # exact diameter (no island/terrace coincidence there).
                cb_r = float(h.counterbore_diameter) / 2.0
                if opens_below_top:
                    diag = float(body.Shape.BoundBox.DiagonalLength) or L
                    cb_r += 1.5e-3 * diag
                cbs = [SketchCircle(center=np.array(pos), radius=cb_r)
                       for pos in h.positions]
                s2 = _circle_sketch(doc, body, plan, z_at, cbs,
                                    f"CBoreProfile{k}", flip=not top)
                op2 = doc.addObject("PartDesign::Pocket", f"Counterbore{k}")
                body.addObject(op2)
                op2.Profile = s2
                # counterbore depth + column height above the opening face,
                # so cutting from the global top reaches the counterbore floor
                op2.Length = float(h.counterbore_depth) + cb_extra
                _apply_refine(op2)
                s2.Visibility = False
                _rollback_if_broken(doc, body, op2, s2, deferred=deferred)
            if h.countersink_diameter:
                # The pocket path cuts cylinders only, so the CONICAL entry
                # of a countersunk or counterdrilled hole used to be dropped
                # here (the PartDesign::Hole path above does it natively via
                # HoleCutType). Cut it as a placed SubtractiveCone: the mouth
                # of the taper is the opening face for a plain countersink,
                # and the BORE FLOOR when a counterbore sits above it.
                dr = float(h.diameter) / 2.0
                cr = float(h.countersink_diameter) / 2.0
                ha = math.radians(float(h.countersink_angle or 90.0) / 2.0)
                run = (cr - dr) / max(math.tan(ha), 1e-9)
                cbd = float(h.counterbore_depth) if h.counterbore_diameter \
                    else 0.0
                face_z = float(surf_z) if surf_z is not None \
                    else (L if top else 0.0)
                if top:
                    mouth_z = face_z - cbd
                    z_lo, r_lo, r_hi = mouth_z - run, dr, cr
                else:
                    # from-bottom: the taper still widens AT the opening face
                    # (the recess floor) and narrows DOWN toward the drill, so
                    # the cone sits BELOW the mouth exactly like the top case.
                    # (The old z_lo=mouth_z, r_lo=cr form put the cone ABOVE
                    # the floor, in the already-empty recess, cutting nothing
                    # -- field-observed: no countersink on the bottom hole.)
                    mouth_z = face_z + cbd
                    z_lo, r_lo, r_hi = mouth_z - run, dr, cr
                for pos in h.positions:
                    _add_cone(doc, body, plan, f"Countersink{k}",
                              pos[0], pos[1], z_lo, r_lo, r_hi, run,
                              label=f"Countersink{k}", deferred=deferred)
            doc.recompute()

    doc.recompute()

    # ---- conical pockets: placed Subtractive/AdditiveCone primitives -------
    for k, c in enumerate(getattr(plan, "cones", [])):
        try:
            diag = float(body.Shape.BoundBox.DiagonalLength) or float(L)
        except Exception:                       # noqa: BLE001
            diag = float(L)
        pad = 1.5e-3 * diag                     # same buffer doctrine as the
        #  terraces: never end a cut exactly ON the face it opens through
        slope = (c.r_mouth - c.r_far) / max(c.depth, 1e-12)
        face_z = c.surface_z if c.surface_z is not None \
            else (L if c.from_top else 0.0)
        far_pad = pad if c.through else 0.0
        r_end = max(c.r_far - slope * far_pad, 0.0)
        height = c.depth + far_pad + pad
        if c.from_top:
            z_lo = float(face_z) - c.depth - far_pad
            r_lo, r_hi = r_end, c.r_mouth + slope * pad
        else:
            z_lo = float(face_z) - pad
            r_lo, r_hi = c.r_mouth + slope * pad, r_end
        for pos in c.positions:
            _add_cone(doc, body, plan, f"ConePocket{k}", pos[0], pos[1],
                      z_lo, r_lo, r_hi, height, additive=c.additive,
                      label=c.label or f"ConePocket{k}", deferred=deferred)
    doc.recompute()

    # ---- cross-axis holes: through = midplane pocket, blind = Length -------
    import numpy as np
    for k, ch in enumerate(getattr(plan, "cross_holes", [])):
        axis = np.asarray(ch.axis, dtype=float)
        blind = not getattr(ch, "through", True)
        # blind holes sketch on the ENTRY wall with the OUTWARD normal, so a
        # normal (into-material) Length pocket drills inward by the depth;
        # through holes sketch on a midplane normal to the axis.
        sn = (-np.asarray(ch.entry_direction, dtype=float)
              if blind else axis)
        u, v = _axis_frame(sn)
        anchor0 = np.asarray(ch.positions3d[0], dtype=float)
        m = App.Matrix(
            float(u[0]), float(v[0]), float(sn[0]), float(anchor0[0]),
            float(u[1]), float(v[1]), float(sn[1]), float(anchor0[1]),
            float(u[2]), float(v[2]), float(sn[2]), float(anchor0[2]),
            0.0, 0.0, 0.0, 1.0)
        s = doc.addObject("Sketcher::SketchObject", f"CrossHoleProfile{k}")
        body.addObject(s)
        s.Placement = App.Placement(m)
        circles = []
        for pos in ch.positions3d:
            rel = np.asarray(pos, dtype=float) - anchor0
            circles.append(SketchCircle(
                center=np.array([float(rel @ u), float(rel @ v)]),
                radius=ch.diameter / 2))
        _add_geometry(s, circles)
        s.Visibility = False
        op = doc.addObject("PartDesign::Pocket", f"CrossHole{k}")
        body.addObject(op)
        op.Profile = s
        if blind:
            # a depth-limited bore from the wall: one-sided Length cut
            op.Type = "Length"
            op.Length = float(ch.depth)
        else:
            # Cut the through-hole SYMMETRICALLY by an explicit large length
            # each way (TwoLengths) instead of ThroughAll + Midplane/SideType.
            # The two-sided handling can misfire on FreeCAD 1.1 (the SideType
            # enum varies) and cut only ONE direction, leaving a half-cylinder
            # standing at the far end (field-observed: "the hole is not cut
            # through"). A length of the body diagonal each way is guaranteed
            # to exit both faces.
            reach = float(body.Shape.BoundBox.DiagonalLength) or float(L)
            two_sided = False
            try:
                types = list(op.getEnumerationsOfProperty("Type") or [])
                if "TwoLengths" in types and hasattr(op, "Length2"):
                    op.Type = "TwoLengths"
                    op.Length = reach
                    op.Length2 = reach
                    two_sided = True
            except Exception:  # noqa: BLE001
                pass
            if not two_sided:
                op.Type = "ThroughAll"
                # FreeCAD 1.1 deprecates Midplane in favour of SideType; pick
                # the symmetric/two-sided enum entry by introspection (exact
                # strings vary across builds), falling back to Midplane on
                # older builds
                side_set = False
                if hasattr(op, "SideType"):
                    try:
                        options = op.getEnumerationsOfProperty("SideType") or []
                        for want in ("symmetric", "two"):
                            mm = [o for o in options if want in o.lower()]
                            if mm:
                                op.SideType = mm[0]
                                side_set = True
                                break
                    except Exception:  # noqa: BLE001
                        pass
                if not side_set:
                    op.Midplane = True
        op.Label = ch.label or op.Name
        _apply_refine(op)
        _rollback_if_broken(doc, body, op, s, deferred=deferred)

    # ---- horizontal fillets: geometric edge matching on the sharp body ----
    # No topological names are trusted: for each FilletOp, straight edges
    # of the current tip feature whose endpoints lie on the computed sharp
    # edge segment (pure matcher, tested) are collected and dressed up.
    if getattr(plan, "fillets", None) or getattr(plan, "chamfers", None):
        import numpy as np
        diag = float(np.linalg.norm(
            np.array(body.Shape.BoundBox.DiagonalLength)))
        # The detected sharp edge is reconstructed from the mesh's blend
        # cylinder + plane fits, so it can sit a few tenths of a mm off the
        # rebuilt body's true edge (coarse-tessellation fit error, field:
        # issue #5's bracket rim fillets all missed at a 0.09 mm tol). Scale
        # the match tolerance with the blend size so the matcher tolerates
        # that fit error without accepting a parallel edge on the wrong line.
        blend_sizes = [float(fo.radius) for fo in getattr(plan, "fillets", [])]
        blend_sizes += [float(co.size) for co in getattr(plan, "chamfers", [])]
        tol = max(0.5 * max(blend_sizes), 1e-3 * diag, 1e-6)
        prev = body.Tip
        dressups = [("PartDesign::Fillet", "Radius", fo.radius, fo)
                    for fo in getattr(plan, "fillets", [])]
        dressups += [("PartDesign::Chamfer", "Size", co.size, co)
                     for co in getattr(plan, "chamfers", [])]
        for k, (type_id, prop, value, fo) in enumerate(dressups):
            names = []
            for idx, edge in enumerate(prev.Shape.Edges):
                try:
                    if edge.Curve.TypeId != "Part::GeomLine":
                        continue
                    p0 = np.array([edge.Vertexes[0].X, edge.Vertexes[0].Y,
                                   edge.Vertexes[0].Z])
                    p1 = np.array([edge.Vertexes[-1].X, edge.Vertexes[-1].Y,
                                   edge.Vertexes[-1].Z])
                except Exception:  # noqa: BLE001 - odd edge types: skip
                    continue
                if fillet_edge_matches(fo, p0, p1, tol):
                    names.append(f"Edge{idx + 1}")
            if not names:
                if type_id == "PartDesign::Fillet":
                    # No parametric edge at the detected sharp edge (the
                    # freeform-band class, issue #6): dress it geometrically
                    # at the mesh-fit position instead of dropping it.
                    fb = _geometric_blend(doc, body, fo, k, "fillet")
                    if fb is not None:
                        prev = fb
                        continue
                App.Console.PrintWarning(
                    f"[meshtofeatures] no body edge matched "
                    f"{type_id.rsplit('::', 1)[-1].lower()} "
                    f"'{fo.label}'; skipped\n")
                continue
            op = doc.addObject(type_id, f"Dressup{k}")
            body.addObject(op)
            op.Base = (prev, names)
            setattr(op, prop, float(value))
            op.Label = fo.label or op.Name
            doc.recompute()
            broken = "Invalid" in list(getattr(op, "State", [])) \
                or (hasattr(op, "isValid") and not op.isValid())
            if broken:
                App.Console.PrintWarning(
                    f"[meshtofeatures] fillet '{fo.label}' failed to compute; "
                    f"removing\n")
                doc.removeObject(op.Name)
                doc.recompute()
            else:
                prev = op

    # ---- deferred cuts: retry failed pockets at the END of the chain ------
    # A cut that failed in place often computes cleanly here (order-dependent
    # boolean flake; field-proven). Subtraction commutes, so the final solid
    # is identical to the planned one. The re-add mirrors the field-verified
    # probe retry EXACTLY: pristine sketch, Profile + Length only (Type set
    # only when non-default), no Refine, label applied only after validation.
    for sketch, spec in deferred:
        msg = (f"[meshtofeatures] retrying deferred cut "
               f"'{spec.get('Label')}' at the end of the chain\n")
        App.Console.PrintMessage(msg)
        print(msg.strip())
        op = doc.addObject("PartDesign::Pocket", "Deferred" + sketch.Name)
        body.addObject(op)
        op.Profile = sketch
        try:
            if spec.get("Type") and spec["Type"] != "Length":
                op.Type = spec["Type"]
        except Exception:  # noqa: BLE001
            pass
        for prop in ("Length", "Length2"):
            if spec.get(prop) is not None:
                try:
                    setattr(op, prop, float(spec[prop]))
                except Exception:  # noqa: BLE001
                    pass
        for prop in ("Reversed", "Midplane"):
            if spec.get(prop):
                try:
                    setattr(op, prop, bool(spec[prop]))
                except Exception:  # noqa: BLE001
                    pass
        sketch.Visibility = False
        if not _rollback_if_broken(doc, body, op, sketch):
            try:
                op.Label = spec.get("Label") or op.Name
            except Exception:  # noqa: BLE001
                pass
            msg = (f"[meshtofeatures] deferred feature "
                   f"'{spec.get('Label')}' RECOVERED at the end of the "
                   f"chain\n")
            App.Console.PrintMessage(msg)
            print(msg.strip())

    # ---- deviation correction: mesh-accurate terminal shape ---------------
    _apply_corrections(doc, body, plan, name)

    doc.recompute()
    return body
