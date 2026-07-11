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

from meshtofeatures.history import (BuildPlan, SketchArc, SketchCircle,
                               SketchLine, fillet_edge_matches,
                               hole_op_properties)
from meshtofeatures.fitting import _axis_frame


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
    from meshtofeatures.history import SketchArc, SketchCircle, SketchLine
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
    from meshtofeatures.history import lateral_pad_world_frame
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
    for k, p in enumerate(plan.pads):
        if getattr(p, "axis", None) is not None:
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
        entries = list(p.profile)
        for hp in getattr(p, "hole_profiles", []):
            entries.extend(hp)      # nested wires become holes in the cut
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
        s = _circle_sketch(doc, body, plan, z_at, circles, f"HoleProfile{k}",
                           flip=not top)
        s.Visibility = False
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
            doc.removeObject(op.Name) if 'op' in dir() else None
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
            doc.recompute()

    doc.recompute()

    # ---- cross-axis through holes: midplane through-all pockets -----------
    import numpy as np
    for k, ch in enumerate(getattr(plan, "cross_holes", [])):
        axis = np.asarray(ch.axis, dtype=float)
        u, v = _axis_frame(axis)
        anchor0 = np.asarray(ch.positions3d[0], dtype=float)
        m = App.Matrix(
            float(u[0]), float(v[0]), float(axis[0]), float(anchor0[0]),
            float(u[1]), float(v[1]), float(axis[1]), float(anchor0[1]),
            float(u[2]), float(v[2]), float(axis[2]), float(anchor0[2]),
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
        # Cut the through-hole SYMMETRICALLY by an explicit large length each
        # way (TwoLengths) instead of ThroughAll + Midplane/SideType. The two-
        # sided handling can misfire on FreeCAD 1.1 (the SideType enum varies)
        # and cut only ONE direction, leaving a half-cylinder standing at the
        # far end (field-observed: "the hole is not cut through"). A length of
        # the body diagonal each way is guaranteed to exit both faces.
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
            # FreeCAD 1.1 deprecates Midplane in favour of SideType; pick the
            # symmetric/two-sided enum entry by introspection (exact strings
            # vary across builds), falling back to Midplane on older builds
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
        tol = max(1e-3 * diag, 1e-6)
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
                App.Console.PrintWarning(
                    f"[meshtofeatures] no body edge matched fillet "
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


    doc.recompute()
    return body
