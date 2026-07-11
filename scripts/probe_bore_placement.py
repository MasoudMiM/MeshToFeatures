# SPDX-License-Identifier: LGPL-2.1-or-later
# ============================================================================
#  MeshToFeatures bore-placement probe  (NO pytest required)
# ----------------------------------------------------------------------------
#  WHAT IT DOES
#    1. prints the LOADED addon version (catches a stale module cache)
#    2. plans your STL and prints every hole's surface_z + which build path
#       it takes (POCKET = cut only, can't tower;  HOLE = PartDesign::Hole)
#    3. BUILDS the body in FreeCAD and dumps every feature with its running
#       height, so the feature that adds the columns is obvious
#
#  HOW TO RUN  (two edits, then one line in the FreeCAD Python console)
#    a) Save this file somewhere, e.g.
#         /home/masoud/Documents/projects/mtf_probe.py
#    b) Edit the two ALL-CAPS paths just below (ADDON_DIR and STL_PATH).
#    c) In FreeCAD:  View > Panels > Python console, then paste EXACTLY:
#         exec(open('/home/masoud/Documents/projects/mtf_probe.py').read())
#    d) Copy the WHOLE printed block back to me.
# ============================================================================

import sys

# --- EDIT THESE TWO PATHS ----------------------------------------------------
ADDON_DIR = '/home/masoud/.local/share/FreeCAD/v1-1/Mod/MeshToFeatures'
STL_PATH  = '/home/masoud/Documents/projects/featuretype.STL'
# -----------------------------------------------------------------------------

if ADDON_DIR not in sys.path:
    sys.path.insert(0, ADDON_DIR)

print('=' * 68)

# 1) loaded version -----------------------------------------------------------
import meshtofeatures
print('LOADED addon version :', getattr(meshtofeatures, '__version__', '???'))
print('LOADED from          :', meshtofeatures.__file__)

# 2) plan + routing -----------------------------------------------------------
import trimesh
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report
from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.history import plan_history

mesh = trimesh.load(STL_PATH, force='mesh')
report = snap_report(reconstruct(mesh)).report
patches = plan_patches(report)
feats = detect_features(report, patches)
plan = plan_history(report, feats, detect_patterns(feats), patches)

L = plan.base.length
mesh_zmax = float(mesh.bounds[1][2])
n_terr = sum('Terrace' in (p.label or '') for p in plan.pockets)
print('-' * 68)
print('base_len L = %.4f   mesh z-max = %.4f' % (L, mesh_zmax))
print('holes=%d  pads=%d  pockets=%d (terraces=%d)'
      % (len(plan.holes), len(plan.pads), len(plan.pockets), n_terr))

def _routes_pocket(h):
    sz = getattr(h, 'surface_z', None)
    below = sz is not None and float(sz) < L - 1e-3 * (L or 1.0)
    return (not getattr(h, 'from_top', True)) or below

print('-' * 68)
for h in plan.holes:
    sz = getattr(h, 'surface_z', None)
    print('HOLE %-40s\n     from_top=%s  surface_z=%s  n_pos=%d  -> %s'
          % ((h.label or '')[:40], getattr(h, 'from_top', True),
             ('%.4f' % sz) if sz is not None else 'None',
             len(h.positions),
             'POCKET (cut, no tower)' if _routes_pocket(h)
             else 'PartDesign::Hole'))
for p in plan.pads:
    print('PAD  %-40s  from_top=%s  length=%s  axis=%s'
          % ((getattr(p, 'label', '') or '')[:40],
             getattr(p, 'from_top', None), getattr(p, 'length', None),
             getattr(p, 'axis', None) is not None))

# 3) build the body and dump features ----------------------------------------
print('-' * 68)
try:
    import FreeCAD as App
    try:
        from freecad.meshtofeatures_wb import build
    except Exception:
        from meshtofeatures_wb import build  # alternate load path
    doc = App.newDocument('mtf_probe')
    body = build.build_body(doc, plan, 'Probe')
    doc.recompute()
    print('BUILT FEATURES  (name | type | running z-max):')
    for o in doc.Objects:
        try:
            zmax = '%.3f' % o.Shape.BoundBox.ZMax
        except Exception:
            zmax = '   -'
        print('  %-18s %-28s zmax=%s' % (o.Name, o.TypeId, zmax))
    bz = body.Shape.BoundBox.ZMax
    print('-' * 68)
    print('FINAL body z-max = %.4f   (mesh z-max = %.4f)' % (bz, mesh_zmax))

    # ---- MISSING-FEATURE CHECK + LIVE RETRY -------------------------------
    # The executor's rollback guard removes features that fail to compute.
    # Detect any planned pocket whose op is absent, then RE-ADD it at the end
    # of the chain and report FreeCAD's own verdict: if it computes now, the
    # failure was an order-dependent boolean flake (and the body in THIS
    # document becomes the corrected part); if it fails again, the profile
    # trips OCC intrinsically at that site.
    try:
        print()
        print('MISSING-FEATURE CHECK:')
        missing = []
        for k, pk in enumerate(plan.pockets):
            if doc.getObject('Pocket%d' % k) is None:
                pts = [pr.start for pr in pk.profile]
                cu = sum(float(q[0]) for q in pts) / len(pts)
                cv = sum(float(q[1]) for q in pts) / len(pts)
                missing.append((k, pk))
                print('  MISSING Pocket%d  label=%r depth=%.4f '
                      'centroid=(%.2f, %.2f) -- dropped by rollback guard'
                      % (k, pk.label, pk.depth, cu, cv))
        if not missing:
            print('  all %d planned pockets present' % len(plan.pockets))
        for k, pk in missing:
            s = doc.getObject('PocketProfile%d' % k)
            if s is None:
                print('  Pocket%d: its sketch is also gone; cannot retry' % k)
                continue
            print('  RETRYING Pocket%d at the end of the chain...' % k)
            op = doc.addObject('PartDesign::Pocket', 'RetryPocket%d' % k)
            body.addObject(op)
            op.Profile = s
            try:
                op.Length = float(pk.depth)
            except Exception:               # noqa: BLE001
                pass
            s.Visibility = False
            doc.recompute()
            broken = 'Invalid' in list(getattr(op, 'State', [])) \
                or (hasattr(op, 'isValid') and not op.isValid())
            print('    retry state=%s  valid=%s'
                  % (list(getattr(op, 'State', [])), not broken))
            if broken:
                print('    -> STILL FAILS when re-added: intrinsic OCC '
                      'failure for this profile at this site')
                doc.removeObject(op.Name)
                doc.recompute()
            else:
                print('    -> SUCCEEDS when re-added after the full chain: '
                      'order-dependent boolean flake.')
                print('       Feature kept -- the body in THIS document is '
                      'now corrected; check the wall visually!')
    except Exception as _e:                 # noqa: BLE001
        print('MISSING-FEATURE CHECK failed: %r' % (_e,))
    # DEFINITIVE column check using FreeCAD's OWN solid test (no trimesh /
    # rtree): a from-top bore that opens below the top must be HOLLOW above
    # its opening face; if the body is solid there, a column stands. Reports
    # each offending site so a single stubborn bore is identified by name.
    import numpy as np
    fo = np.asarray(plan.frame_origin, float)
    fx = np.asarray(plan.frame_x, float)
    fy = np.asarray(plan.frame_y, float)
    fz = np.asarray(plan.frame_z, float)
    bad = []
    n_sites = 0
    for h in plan.holes:
        sz = h.surface_z if h.surface_z is not None else L
        if not (getattr(h, 'from_top', True) and sz < L - 1e-3 * (L or 1.0)):
            continue                            # only below-top from-top bores
        # sample the CENTER and an EDGE RING (a partial/half wall shows only
        # at the rim, not the centre)
        rad = 0.5 * (getattr(h, 'counterbore_diameter', None)
                     or h.diameter) * 0.85
        offs = [(0.0, 0.0)] + [(rad * np.cos(a), rad * np.sin(a))
                               for a in np.linspace(0, 2 * np.pi, 8,
                                                    endpoint=False)]
        for (u, v) in h.positions:
            n_sites += 1
            hit = 0
            zc = 0.5 * (float(sz) + L)
            for (du, dv) in offs:
                w = fo + (u + du) * fx + (v + dv) * fy + zc * fz
                try:
                    if body.Shape.isInside(
                            App.Vector(float(w[0]), float(w[1]), float(w[2])),
                            1e-6, True):
                        hit += 1
                except Exception:               # noqa: BLE001
                    pass
            if hit:
                bad.append((round(float(u), 3), round(float(v), 3),
                            '%d/9pts' % hit))
    print('COLUMN CHECK (FreeCAD solid test): %d of %d below-top bore sites '
          'still SOLID above their opening face' % (len(bad), n_sites))
    if bad:
        print('  offending (u,v) sites:', bad)
    print('==> COLUMNS present at %d site(s)' % len(bad) if bad
          else ('==> TOWERS above top' if bz > mesh_zmax + 1e-3 * (L or 1.0)
                else '==> CLEAN (no columns, no towers)'))

    # ---- CURVED-TERRACE WALL CHECK ---------------------------------------
    # Thin 'sheet' walls stand along the boundary of a curved terrace recess
    # (the semicircular pocket, the wavy front). Above a terrace floor, just
    # inside its profile, should be AIR. Sample points nudged INWARD from
    # each profile vertex at mid-recess height and count any that read SOLID.
    # NOTE: pure-python math only (np.linalg crashes with a SystemError in
    # some FreeCAD-bundled numpy builds).
    try:
        import math
        wall_hits = []
        for pk in plan.pockets:
            if 'Terrace' not in (pk.label or ''):
                continue
            pts = [(float(pr.start[0]), float(pr.start[1]))
                   for pr in pk.profile]
            if len(pts) < 13:
                continue                    # straight/simple: not the curved ones
            cu = sum(q[0] for q in pts) / len(pts)
            cv = sum(q[1] for q in pts) / len(pts)
            du_ = max(q[0] for q in pts) - min(q[0] for q in pts)
            dv_ = max(q[1] for q in pts) - min(q[1] for q in pts)
            step = 0.06 * math.hypot(du_, dv_)
            zc = L - 0.5 * float(pk.depth)  # mid-recess height (should be air)
            hits = 0
            for (u, v) in pts:
                dx, dy = cu - u, cv - v
                n = math.hypot(dx, dy) or 1.0
                uu, vv = u + dx / n * step, v + dy / n * step  # nudge inward
                w = fo + uu * fx + vv * fy + zc * fz
                try:
                    if body.Shape.isInside(
                            App.Vector(float(w[0]), float(w[1]),
                                       float(w[2])), 1e-6, True):
                        hits += 1
                except Exception:           # noqa: BLE001
                    pass
            if hits:
                wall_hits.append((pk.label,
                                  '%d/%d verts solid' % (hits, len(pts))))
        print('CURVED-TERRACE WALL CHECK: %d curved terrace(s) with standing '
              'walls' % len(wall_hits))
        for lab, info in wall_hits:
            print('  WALL in %-28s %s' % (lab, info))
        print('==> CURVED TERRACES CLEAN' if not wall_hits
              else '==> WALLS present in %d curved terrace(s)'
                   % len(wall_hits))
    except Exception as _e:                 # noqa: BLE001
        print('CURVED-TERRACE WALL CHECK failed: %r' % (_e,))

    # ---- PER-FEATURE BORE-SITE TRACE --------------------------------------
    # THE definitive localizer: every PartDesign feature keeps its own Shape
    # (the body state AFTER that feature), so sample each counterbore site
    # after EVERY feature. Per site, three tests:
    #   C = site CENTER solid at z midway between its opening face and the
    #       top (a standing column)
    #   S = thin SHELL ring solid just outside the counterbore radius, above
    #       the opening face (wall above the plate)
    #   s = same shell ring, inside the recess depth zone (wall in recess)
    #   . = clear
    # Reading the matrix top-to-bottom shows exactly which feature CREATES a
    # column/shell at which site and which feature consumes it -- and which
    # single site ends up different from the other seven.
    print()
    print('PER-FEATURE BORE-SITE TRACE (C=column, S=shell above face, '
          's=shell in recess, .=clear):')
    try:
        import math
        cb_h = None
        for h in plan.holes:
            sz0 = h.surface_z if h.surface_z is not None else L
            if getattr(h, 'from_top', True) and sz0 < L - 1e-3 * (L or 1.0) \
                    and getattr(h, 'counterbore_diameter', None):
                cb_h = h
                break
        if cb_h is None:
            print('  (no below-top counterbored hole in plan; trace skipped)')
            raise ValueError('trace skipped')
        sz = float(cb_h.surface_z)
        cbr = 0.5 * float(cb_h.counterbore_diameter)
        cbfloor = sz - float(cb_h.counterbore_depth)
        ring_r = cbr + 0.006          # just outside the counterbore circle
        z_above = 0.5 * (sz + L)      # between opening face and global top
        z_recess = 0.5 * (cbfloor + sz)   # inside the recess depth zone
        sites = [(float(u), float(v)) for (u, v) in cb_h.positions]
        for i, (u, v) in enumerate(sites):
            print('  site%d = (%+.2f, %+.2f)' % (i, u, v))
        ring = [(ring_r * math.cos(2 * math.pi * i / 12.0),
                 ring_r * math.sin(2 * math.pi * i / 12.0))
                for i in range(12)]

        def _inside(shape, w):
            try:
                return shape.isInside(
                    App.Vector(float(w[0]), float(w[1]), float(w[2])),
                    1e-6, True)
            except Exception:           # noqa: BLE001
                return False

        feats = None  # (chain built below in original feature order)
        seen = set()
        chain = []
        for o in body.Group:
            if o.Name in seen or 'Sketch' in o.TypeId:
                continue
            if 'PartDesign::' not in o.TypeId:
                continue
            if not hasattr(o, 'Shape') or o.Shape is None \
                    or o.Shape.isNull():
                continue
            seen.add(o.Name)
            chain.append(o)
        hdr = 'FEATURE'.ljust(34) + ' ' + ' '.join(
            's%d ' % i for i in range(len(sites)))
        print('  ' + hdr)
        for o in chain:
            codes = []
            for (u, v) in sites:
                c = '.'
                wc = fo + u * fx + v * fy + z_above * fz
                if _inside(o.Shape, wc):
                    c = 'C'
                else:
                    shell_above = sum(
                        1 for (du, dv) in ring if _inside(
                            o.Shape,
                            fo + (u + du) * fx + (v + dv) * fy
                            + z_above * fz))
                    shell_rec = sum(
                        1 for (du, dv) in ring if _inside(
                            o.Shape,
                            fo + (u + du) * fx + (v + dv) * fy
                            + z_recess * fz))
                    # a full ring solid in the recess zone is just the
                    # UNCUT plate (before the recess exists): mark shells
                    # only when the ring is PARTIALLY solid (a thin wall)
                    if 0 < shell_above < 12:
                        c = 'S'
                    elif 0 < shell_rec < 12:
                        c = 's'
                    elif shell_above == 12:
                        c = '#'          # fully solid at ring = uncut here
                codes.append(c.ljust(3))
            lab = (o.Label or o.Name)[:33]
            print('  ' + lab.ljust(34) + ' ' + ' '.join(codes))
        print('  (# = uncut solid, C = column, S/s = thin shell, . = clear;'
              ' the site whose column no feature clears, or whose S/s'
              ' persists to the last row, is the culprit)')
    except ValueError:
        pass                                # trace skipped (no cb hole)
    except Exception as _e:                 # noqa: BLE001
        print('PER-FEATURE TRACE failed: %r' % (_e,))
except Exception as exc:  # noqa: BLE001
    print('BUILD STEP SKIPPED/failed (%s: %s).' % (type(exc).__name__, exc))
    print('The version + routing lines above are still the key evidence.')
print('=' * 68)
