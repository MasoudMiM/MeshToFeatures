# SPDX-License-Identifier: LGPL-2.1-or-later
"""Headless end-to-end smoke test to run INSIDE FreeCAD's Python.

Usage (from the addon root):
    freecadcmd scripts/freecad_smoke_test.py
or on Windows:
    "C:/Program Files/FreeCAD 1.x/bin/FreeCADCmd.exe" scripts/freecad_smoke_test.py

It builds a native Part cylinder, tessellates it, runs the full
reconstruct -> snap -> plan -> emit pipeline, checks the results, and
saves a document you can open to inspect visually.
"""

import os as _os
import sys as _sys
if _os.environ.get("_MTF_SMOKE_DONE") == "1":
    _sys.stdout.write("[smoke] duplicate CLI invocation detected; skipping second run\n")
    _sys.stdout.flush()
    _os._exit(0)   # some FreeCAD builds execute CLI scripts twice
_os.environ["_MTF_SMOKE_DONE"] = "1"

import math
import os
import sys
import tempfile

ADDON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ADDON_ROOT not in sys.path:
    sys.path.insert(0, ADDON_ROOT)

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

fails = []


def check(name, cond, info=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({info})" if info else ""))
    if not cond:
        fails.append(name)


# --- dependencies ----------------------------------------------------------
try:
    import numpy as np
    import scipy  # noqa: F401
    import trimesh
    check("python dependencies importable", True)
except ImportError as exc:
    check("python dependencies importable", False, str(exc))
    print("Install into FreeCAD's Python, e.g.: "
          f"'{sys.executable}' -m pip install numpy scipy trimesh")
    sys.exit(1)

# regression check for the freecad namespace-shadowing bug: FreeCAD's own
# submodules must remain importable AFTER our addon's freecad/ dir is on the
# path (this is what broke STL import in v0.3.0)
try:
    from freecad import module_io  # noqa: F401
    check("freecad namespace merges (module_io importable)", True)
except ImportError as exc:
    check("freecad namespace merges (module_io importable)", False, str(exc))

from meshtofeatures.pipeline import reconstruct  # noqa: E402
from meshtofeatures.snapping import snap_report  # noqa: E402
from meshtofeatures.emission import plan_patches  # noqa: E402
from freecad.meshtofeatures_wb import emit  # noqa: E402

# --- build & tessellate a known solid --------------------------------------
solid = Part.makeCylinder(10.0, 30.0)  # r=10, h=30, axis +Z at origin
verts, faces = solid.tessellate(0.05)
tm = trimesh.Trimesh(
    vertices=np.array([[v.x, v.y, v.z] for v in verts]),
    faces=np.array(faces), process=True)
check("tessellation is watertight", tm.is_watertight, f"{len(tm.faces)} faces")

# --- core pipeline ----------------------------------------------------------
report = reconstruct(tm)
check("kinds == cylinder + 2 planes",
      report.kinds() == ["cylinder", "plane", "plane"], str(report.kinds()))

result = snap_report(report)
cyls = result.report.by_kind("cylinder")
check("one cylinder after snapping", len(cyls) == 1)
if cyls:
    c = cyls[0].fit.primitive
    check("radius snapped to exactly 10.0", c.radius == 10.0, f"r={c.radius}")
    check("axis snapped to exactly +/-Z",
          abs(abs(float(c.axis[2])) - 1.0) < 1e-15, f"axis={c.axis}")
for a in result.actions:
    print(f"    [{a.kind}] {'OK ' if a.accepted else 'REJ'} {a.detail}")

# --- emission ----------------------------------------------------------------
patches = plan_patches(result.report)
check("one patch per surface", len(patches) == len(result.report.surfaces))

doc = App.newDocument("MeshToFeaturesSmoke")
group = emit.emit_report(doc, patches, result.actions, group_label="SmokeTest")
part_objs = [o for o in group.Group]
check("all patches emitted as Part features",
      len(part_objs) == len(patches), f"{len(part_objs)}/{len(patches)}")

areas_ok = True
for obj, spec in zip(part_objs, patches):
    if spec.kind == "cylinder":
        expected = 2 * math.pi * spec.primitive.radius * (spec.v_range[1] - spec.v_range[0])
        ok = abs(obj.Shape.Area - expected) / expected < 1e-6
        areas_ok &= ok
        check("cylinder face area matches 2*pi*r*h", ok,
              f"{obj.Shape.Area:.6f} vs {expected:.6f}")
    check(f"{obj.Label}: shape is valid", obj.Shape.isValid())

# --- plane-with-hole emission (annulus) ------------------------------------
washer = Part.makeCylinder(15.0, 4.0).cut(Part.makeCylinder(6.0, 4.0))
wv, wf = washer.tessellate(0.05)
wtm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in wv]),
                      faces=np.array(wf), process=True)
wrep = reconstruct(wtm)
wpatches = plan_patches(wrep)
plane_specs = [p for p in wpatches if p.kind == "plane"]
check("washer: two plane patches", len(plane_specs) == 2)
check("washer: each plane has one hole",
      all(len(p.holes) == 1 for p in plane_specs),
      str([len(p.holes) for p in plane_specs]))
wgroup = emit.emit_report(doc, wpatches, group_label="WasherTest")
expected_area = math.pi * (15.0**2 - 6.0**2)
for obj, spec in zip(wgroup.Group, wpatches):
    check(f"washer {obj.Label}: shape is valid", obj.Shape.isValid())
    if spec.kind == "plane":
        ok = abs(obj.Shape.Area - expected_area) / expected_area < 0.01
        check("washer plane area = pi(R^2 - r^2), hole excluded", ok,
              f"{obj.Shape.Area:.2f} vs {expected_area:.2f}")

# --- PartDesign rebuild (washer: base circle + through hole) ----------------
from meshtofeatures.features import detect_features  # noqa: E402
from meshtofeatures.patterns import detect_patterns  # noqa: E402
from meshtofeatures.history import plan_history  # noqa: E402
from freecad.meshtofeatures_wb import build  # noqa: E402
wfeats = detect_features(wrep, wpatches)
wplan = plan_history(wrep, wfeats, detect_patterns(wfeats), wpatches)
body = build.build_body(doc, wplan, name="RebuiltWasher")
expected_vol = math.pi * (15.0**2 - 6.0**2) * 4.0
ok = abs(body.Shape.Volume - expected_vol) / expected_vol < 0.01
check("rebuilt washer body volume within 1%", ok,
      f"{body.Shape.Volume:.2f} vs {expected_vol:.2f}")
check("rebuilt body shape is valid", body.Shape.isValid())

# --- PartDesign Hole feature (counterbored plate) ---------------------------
cb_solid = Part.makeBox(40, 30, 5, App.Vector(-20, -15, 0)) \
    .cut(Part.makeCylinder(4.0, 50, App.Vector(0, 0, -20))) \
    .cut(Part.makeCylinder(6.0, 2.0, App.Vector(0, 0, 3.0)))
cv, cf = cb_solid.tessellate(0.05)
ctm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in cv]),
                      faces=np.array(cf), process=True)
crep = snap_report(reconstruct(ctm)).report
cpatches = plan_patches(crep)
cfeats = detect_features(crep, cpatches)
cplan = plan_history(crep, cfeats, detect_patterns(cfeats), cpatches)
cbody = build.build_body(doc, cplan, name="RebuiltCounterbore")
check("counterbore body uses a PartDesign::Hole feature",
      any(o.TypeId == "PartDesign::Hole" for o in cbody.Group))
check("counterbore body volume within 1%",
      abs(cbody.Shape.Volume - cb_solid.Volume) / cb_solid.Volume < 0.01,
      f"{cbody.Shape.Volume:.2f} vs {cb_solid.Volume:.2f}")
check("counterbore body shape is valid", cbody.Shape.isValid())

# --- horizontal fillets rebuilt as PartDesign::Fillet ------------------------
fp = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0))
top_edges = [e for e in fp.Edges
             if abs(e.Vertexes[0].Z - 10) < 1e-6
             and abs(e.Vertexes[-1].Z - 10) < 1e-6
             and abs(e.Vertexes[0].X - e.Vertexes[-1].X) < 1e-6]
fp = fp.makeFillet(3.0, top_edges)          # fillet both long top edges
fv, ff = fp.tessellate(0.05)
ftm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in fv]),
                      faces=np.array(ff), process=True)
frep = snap_report(reconstruct(ftm)).report
fpatches = plan_patches(frep)
ffeats = detect_features(frep, fpatches)
fplan = plan_history(frep, ffeats, detect_patterns(ffeats), fpatches)
check("fillet plan carries 2 fillet ops", len(fplan.fillets) == 2,
      str(len(fplan.fillets)))
fbody = build.build_body(doc, fplan, name="RebuiltFilleted")
check("body contains PartDesign::Fillet features",
      sum(1 for o in fbody.Group if o.TypeId == "PartDesign::Fillet") == 2)
check("filleted body volume within 1%",
      abs(fbody.Shape.Volume - fp.Volume) / fp.Volume < 0.01,
      f"{fbody.Shape.Volume:.2f} vs {fp.Volume:.2f}")
check("filleted body shape is valid", fbody.Shape.isValid())

# --- chamfers rebuilt as PartDesign::Chamfer --------------------------------
cp = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0))
top_e = [e for e in cp.Edges
         if abs(e.Vertexes[0].Z - 10) < 1e-6
         and abs(e.Vertexes[-1].Z - 10) < 1e-6
         and abs(e.Vertexes[0].X - e.Vertexes[-1].X) < 1e-6]
cp = cp.makeChamfer(2.0, top_e)
cv2, cf2 = cp.tessellate(0.05)
cptm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in cv2]),
                       faces=np.array(cf2), process=True)
cprep = snap_report(reconstruct(cptm)).report
cppat = plan_patches(cprep)
cpfeats = detect_features(cprep, cppat)
cpplan = plan_history(cprep, cpfeats, detect_patterns(cpfeats), cppat)
check("chamfer plan carries 2 chamfer ops", len(cpplan.chamfers) == 2,
      str(len(cpplan.chamfers)))
cpbody = build.build_body(doc, cpplan, name="RebuiltChamfered")
check("body contains PartDesign::Chamfer features",
      sum(1 for o in cpbody.Group if o.TypeId == "PartDesign::Chamfer") == 2)
check("chamfered body volume within 1%",
      abs(cpbody.Shape.Volume - cp.Volume) / cp.Volume < 0.01,
      f"{cpbody.Shape.Volume:.2f} vs {cp.Volume:.2f}")

# --- cross-axis hole rebuilt as midplane through-all pocket ------------------
xs = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0)) \
    .cut(Part.makeCylinder(3.0, 80, App.Vector(-40, 0, 5), App.Vector(1, 0, 0)))
xv, xf = xs.tessellate(0.05)
xtm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in xv]),
                      faces=np.array(xf), process=True)
xrep = snap_report(reconstruct(xtm)).report
xpat = plan_patches(xrep)
xfeats = detect_features(xrep, xpat)
xplan = plan_history(xrep, xfeats, detect_patterns(xfeats), xpat)
check("cross-hole plan carries 1 op", len(xplan.cross_holes) == 1,
      str(len(xplan.cross_holes)))
xbody = build.build_body(doc, xplan, name="RebuiltCrossHole")
check("cross-hole body volume within 1%",
      abs(xbody.Shape.Volume - xs.Volume) / xs.Volume < 0.01,
      f"{xbody.Shape.Volume:.2f} vs {xs.Volume:.2f}")
check("cross-hole body shape is valid", xbody.Shape.isValid())

# --- multi-level step part: exercises from-bottom/from-top pocket sides ------
sp = Part.makeBox(40, 30, 6, App.Vector(-20, -15, 0)) \
    .fuse(Part.makeBox(16, 10, 4, App.Vector(-8, -5, 6))).removeSplitter()
sv, sf = sp.tessellate(0.05)
sptm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in sv]),
                       faces=np.array(sf), process=True)
sprep = snap_report(reconstruct(sptm)).report
sppat = plan_patches(sprep)
spfeats = detect_features(sprep, sppat)
spplan = plan_history(sprep, spfeats, detect_patterns(spfeats), sppat)
check("step plan carries 1 step", len(spplan.step_labels) == 1,
      str(spplan.step_labels))
spbody = build.build_body(doc, spplan, name="RebuiltStep")
check("step body volume within 1%",
      abs(spbody.Shape.Volume - sp.Volume) / sp.Volume < 0.01,
      f"{spbody.Shape.Volume:.2f} vs {sp.Volume:.2f}")
check("step body shape is valid", spbody.Shape.isValid())

# --- lateral pad: rectangular flange protruding off the -y wall ------------
# design note 36: base 40x30x10 with a full-width flange y[-20,-15] z[3,8].
fl = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0)) \
    .fuse(Part.makeBox(40, 5, 5, App.Vector(-20, -20, 3))).removeSplitter()
flv, flf = fl.tessellate(0.05)
fltm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in flv]),
                       faces=np.array(flf), process=True)
flrep = snap_report(reconstruct(fltm)).report
flpat = plan_patches(flrep)
flfeats = detect_features(flrep, flpat)
flplan = plan_history(flrep, flfeats, detect_patterns(flfeats), flpat)
n_lateral = sum(1 for p in flplan.pads if getattr(p, "axis", None) is not None)
check("flange plan carries 1 lateral pad", n_lateral == 1, str(n_lateral))
flbody = build.build_body(doc, flplan, name="RebuiltFlange")
check("flange body volume within 1%",
      abs(flbody.Shape.Volume - fl.Volume) / fl.Volume < 0.01,
      f"{flbody.Shape.Volume:.2f} vs {fl.Volume:.2f}")
check("flange body shape is valid", flbody.Shape.isValid())

# --- two lugs on one side: multiple lateral pads must not bridge the gap ----
tl = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0)) \
    .fuse(Part.makeBox(10, 5, 5, App.Vector(-18, -20, 3))) \
    .fuse(Part.makeBox(10, 5, 5, App.Vector(8, -20, 3))).removeSplitter()
tlv, tlf = tl.tessellate(0.05)
tltm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in tlv]),
                       faces=np.array(tlf), process=True)
tlrep = snap_report(reconstruct(tltm)).report
tlpat = plan_patches(tlrep)
tlfeats = detect_features(tlrep, tlpat)
tlplan = plan_history(tlrep, tlfeats, detect_patterns(tlfeats), tlpat)
n_tl = sum(1 for p in tlplan.pads if getattr(p, "axis", None) is not None)
check("two-lug plan carries 2 lateral pads", n_tl == 2, str(n_tl))
tlbody = build.build_body(doc, tlplan, name="RebuiltTwoLug")
check("two-lug body volume within 1%",
      abs(tlbody.Shape.Volume - tl.Volume) / tl.Volume < 0.01,
      f"{tlbody.Shape.Volume:.2f} vs {tl.Volume:.2f}")
check("two-lug body shape is valid", tlbody.Shape.isValid())

# --- curved protrusions: horizontal cylinder -> circular lateral pad --------
# a peg boss (axis perpendicular to the wall) and a rounded rail (axis
# parallel to the wall); both become a circular lateral pad.
bs = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0)).fuse(
    Part.makeCylinder(3, 6, App.Vector(0, -15, 5), App.Vector(0, -1, 0))
).removeSplitter()
rl = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0)).fuse(
    Part.makeCylinder(3, 40, App.Vector(-20, -15, 5), App.Vector(1, 0, 0))
).removeSplitter()
for tag, solid in (("boss", bs), ("rail", rl)):
    sv, sf = solid.tessellate(0.02)
    tm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in sv]),
                         faces=np.array(sf), process=True)
    rep = snap_report(reconstruct(tm)).report
    pat = plan_patches(rep)
    fts = detect_features(rep, pat)
    pln = plan_history(rep, fts, detect_patterns(fts), pat)
    ncirc = sum(1 for p in pln.pads if getattr(p, "axis", None) is not None)
    check(f"{tag} plan carries a circular lateral pad", ncirc == 1, str(ncirc))
    body = build.build_body(doc, pln, name=f"Rebuilt{tag.capitalize()}")
    check(f"{tag} body volume within 1%",
          abs(body.Shape.Volume - solid.Volume) / solid.Volume < 0.01,
          f"{body.Shape.Volume:.2f} vs {solid.Volume:.2f}")
    check(f"{tag} body shape is valid", body.Shape.isValid())

import math as _math
# --- frame/ring base: an octagonal through-opening the base must carry ------
_oct = [App.Vector(12 * _math.cos(i * _math.pi / 4),
                   12 * _math.sin(i * _math.pi / 4), -5) for i in range(8)]
_prism = Part.Face(Part.makePolygon(_oct + [_oct[0]])).extrude(
    App.Vector(0, 0, 20))
frm = Part.makeBox(40, 40, 10, App.Vector(-20, -20, 0)).cut(_prism)
fmv, fmf = frm.tessellate(0.02)
fmtm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in fmv]),
                       faces=np.array(fmf), process=True)
fmrep = snap_report(reconstruct(fmtm)).report
fmpat = plan_patches(fmrep)
fmfeats = detect_features(fmrep, fmpat)
fmplan = plan_history(fmrep, fmfeats, detect_patterns(fmfeats), fmpat)
check("frame plan carries a base through-hole",
      len(getattr(fmplan.base, "hole_profiles", [])) == 1,
      str(len(getattr(fmplan.base, "hole_profiles", []))))
fmbody = build.build_body(doc, fmplan, name="RebuiltFrame")
check("frame body volume within 3%",
      abs(fmbody.Shape.Volume - frm.Volume) / frm.Volume < 0.03,
      f"{fmbody.Shape.Volume:.2f} vs {frm.Volume:.2f}")
check("frame body shape is valid", fmbody.Shape.isValid())

# --- from-bottom counterbored holes: ThroughAll cuts the wrong way on a
#     flipped (from-bottom) sketch, so a Length cut through the base is used --
cbp = Part.makeBox(40, 30, 10, App.Vector(-20, -15, 0))
for cx, cy in [(-8, 0), (8, 0)]:
    cbp = cbp.cut(Part.makeCylinder(2.0, 40, App.Vector(cx, cy, -15)))   # through
    cbp = cbp.cut(Part.makeCylinder(4.0, 3.0, App.Vector(cx, cy, 7)))    # cbore @ top
cbp.rotate(App.Vector(0, 0, 0), App.Vector(1, 0, 0), 180)   # cbore face -> bottom
cbv, cbf = cbp.tessellate(0.05)
cbtm = trimesh.Trimesh(vertices=np.array([[v.x, v.y, v.z] for v in cbv]),
                       faces=np.array(cbf), process=True)
cbrep = snap_report(reconstruct(cbtm)).report
cbpat = plan_patches(cbrep)
cbfeats = detect_features(cbrep, cbpat)
cbplan = plan_history(cbrep, cbfeats, detect_patterns(cbfeats), cbpat)
cbholes = [h for h in cbplan.holes if h.counterbore_diameter]
check("cbore plan found a counterbored hole", len(cbholes) >= 1,
      f"{len(cbholes)} cbore holes")
check("cbore hole is from-bottom (the path that failed)",
      bool(cbholes) and not cbholes[0].from_top,
      f"from_top={cbholes[0].from_top if cbholes else '?'}")
cbbody = build.build_body(doc, cbplan, name="RebuiltCBoreBottom")
check("cbore body volume within 5% (holes+counterbores actually cut)",
      abs(cbbody.Shape.Volume - cbp.Volume) / cbp.Volume < 0.05,
      f"{cbbody.Shape.Volume:.2f} vs {cbp.Volume:.2f}")
check("cbore body shape is valid", cbbody.Shape.isValid())

out = os.path.join(tempfile.gettempdir(), "meshtofeatures_smoke.FCStd")
doc.saveAs(out)
print(f"\nDocument saved to: {out}  (open it in FreeCAD to inspect)")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"))
sys.exit(0 if not fails else 1)
