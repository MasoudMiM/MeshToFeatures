# SPDX-License-Identifier: LGPL-2.1-or-later
"""Geometric blend fallback (issue #6): executor wiring via stubbed
FreeCAD/Part.

When a detected fillet's sharp edge matches NO edge of the parametric
body (the freeform-band class: the tree approximates the band, so no
edge exists at the mesh-fit position), the executor must NOT silently
drop the blend. It dresses it geometrically instead: the shared corner
cross-section (:func:`blend_corner_profile`) sketched on the plane
through ``edge_start`` perpendicular to the edge, extruded along the
detected span -- a Pocket for convex (material removed), a Pad for
concave. Real-FreeCAD geometry stays covered by
scripts/freecad_smoke_test.py and the terminal deviation-correction
pass; here we pin the wiring.
"""

import sys
import types

import numpy as np
import pytest

from freecad.meshtofeatures_wb.core.history import (BasePad, BuildPlan,
                                                    FilletOp, ChamferOp,
                                                    SketchLine)

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ stubs

class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class FakeMatrix:
    def __init__(self, *vals):
        assert len(vals) == 16
        self.A = np.array(vals, dtype=float).reshape(4, 4)


class FakePlacement:
    def __init__(self, matrix):
        self.matrix = matrix


class FakeConsole:
    messages = []

    @staticmethod
    def PrintWarning(msg):  # noqa: N802
        FakeConsole.messages.append(("warn", msg))

    @staticmethod
    def PrintMessage(msg):  # noqa: N802
        FakeConsole.messages.append(("msg", msg))


class FakeObj:
    """A recording stand-in for a FreeCAD document object. Deliberately
    omits ``isValid`` so `_rollback_if_broken` treats it as valid."""

    def __init__(self, name):
        self.Name = name
        self.Label = name
        self.State = []
        self.children = []
        self.geometry = []
        self.Tip = None
        self.Shape = types.SimpleNamespace(
            Edges=[],
            BoundBox=types.SimpleNamespace(DiagonalLength=100.0))

    def addObject(self, obj):
        self.children.append(obj)
        self.Tip = obj              # FreeCAD promotes the tip on add

    def addGeometry(self, geom, construction=False):
        self.geometry.append(geom)
        return len(self.geometry)


class FakeDoc:
    def __init__(self):
        self.objects = {}
        self.order = []

    def addObject(self, type_name, name):
        obj = FakeObj(name)
        obj.TypeId = type_name
        self.objects[name] = obj
        self.order.append(obj)
        return obj

    def removeObject(self, name):
        self.objects.pop(name, None)

    def recompute(self):
        return None


def _install_stubs(monkeypatch):
    app = types.ModuleType("FreeCAD")
    app.Vector = FakeVector
    app.Matrix = FakeMatrix
    app.Placement = FakePlacement
    app.Console = FakeConsole
    app.GuiUp = False

    part = types.ModuleType("Part")
    part.LineSegment = lambda a, b: ("line", a, b)
    part.Circle = lambda c, n, r: ("circle", c, r)
    part.ArcOfCircle = lambda *a: ("arc", a)

    monkeypatch.setitem(sys.modules, "FreeCAD", app)
    monkeypatch.setitem(sys.modules, "Part", part)
    monkeypatch.setattr(FakeConsole, "messages", [])
    sys.modules.pop("freecad.meshtofeatures_wb.build", None)
    import freecad.meshtofeatures_wb.build as build
    return build


# ------------------------------------------------------------------ plans

def _box_plan(blends):
    profile = [
        SketchLine(start=np.array([-20.0, -15.0]),
                   end=np.array([20.0, -15.0])),
        SketchLine(start=np.array([20.0, -15.0]),
                   end=np.array([20.0, 15.0])),
        SketchLine(start=np.array([20.0, 15.0]),
                   end=np.array([-20.0, 15.0])),
        SketchLine(start=np.array([-20.0, 15.0]),
                   end=np.array([-20.0, -15.0])),
    ]
    return BuildPlan(frame_origin=np.zeros(3), frame_x=X.copy(),
                     frame_y=Y.copy(), frame_z=Z.copy(),
                     base=BasePad(profile=profile, length=10.0),
                     fillets=[b for b in blends
                              if isinstance(b, FilletOp)],
                     chamfers=[b for b in blends
                               if isinstance(b, ChamferOp)])


def _band_fillet(convex=True, z=20.0):
    """A fillet whose detected sharp edge is at z=20 -- 10 mm ABOVE the
    body's top face: the issue-#6 situation where the parametric body has
    no edge at the mesh-fit position (tol can never bridge that gap)."""
    return FilletOp(radius=3.0,
                    edge_start=np.array([20.0, -15.0, z]),
                    edge_end=np.array([20.0, 15.0, z]),
                    direction=Y.copy(), n_a=X.copy(), n_b=Z.copy(),
                    convex=convex, label="band edge")


# ------------------------------------------------------------------ tests

class FakeEdge:
    """A straight body edge for the executor's matcher to scan."""

    def __init__(self, p0, p1):
        self.Curve = types.SimpleNamespace(TypeId="Part::GeomLine")
        self.Vertexes = [types.SimpleNamespace(X=p0[0], Y=p0[1], Z=p0[2]),
                         types.SimpleNamespace(X=p1[0], Y=p1[1], Z=p1[2])]


class EdgeDoc(FakeDoc):
    """FakeDoc whose BasePad carries one straight top edge so the
    primary PartDesign::Fillet path can match it."""

    def addObject(self, type_name, name):
        obj = super().addObject(type_name, name)
        if name == "BasePad":
            obj.Shape.Edges.append(FakeEdge((20.0, -15.0, 10.0),
                                            (20.0, 15.0, 10.0)))
        return obj


class TestGeometricBlendFallback:
    def test_unmatched_convex_fillet_becomes_pocket(self, monkeypatch):
        build = _install_stubs(monkeypatch)
        doc = FakeDoc()
        plan = _box_plan([_band_fillet(convex=True)])
        build.build_body(doc, plan, name="Rebuilt")

        op = doc.objects.get("GeometricBlend0")
        assert op is not None, "fallback blend op missing"
        assert op.TypeId == "PartDesign::Pocket"
        assert np.isclose(op.Length, 30.0)          # the detected span

        sk = doc.objects.get("BlendProfile0")
        assert sk is not None
        A = sk.Placement.matrix.A
        # Pocket plane: (n_a, -n_b, -d) at edge_start -- the pocket cuts
        # along +d (edge_start -> edge_end), opposite the sketch normal.
        # The test fillet's order is LEFT-handed (X x Z = -Y = -d), so the
        # executor swaps n_a/n_b first: the plane is (Z, -X, -Y).
        assert np.allclose(A[:3, 0], Z, atol=1e-9)
        assert np.allclose(A[:3, 1], -X, atol=1e-9)
        assert np.allclose(A[:3, 2], -Y, atol=1e-9)
        assert np.allclose(A[:3, 3], [20.0, -15.0, 20.0], atol=1e-9)
        # profile: the mirrored corner sliver (two lines + one arc)
        kinds = [g[0] for g in sk.geometry]
        assert kinds.count("line") == 2 and kinds.count("arc") == 1
        # loud, named report -- not a silent drop
        assert any("dressed geometrically" in m for _, m
                   in FakeConsole.messages)

    def test_unmatched_concave_fillet_becomes_pad(self, monkeypatch):
        build = _install_stubs(monkeypatch)
        doc = FakeDoc()
        plan = _box_plan([_band_fillet(convex=False)])
        build.build_body(doc, plan, name="Rebuilt")

        op = doc.objects.get("GeometricBlend0")
        assert op is not None
        assert op.TypeId == "PartDesign::Pad"       # material ADDED
        assert np.isclose(op.Length, 30.0)

        sk = doc.objects.get("BlendProfile0")
        A = sk.Placement.matrix.A
        # Pad plane: (n_a, n_b, d), unmirrored, normal = +d. Left-handed
        # test order (X x Z = -d) is swapped to (Z, X, Y) first.
        assert np.allclose(A[:3, 0], Z, atol=1e-9)
        assert np.allclose(A[:3, 1], X, atol=1e-9)
        assert np.allclose(A[:3, 2], Y, atol=1e-9)
        # legs buried 0.02 * radius past the origin (fuse doctrine)
        starts = [g[1] for g in sk.geometry if g[0] == "line"]
        assert min(v.y for v in starts) == pytest.approx(-0.06)

    def test_plane_is_right_handed_for_both_neighbour_orders(self,
                                                            monkeypatch):
        # _fillet_op collects the two neighbour normals in surface order,
        # so n_a x n_b = +-d both occur in the field. App.Placement
        # silently negates a left-handed rotation (R -> R*(-I)), which
        # would extrude the blend along -d into the wrong quadrant -- so
        # the executor must normalize the order. Both orderings of the
        # SAME physical fillet must yield the identical plane.
        build = _install_stubs(monkeypatch)
        planes = []
        for na, nb in ((X, Z), (Z, X)):
            doc = FakeDoc()
            f = FilletOp(radius=3.0,
                         edge_start=np.array([20.0, -15.0, 20.0]),
                         edge_end=np.array([20.0, 15.0, 20.0]),
                         direction=Y.copy(), n_a=na.copy(),
                         n_b=nb.copy(), convex=True, label="band edge")
            build.build_body(doc, _box_plan([f]), name="Rebuilt")
            sk = doc.objects.get("BlendProfile0")
            assert sk is not None
            A = sk.Placement.matrix.A
            # right-handed: col0 x col1 = col2
            assert np.allclose(np.cross(A[:3, 0], A[:3, 1]), A[:3, 2],
                               atol=1e-9)
            planes.append(A[:3, :3])
        assert np.allclose(planes[0], planes[1], atol=1e-9)

    def test_matched_edge_uses_primary_dressup(self, monkeypatch):
        build = _install_stubs(monkeypatch)
        doc = EdgeDoc()
        # detected sharp edge ON a body edge (x=20, z=10): the primary
        # PartDesign::Fillet path must win, no fallback op is created
        plan = _box_plan([_band_fillet(convex=True, z=10.0)])
        build.build_body(doc, plan, name="Rebuilt")

        assert doc.objects.get("GeometricBlend0") is None
        assert doc.objects.get("Dressup0") is not None
        assert doc.objects["Dressup0"].TypeId == "PartDesign::Fillet"

    def test_unmatched_chamfer_still_skips_loudly(self, monkeypatch):
        build = _install_stubs(monkeypatch)
        doc = FakeDoc()
        op = ChamferOp(size=3.0,
                       edge_start=np.array([20.0, -15.0, 20.0]),
                       edge_end=np.array([20.0, 15.0, 20.0]),
                       direction=Y.copy(), n_a=X.copy(), n_b=Z.copy(),
                       label="band chamfer")
        plan = _box_plan([op])
        build.build_body(doc, plan, name="Rebuilt")

        assert doc.objects.get("GeometricBlend0") is None
        assert any("no body edge matched" in m and "skipped" in m
                   for _, m in FakeConsole.messages)

    def test_degenerate_blend_declined_without_op(self, monkeypatch):
        build = _install_stubs(monkeypatch)
        doc = FakeDoc()
        blend = _band_fillet(convex=True)
        blend.n_a = Y.copy()                        # parallel to direction
        plan = _box_plan([blend])
        build.build_body(doc, plan, name="Rebuilt")

        assert doc.objects.get("GeometricBlend0") is None
        assert doc.objects.get("BlendProfile0") is None
        # still reported, not silent
        assert any("no body edge matched" in m for _, m
                   in FakeConsole.messages)

    def test_chain_continues_after_fallback(self, monkeypatch):
        build = _install_stubs(monkeypatch)
        doc = FakeDoc()
        # two band fillets: both dressed geometrically, chain intact
        plan = _box_plan([_band_fillet(convex=True),
                          _band_fillet(convex=False)])
        build.build_body(doc, plan, name="Rebuilt")

        for k in (0, 1):
            assert doc.objects.get(f"GeometricBlend{k}") is not None
            assert doc.objects.get(f"BlendProfile{k}") is not None
