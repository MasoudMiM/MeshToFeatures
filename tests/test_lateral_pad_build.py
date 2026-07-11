# SPDX-License-Identifier: LGPL-2.1-or-later
"""Lateral-pad FreeCAD executor wiring, via stubbed FreeCAD/Part.

FreeCAD cannot run in CI, but the executor's *wiring* can be checked
with stubs: a lateral PadOp must produce a PartDesign::Pad (not a
Pocket) whose Length is the pad length and whose sketch Placement
carries the world basis (u, v, axis, origin) from
`lateral_pad_world_frame`. Real-FreeCAD geometry is covered by
scripts/freecad_smoke_test.py (the RebuiltFlange case).
"""

import sys
import types

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from .test_lateral_pad import flanged_plate                       # noqa: E402
from .test_adversarial import _plan                               # noqa: E402


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
    @staticmethod
    def PrintWarning(msg):  # noqa: N802
        pass

    @staticmethod
    def PrintMessage(msg):  # noqa: N802
        pass


class FakeObj:
    """A recording stand-in for a FreeCAD document object. Deliberately
    omits ``isValid`` so `_rollback_if_broken` treats it as valid."""

    def __init__(self, name):
        self.Name = name
        self.Label = name
        self.State = []
        self.children = []

    def addObject(self, obj):
        self.children.append(obj)

    def addGeometry(self, geom, construction=False):
        return len(self.children)


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
    sys.modules.pop("freecad.meshtofeatures_wb.build", None)
    import freecad.meshtofeatures_wb.build as build
    return build


# ------------------------------------------------------------------- test

class TestLateralPadExecutor:
    def test_lateral_pad_becomes_a_pad_with_world_placement(self, monkeypatch):
        build = _install_stubs(monkeypatch)
        from meshtofeatures.history import lateral_pad_world_frame

        mesh = flanged_plate()
        _, _, plan = _plan(mesh)
        pad = [p for p in plan.pads
               if getattr(p, "axis", None) is not None][0]

        doc = FakeDoc()
        build.build_body(doc, plan, name="RebuiltFlange")

        # a lateral pad becomes a Pad (not a Pocket)
        op = doc.objects.get("LateralPad0")
        assert op is not None and op.TypeId == "PartDesign::Pad"
        assert np.isclose(op.Length, pad.length)

        # its sketch placement carries the world basis
        sk = doc.objects.get("LateralPadProfile0")
        assert sk is not None
        A = sk.Placement.matrix.A
        origin, u, v, axis = lateral_pad_world_frame(plan, pad)
        assert np.allclose(A[:3, 0], u, atol=1e-9)
        assert np.allclose(A[:3, 1], v, atol=1e-9)
        assert np.allclose(A[:3, 2], axis, atol=1e-9)
        assert np.allclose(A[:3, 3], origin, atol=1e-9)
