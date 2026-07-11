# SPDX-License-Identifier: LGPL-2.1-or-later
"""FreeCAD-adapter tests using stubbed FreeCAD/Part modules.

FreeCAD itself cannot run in CI, but the adapter's *math* can be verified:
placement matrices must encode the patch frame, and OCC parameter
conversions (notably the cone's slant-distance parametrization) must be
exact. Real-FreeCAD behaviour is covered by scripts/freecad_smoke_test.py.
"""

import math
import sys
import types

import numpy as np
import pytest
import trimesh


# ---------------------------------------------------------------- stubs

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


class FakeShape:
    def __init__(self, surface, args):
        self.surface = surface
        self.toShape_args = args
        self.Placement = None


class _FakeQuadric:
    def __init__(self):
        self.Radius = 1.0
        self.SemiAngle = 0.0

    def toShape(self, *args):  # noqa: N802 (FreeCAD API)
        return FakeShape(self, args)


class FakeConsole:
    @staticmethod
    def PrintWarning(msg):  # noqa: N802
        pass

    @staticmethod
    def PrintMessage(msg):  # noqa: N802
        pass


def _install_stubs(monkeypatch):
    fake_app = types.ModuleType("FreeCAD")
    fake_app.Vector = FakeVector
    fake_app.Matrix = FakeMatrix
    fake_app.Placement = FakePlacement
    fake_app.Console = FakeConsole
    fake_app.GuiUp = False

    fake_part = types.ModuleType("Part")
    fake_part.Cylinder = _FakeQuadric
    fake_part.Cone = _FakeQuadric
    fake_part.Sphere = _FakeQuadric
    fake_part.makePolygon = lambda pts: pts

    class FakeFace:
        def __init__(self, wire):
            self.wire = wire
            self.cuts = []

        def cut(self, other):
            self.cuts.append(other)
            return self

    fake_part.Face = FakeFace

    monkeypatch.setitem(sys.modules, "FreeCAD", fake_app)
    monkeypatch.setitem(sys.modules, "Part", fake_part)
    # force re-import with stubs in place
    sys.modules.pop("freecad.meshtofeatures_wb.emit", None)
    import freecad.meshtofeatures_wb.emit as emit
    return emit


# ---------------------------------------------------------------- helpers

def _patches_for(mesh):
    from meshtofeatures.pipeline import reconstruct
    from meshtofeatures.emission import plan_patches
    return plan_patches(reconstruct(mesh))


# ---------------------------------------------------------------- tests

class TestPlacement:
    def test_matrix_encodes_frame(self, monkeypatch):
        emit = _install_stubs(monkeypatch)
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        spec = [p for p in _patches_for(mesh) if p.kind == "cylinder"][0]
        shape = emit.patch_to_shape(spec)
        A = shape.Placement.matrix.A
        assert np.allclose(A[:3, 0], spec.x_dir)
        assert np.allclose(A[:3, 1], spec.y_dir)
        assert np.allclose(A[:3, 2], spec.z_dir)
        assert np.allclose(A[:3, 3], spec.origin)
        assert np.allclose(A[3], [0, 0, 0, 1])


class TestCylinderShape:
    def test_ranges_pass_through(self, monkeypatch):
        emit = _install_stubs(monkeypatch)
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        spec = [p for p in _patches_for(mesh) if p.kind == "cylinder"][0]
        shape = emit.patch_to_shape(spec)
        u0, u1, v0, v1 = shape.toShape_args
        assert (u0, u1) == spec.u_range
        assert (v0, v1) == spec.v_range
        assert shape.surface.Radius == spec.primitive.radius


class TestConeShape:
    def test_slant_conversion(self, monkeypatch):
        emit = _install_stubs(monkeypatch)
        mesh = trimesh.creation.cone(radius=1.0, height=2.0, sections=64)
        spec = [p for p in _patches_for(mesh) if p.kind == "cone"][0]
        shape = emit.patch_to_shape(spec)
        alpha = spec.primitive.half_angle
        h0, h1 = spec.v_range
        u0, u1, s0, s1 = shape.toShape_args
        # reference circle sits at height h0 above the apex
        assert np.isclose(shape.surface.Radius, h0 * math.tan(alpha))
        # v parameter is slant distance from that reference circle
        assert np.isclose(s0, 0.0)
        assert np.isclose(s1, (h1 - h0) / math.cos(alpha))
        # placement origin is apex + h0 * axis
        A = shape.Placement.matrix.A
        assert np.allclose(A[:3, 3], spec.origin + h0 * spec.z_dir)

    def test_reconstructed_tip_radius_zero(self, monkeypatch):
        emit = _install_stubs(monkeypatch)
        mesh = trimesh.creation.cone(radius=1.0, height=2.0, sections=64)
        spec = [p for p in _patches_for(mesh) if p.kind == "cone"][0]
        shape = emit.patch_to_shape(spec)
        # the segment reaches the apex, so h0 ~ 0 and the reference radius
        # must be ~0 within fit precision (OCC accepts R = 0)
        assert shape.surface.Radius == pytest.approx(0.0, abs=1e-6)


class TestPlaneShape:
    def test_polygon_is_closed(self, monkeypatch):
        emit = _install_stubs(monkeypatch)
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        spec = [p for p in _patches_for(mesh) if p.kind == "plane"][0]
        face = emit.patch_to_shape(spec)
        wire = face.wire
        first, last = wire[0], wire[-1]
        assert (first.x, first.y, first.z) == (last.x, last.y, last.z)
        assert len(wire) == len(spec.polygon) + 1
        assert face.cuts == []  # box faces have no holes

    def test_plane_with_hole_is_cut(self, monkeypatch):
        emit = _install_stubs(monkeypatch)
        mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.5, height=1.0, sections=64)
        specs = [p for p in _patches_for(mesh) if p.kind == "plane"]
        assert len(specs) == 2
        for spec in specs:
            face = emit.patch_to_shape(spec)
            assert len(face.cuts) == 1  # exactly one hole cut per annulus face


class TestMeshConversion:
    def test_mesh_object_roundtrip(self, monkeypatch):
        emit = _install_stubs(monkeypatch)
        src = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        class FakeMesh:
            Topology = (
                [FakeVector(*v) for v in src.vertices],
                [tuple(f) for f in src.faces],
            )

        class FakeMeshObj:
            Mesh = FakeMesh()

        tm = emit.mesh_object_to_trimesh(FakeMeshObj())
        assert np.isclose(tm.volume, 1.0)
        assert len(tm.faces) == len(src.faces)
