# SPDX-License-Identifier: LGPL-2.1-or-later
"""Issue #7 regression: the corrected Part::Feature installed by the
terminal deviation-correction pass could be an invalid Part.Compound
whose volume is off by ~3%.

Root cause: mesh booleans round-trip through float32 (trimesh's manifold
engine), which leaves debris on the corrected mesh -- ~zero-volume sliver
components touching the real solid, cracked edges where coincident
shells share a geometric edge. A shell built from such a mesh is
topologically invalid for OCC (an edge with four faces), so
``Part.Solid`` fails, the fallback keeps the raw shell as a Compound, and
that compound integrates to a wrong volume.

The fix: ``conditioning.repair_solid`` normalizes the mesh before the
BRep conversion (manifold-engine round-trip, micro-debris components
dropped, surviving components fused), ``solidify.apply_corrections``
fuses the UNDER patches in a single combined boolean instead of one
round-trip per patch, and ``build._shape_from_mesh`` gates the Part
volume against the trimesh volume with a loud warning on mismatch.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh

pytest.importorskip("manifold3d")

from freecad.meshtofeatures_wb.core.conditioning import repair_solid     # noqa: E402
from freecad.meshtofeatures_wb.core.solidify import (                     # noqa: E402
    apply_corrections, plan_corrections, plan_to_mesh)

_STL = Path(__file__).parent / "fixtures" / "Simple_Corner_Bracket.stl"


def _full():
    from freecad.meshtofeatures_wb.core.emission import plan_patches
    from freecad.meshtofeatures_wb.core.features import detect_features
    from freecad.meshtofeatures_wb.core.history import plan_history
    from freecad.meshtofeatures_wb.core.patterns import detect_patterns
    from freecad.meshtofeatures_wb.core.pipeline import reconstruct
    from freecad.meshtofeatures_wb.core.snapping import snap_report

    mesh = trimesh.load(str(_STL), force="mesh")
    report = snap_report(reconstruct(mesh)).report
    feats = detect_features(report, plan_patches(report))
    plan = plan_history(report, feats, detect_patterns(feats),
                        plan_patches(report))
    return mesh, plan


def _nonmanifold_edges(m):
    e = m.edges_sorted
    key = e[:, 0].astype(np.int64) * len(m.vertices) \
        + e[:, 1].astype(np.int64)
    _, counts = np.unique(key, return_counts=True)
    return int((counts > 2).sum())


# --------------------------------------------------------------------------
# repair_solid (headless)
# --------------------------------------------------------------------------

class TestRepairSolid:
    def test_clean_solid_passes_through(self):
        box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
        out, rep = repair_solid(box)
        assert out.volume == pytest.approx(8000.0, rel=1e-3)
        assert out.is_watertight
        assert rep.components_dropped == 0
        assert len(out.split(only_watertight=False)) == 1

    def test_micro_sliver_component_dropped(self):
        # A 0.04 mm^3 sliver on a 8000 mm^3 part is boolean debris
        # (plan_corrections never emits patches below 1e-4 of the part).
        box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
        slab = trimesh.creation.box(extents=[20.0, 20.0, 1e-4])
        slab.apply_translation([0.0, 0.0, 10.0 + 5e-5])
        dirty = trimesh.util.concatenate([box, slab])
        assert len(dirty.split(only_watertight=False)) == 2
        out, rep = repair_solid(dirty)
        assert rep.components_dropped == 1
        assert out.volume == pytest.approx(8000.0, rel=1e-4)
        assert out.is_watertight
        assert len(out.split(only_watertight=False)) == 1

    def test_real_patch_component_kept(self):
        # A genuinely separate piece above the debris floor must survive.
        box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
        chunk = trimesh.creation.box(extents=[2.0, 2.0, 2.0])   # 8 = 1e-3
        chunk.apply_translation([30.0, 30.0, 30.0])
        dirty = trimesh.util.concatenate([box, chunk])
        out, rep = repair_solid(dirty)
        assert rep.components_dropped == 0
        assert out.volume == pytest.approx(8008.0, rel=1e-4)


@pytest.mark.skipif(not _STL.exists(), reason="fixture not present")
class TestBracketCorrection:
    def test_apply_corrections_single_fuse_is_watertight(self):
        mesh, plan = _full()
        solid = plan_to_mesh(plan)
        corrected = apply_corrections(solid, mesh,
                                      plan_corrections(plan, mesh))
        assert corrected is not None
        assert corrected.is_watertight
        assert corrected.is_winding_consistent
        assert corrected.volume == pytest.approx(mesh.volume, rel=0.01)

    def test_correction_mesh_repaired_to_single_manifold(self):
        mesh, plan = _full()
        solid = plan_to_mesh(plan)
        corrected = apply_corrections(solid, mesh,
                                      plan_corrections(plan, mesh))
        repaired, rep = repair_solid(corrected)
        # the boolean debris was found and removed, and the result is
        # exactly what a BRep shell needs: one watertight manifold solid
        assert rep.components_found >= 2
        assert rep.components_dropped >= 1
        assert repaired.is_watertight
        assert repaired.is_winding_consistent
        assert _nonmanifold_edges(repaired) == 0
        assert len(repaired.split(only_watertight=False)) == 1
        # and no real material was lost: within 1% of the source mesh
        # (and of the pre-repair corrected mesh)
        assert repaired.volume == pytest.approx(mesh.volume, rel=0.01)
        assert repaired.volume == pytest.approx(corrected.volume, rel=0.01)


# --------------------------------------------------------------------------
# _shape_from_mesh (stubbed FreeCAD/Part, adapter math only)
# --------------------------------------------------------------------------

class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def isEqual(self, other, tol):
        return (abs(self.x - other.x) <= tol
                and abs(self.y - other.y) <= tol
                and abs(self.z - other.z) <= tol)


class FakeShell:
    def __init__(self, faces, sewn_from=None):
        self.faces = faces
        self.sewn_from = sewn_from
        self.Volume = 0.0
        self.fix_calls = 0

    def isValid(self):
        return False

    def sewShape(self, tol):
        return FakeShell(self.faces, sewn_from=self)

    def fix(self, *args):
        self.fix_calls += 1


class FakeSolid:
    def __init__(self, volume, valid=True):
        self.Volume = volume
        self._valid = valid
        self.fix_calls = 0

    def isValid(self):
        return self._valid

    def fix(self, *args):
        self.fix_calls += 1


class FakeConsole:
    def __init__(self):
        self.warnings = []
        self.messages = []

    def PrintWarning(self, msg):
        self.warnings.append(msg)

    def PrintMessage(self, msg):
        self.messages.append(msg)


def _install_build_stubs(monkeypatch, solid_behavior, console=None):
    """Import build.py against stubbed FreeCAD/Part modules.

    ``solid_behavior(shell)`` is called for each Part.Solid attempt and
    returns a FakeSolid or raises -- scripted per test.
    """
    console = console if console is not None else FakeConsole()
    fake_app = types.ModuleType("FreeCAD")
    fake_app.Vector = FakeVector
    fake_app.Console = console
    fake_app.GuiUp = False

    fake_part = types.ModuleType("Part")
    fake_part.Face = lambda wire: wire
    fake_part.makePolygon = lambda pts: list(pts)
    fake_part.makeShell = lambda faces: FakeShell(list(faces))
    fake_part.Solid = solid_behavior

    monkeypatch.setitem(sys.modules, "FreeCAD", fake_app)
    monkeypatch.setitem(sys.modules, "Part", fake_part)
    sys.modules.pop("freecad.meshtofeatures_wb.build", None)
    import freecad.meshtofeatures_wb.build as build
    return build, console


class TestShapeFromMesh:
    def test_debris_repaired_before_conversion(self, monkeypatch):
        box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
        slab = trimesh.creation.box(extents=[20.0, 20.0, 1e-4])
        slab.apply_translation([0.0, 0.0, 10.0 + 5e-5])
        dirty = trimesh.util.concatenate([box, slab])
        captured = {}

        def solid(shell):
            captured["shell"] = shell
            return FakeSolid(volume=8000.0)

        build, console = _install_build_stubs(monkeypatch, solid)
        shape = build._shape_from_mesh(dirty)
        # the sliver's faces never reach the shell: repair ran first
        assert isinstance(shape, FakeSolid)
        assert len(captured["shell"].faces) == len(box.faces)
        assert not console.warnings          # volumes agree: no gate hit
        assert shape.Volume == pytest.approx(8000.0)

    def test_gate_warns_on_volume_mismatch(self, monkeypatch):
        box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])      # vol 8

        def solid(shell):
            if shell.sewn_from is None:
                raise RuntimeError("Creation of solid failed")
            # the sewn retry "succeeds" but with a 10%-under volume
            return FakeSolid(volume=7.2)

        build, console = _install_build_stubs(monkeypatch, solid)
        build._shape_from_mesh(box)
        assert len(console.warnings) == 1
        assert "volume gate" in console.warnings[0]
        assert "7.2" in console.warnings[0]

    def test_gate_silent_when_volumes_agree(self, monkeypatch):
        box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])      # vol 8

        def solid(shell):
            return FakeSolid(volume=8.0 * 0.9995)

        build, console = _install_build_stubs(monkeypatch, solid)
        shape = build._shape_from_mesh(box)
        assert isinstance(shape, FakeSolid)
        assert not console.warnings

    def test_solid_failure_ladders_through_sew_and_fix(self, monkeypatch):
        box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])      # vol 8
        attempts = {"n": 0}

        def solid(shell):
            attempts["n"] += 1
            raise RuntimeError("Creation of solid failed")

        build, console = _install_build_stubs(monkeypatch, solid)
        shape = build._shape_from_mesh(box)
        # raw shell attempt + sewed retry, then fix() as last resort
        assert attempts["n"] == 2
        assert isinstance(shape, FakeShell)
        assert shape.fix_calls == 1
        # the raw shell integrates 0 -> the gate must have fired loudly
        assert len(console.warnings) == 1
        assert "volume gate" in console.warnings[0]
