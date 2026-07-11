# SPDX-License-Identifier: LGPL-2.1-or-later
"""Pre-publication rigor suite: dimensions no prior suite touched.

* SCALE EXTREMES -- the same part at 1000x and 0.01x must reconstruct
  identically; failures reveal hidden absolute epsilons.
* HOSTILE INPUTS -- empty meshes, single faces, loops with duplicated
  vertices: graceful results or documented exceptions, never crashes.
* PROPERTY CONTRACTS -- snapping is a fixed point: snapping an already
  snapped report must change nothing.
* IN-PLANE ROTATION -- a slot at 30 degrees exercises every "is it
  axis-aligned?" shortcut in the loop/feature machinery.
* HONESTY UNDER AMBIGUITY -- two overlapping drills form a compound void
  that is NOT two holes; no rule may pretend otherwise.
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.features import detect_features
from meshtofeatures.history import SketchLine, loop_to_sketch, plan_history
from meshtofeatures.patterns import detect_patterns
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

pytest.importorskip("manifold3d")

SECTIONS = 64


def _counterbored(scale=1.0):
    plate = trimesh.creation.box(extents=[40.0, 30.0, 5.0])
    drill = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=SECTIONS)
    bore = trimesh.creation.cylinder(radius=6.0, height=4.0, sections=SECTIONS)
    bore.apply_translation([0.0, 0.0, 2.5])
    mesh = plate.difference(drill).difference(bore)
    mesh.apply_scale(scale)
    return mesh


class TestScaleInvariance:
    @pytest.mark.parametrize("scale", [1000.0, 0.01])
    def test_counterbore_inventory_and_parameters(self, scale):
        rep = snap_report(reconstruct(_counterbored(scale))).report
        assert rep.kinds() == ["cylinder"] * 2 + ["plane"] * 7
        assert rep.coverage == 1.0
        radii = sorted(s.fit.primitive.radius for s in rep.by_kind("cylinder"))
        assert np.allclose(radii, [4.0 * scale, 6.0 * scale], rtol=1e-6)

    @pytest.mark.parametrize("scale", [1000.0, 0.01])
    def test_features_and_plan_survive_scaling(self, scale):
        mesh = _counterbored(scale)
        rep = snap_report(reconstruct(mesh)).report
        patches = plan_patches(rep)
        feats = detect_features(rep, patches)
        assert len(feats.by_kind("counterbore")) == 1
        plan = plan_history(rep, feats, detect_patterns(feats), patches)
        assert np.isclose(plan.base.length, 5.0 * scale, rtol=1e-6)
        assert len(plan.holes) == 1


class TestHostileInputs:
    def test_empty_mesh_reconstructs_to_empty_report(self):
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)),
                               faces=np.zeros((0, 3), dtype=int))
        rep = reconstruct(mesh)
        assert rep.surfaces == [] and rep.unrecognized == []
        assert rep.coverage == 0.0

    def test_empty_report_snaps_to_empty(self):
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)),
                               faces=np.zeros((0, 3), dtype=int))
        result = snap_report(reconstruct(mesh))
        assert result.report.surfaces == []

    def test_plan_without_planes_raises_cleanly(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
        rep = snap_report(reconstruct(mesh)).report
        feats = detect_features(rep, plan_patches(rep))
        with pytest.raises(ValueError):
            plan_history(rep, feats, detect_patterns(feats))

    def test_loop_with_duplicated_vertex(self):
        loop = np.array([[0, 0], [10, 0], [10, 0], [10, 6], [0, 6.0]])
        prims = loop_to_sketch(loop)
        assert all(isinstance(p, SketchLine) for p in prims)
        assert len(prims) == 4
        for p in prims:
            assert np.linalg.norm(p.end - p.start) > 1e-9  # no zero-length junk


class TestSnapIdempotence:
    def test_snapping_is_a_fixed_point(self):
        rep = reconstruct(_counterbored())
        once = snap_report(rep)
        twice = snap_report(once.report)
        for a, b in zip(once.report.surfaces, twice.report.surfaces):
            pa, pb = a.fit.primitive, b.fit.primitive
            assert pa.kind == pb.kind
            for attr in ("radius", "point", "normal", "axis", "center",
                         "apex", "half_angle"):
                va, vb = getattr(pa, attr, None), getattr(pb, attr, None)
                if va is not None:
                    assert np.allclose(va, vb, atol=1e-12), (pa.kind, attr)


class TestInPlaneRotation:
    def test_rotated_slot_detected(self):
        plate = trimesh.creation.box(extents=[60.0, 40.0, 6.0])
        r = 4.0
        a = 11.0  # half-distance between end centers
        tool_parts = [trimesh.creation.box(extents=[2 * a, 2 * r, 20.0])]
        for sx in (-a, a):
            c = trimesh.creation.cylinder(radius=r, height=20.0,
                                          sections=SECTIONS)
            c.apply_translation([sx, 0, 0])
            tool_parts.append(c)
        tool = tool_parts[0]
        for p in tool_parts[1:]:
            tool = tool.union(p)
        rot = trimesh.transformations.rotation_matrix(np.deg2rad(30), [0, 0, 1])
        tool.apply_transform(rot)
        mesh = plate.difference(tool)
        rep = snap_report(reconstruct(mesh)).report
        feats = detect_features(rep, plan_patches(rep))
        slots = feats.by_kind("slot")
        assert len(slots) == 1
        assert np.isclose(slots[0].params["width"], 8.0, atol=0.05)
        assert np.isclose(slots[0].params["length"], 30.0, atol=0.1)


class TestHonestyUnderAmbiguity:
    def test_overlapping_drills_are_not_two_holes(self):
        plate = trimesh.creation.box(extents=[40.0, 30.0, 5.0])
        d1 = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=SECTIONS)
        d2 = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=SECTIONS)
        d2.apply_translation([5.0, 0.0, 0.0])   # overlapping: compound void
        mesh = plate.difference(d1).difference(d2)
        rep = snap_report(reconstruct(mesh)).report
        feats = detect_features(rep, plan_patches(rep))
        # the walls are partial cylinders: not full-revolution holes
        assert feats.by_kind("hole") == []
        # and nothing pretends the compound void is a slot either
        assert feats.by_kind("slot") == []


# ------------------------------------------------- sweep regressions

T_SHAFT = np.array([[-0.255163165864429, 0.9511846836462294, 0.17360718989392254, 7.661888296127683], [-0.8863163427458662, -0.15833582048152992, -0.43517020639519094, 6.813048744697662], [-0.386438998248927, -0.26491029717484244, 0.8834519993090189, 2.786466135971529], [0.0, 0.0, 0.0, 1.0]])
T_PLATE = np.array([[0.7780797083486832, -0.5917960904838828, 0.21064034453071967, 0.11071015299062315], [0.13150023140774483, -0.17444363073807417, -0.975846867512843, -1.7439958095471033], [0.6142472275849016, 0.7869859001169656, -0.05790972648313686, -1.242992907527527], [0.0, 0.0, 0.0, 1.0]])


class TestSweepRegressions:
    """Exact transforms from the pre-publication random sweep that
    exposed (1) pad side-blindness under frame-z flips and (2) position
    grid-snapping perturbing rotated (non-world-aligned) holes off their
    boundary loops."""

    def test_rotated_stepped_shaft_boss_side(self):
        from .test_composites import stepped_shaft
        from .test_adversarial import assert_geometry_match
        from meshtofeatures.history import plan_history
        mesh = stepped_shaft()
        mesh.apply_transform(T_SHAFT)
        rep = snap_report(reconstruct(mesh)).report
        patches = plan_patches(rep)
        feats = detect_features(rep, patches)
        plan = plan_history(rep, feats, detect_patterns(feats), patches)
        assert len(plan.pads) == 1
        assert_geometry_match(mesh, plan)

    def test_rotated_drilled_plate_keeps_its_hole(self):
        from .test_composites import drilled_plate
        from .test_adversarial import assert_geometry_match
        from meshtofeatures.history import plan_history
        mesh = drilled_plate()
        mesh.apply_transform(T_PLATE)
        rep = snap_report(reconstruct(mesh)).report
        patches = plan_patches(rep)
        feats = detect_features(rep, patches)
        assert len(feats.by_kind("hole")) == 1
        plan = plan_history(rep, feats, detect_patterns(feats), patches)
        assert len(plan.holes) == 1
        assert_geometry_match(mesh, plan)


T_FILLET = np.array([[ 9.3035642355119974e-01,  1.2701465810865398e-03,  -3.6665421296459066e-01,  2.0640217695101519e+00], [ 3.6664560319793044e-01,  4.4559114122376745e-03,   9.3035001290326869e-01, -3.6960954963345403e+00], [ 2.8154595799968762e-03, -9.9998926573296232e-01,   3.6798921457781505e-03,  7.1670580316716661e+00], [ 0.0000000000000000e+00,  0.0000000000000000e+00,   0.0000000000000000e+00,  1.0000000000000000e+00]])


class TestScaledRotatedFillets:
    """Sweep regression: a 25x-scaled, arbitrarily rotated filleted plate.

    Exposed that absolute paddings anywhere in the fillet chain fail at
    scale (fit noise in edge estimates scales with the part); the
    round-trip corner tools now pad relative to radius and span."""

    def test_geometry_roundtrip(self):
        from .test_fillet_ops import rounded_top_plate
        from .test_adversarial import assert_geometry_match
        from meshtofeatures.history import plan_history
        mesh = rounded_top_plate()
        mesh.apply_transform(T_FILLET)
        mesh.apply_scale(25.0)
        rep = snap_report(reconstruct(mesh)).report
        patches = plan_patches(rep)
        feats = detect_features(rep, patches)
        plan = plan_history(rep, feats, detect_patterns(feats), patches)
        assert len(plan.fillets) == 2
        assert_geometry_match(mesh, plan)
