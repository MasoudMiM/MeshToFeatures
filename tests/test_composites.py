# SPDX-License-Identifier: LGPL-2.1-or-later
"""Composite-part regression suite.

Real parts are booleans of primitives, and boolean geometry stresses the
pipeline differently from pristine trimesh primitives: retriangulated
regions, thin slivers along intersection curves, holes puncturing faces,
coaxial feature stacks. These fixtures have *fully known* ground truth --
every expected surface, radius, and hole loop is asserted.

This file is a permanent contract: every future change (curvature-aware
segmentation, learned components, history inference) must keep it green.

Fixtures (dimensions chosen as "nice" values so snapping must recover
them exactly):

* drilled plate     -- 40 x 30 x 5 plate, one 8mm-diameter through-hole
* counterbored hole -- same plate, 12mm counterbore of depth 2 on top
* stepped shaft     -- r=10 h=20 shaft with a coaxial r=6 h=15 step
* pocketed plate    -- plate with an open rectangular pocket

Requires the ``manifold3d`` boolean engine (dev dependency only).
"""

import numpy as np
import pytest
import trimesh

from meshtofeatures.emission import plan_patches
from meshtofeatures.pipeline import reconstruct
from meshtofeatures.snapping import snap_report

Z = np.array([0.0, 0.0, 1.0])
SECTIONS = 64

pytest.importorskip("manifold3d", reason="composite fixtures need a boolean engine")


# ------------------------------------------------------------------ fixtures

def drilled_plate() -> trimesh.Trimesh:
    plate = trimesh.creation.box(extents=[40.0, 30.0, 5.0])
    drill = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=SECTIONS)
    drill.apply_translation([10.0, 5.0, 0.0])
    return plate.difference(drill)


def counterbored_plate() -> trimesh.Trimesh:
    plate = trimesh.creation.box(extents=[40.0, 30.0, 5.0])   # z in [-2.5, 2.5]
    drill = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=SECTIONS)
    bore = trimesh.creation.cylinder(radius=6.0, height=4.0, sections=SECTIONS)
    bore.apply_translation([0.0, 0.0, 2.5])                    # z in [0.5, 4.5]
    return plate.difference(drill).difference(bore)


def stepped_shaft() -> trimesh.Trimesh:
    big = trimesh.creation.cylinder(radius=10.0, height=20.0, sections=SECTIONS)
    big.apply_translation([0.0, 0.0, 10.0])                    # z in [0, 20]
    small = trimesh.creation.cylinder(radius=6.0, height=15.0, sections=SECTIONS)
    small.apply_translation([0.0, 0.0, 27.5])                  # z in [20, 35]
    return big.union(small)


def pocketed_plate() -> trimesh.Trimesh:
    plate = trimesh.creation.box(extents=[40.0, 30.0, 10.0])   # z in [-5, 5]
    pocket = trimesh.creation.box(extents=[20.0, 12.0, 8.0])
    pocket.apply_translation([0.0, 0.0, 5.0])                  # z in [1, 9]: open at top
    return plate.difference(pocket)


def _run(mesh):
    result = snap_report(reconstruct(mesh))
    return result.report, plan_patches(result.report)


# ------------------------------------------------------------------ tests

class TestDrilledPlate:
    def test_surface_inventory(self):
        report, _ = _run(drilled_plate())
        assert report.kinds() == ["cylinder"] + ["plane"] * 6
        assert report.unrecognized == []
        assert report.coverage == 1.0

    def test_hole_parameters_exact(self):
        report, _ = _run(drilled_plate())
        cyl = report.by_kind("cylinder")[0].fit.primitive
        assert cyl.radius == 4.0
        assert abs(float(cyl.axis @ Z)) == 1.0
        # snapped hole position at (10, 5)
        anchor = cyl.point - (cyl.point @ cyl.axis) * cyl.axis
        assert np.allclose(anchor[:2], [10.0, 5.0])

    def test_top_and_bottom_planes_have_the_hole(self):
        _, patches = _run(drilled_plate())
        z_planes = [p for p in patches
                    if p.kind == "plane" and abs(p.z_dir @ Z) > 0.999]
        assert len(z_planes) == 2
        for p in z_planes:
            assert len(p.holes) == 1

    def test_side_planes_have_no_holes(self):
        _, patches = _run(drilled_plate())
        for p in patches:
            if p.kind == "plane" and abs(p.z_dir @ Z) < 0.001:
                assert p.holes == []


class TestCounterbore:
    def test_surface_inventory(self):
        report, _ = _run(counterbored_plate())
        # 6 box planes + annular shoulder plane; drill wall + bore wall
        assert report.kinds() == ["cylinder"] * 2 + ["plane"] * 7
        assert report.coverage == 1.0

    def test_radii_exact(self):
        report, _ = _run(counterbored_plate())
        radii = sorted(s.fit.primitive.radius for s in report.by_kind("cylinder"))
        assert radii == [4.0, 6.0]

    def test_bores_exactly_coaxial(self):
        report, _ = _run(counterbored_plate())
        small, big = sorted((s.fit.primitive for s in report.by_kind("cylinder")),
                            key=lambda c: c.radius)
        assert abs(float(small.axis @ big.axis)) == 1.0
        assert small.radial_distance([big.point])[0] < 1e-12

    def test_shoulder_is_an_annulus(self):
        _, patches = _run(counterbored_plate())
        # the shoulder: a z-normal plane whose outer loop is the r=6 circle
        # and whose single hole is the r=4 drill
        shoulders = []
        for p in patches:
            if p.kind == "plane" and abs(p.z_dir @ Z) > 0.999 and len(p.holes) == 1:
                outer_r = np.linalg.norm(p.polygon, axis=1).max()
                if outer_r < 8.0:  # the plate top/bottom outer loops are huge
                    shoulders.append((p, outer_r))
        assert len(shoulders) == 1
        p, outer_r = shoulders[0]
        hole_r = np.linalg.norm(p.holes[0] - p.holes[0].mean(axis=0), axis=1).mean()
        assert np.isclose(outer_r, 6.0, atol=0.05)
        assert np.isclose(hole_r, 4.0, atol=0.05)


class TestSteppedShaft:
    def test_surface_inventory(self):
        report, _ = _run(stepped_shaft())
        # 2 barrels; bottom cap, top cap, shoulder annulus
        assert report.kinds() == ["cylinder"] * 2 + ["plane"] * 3
        assert report.coverage == 1.0

    def test_step_radii_and_coaxiality(self):
        report, _ = _run(stepped_shaft())
        cyls = sorted((s.fit.primitive for s in report.by_kind("cylinder")),
                      key=lambda c: c.radius)
        assert [c.radius for c in cyls] == [6.0, 10.0]
        assert cyls[0].radial_distance([cyls[1].point])[0] < 1e-12

    def test_shoulder_annulus_hole(self):
        _, patches = _run(stepped_shaft())
        shoulder = [p for p in patches if p.kind == "plane" and len(p.holes) == 1]
        assert len(shoulder) == 1

    def test_barrel_heights(self):
        _, patches = _run(stepped_shaft())
        spans = sorted(p.v_range[1] - p.v_range[0]
                       for p in patches if p.kind == "cylinder")
        assert np.allclose(spans, [15.0, 20.0], atol=1e-6)


class TestPocketedPlate:
    def test_surface_inventory(self):
        report, _ = _run(pocketed_plate())
        # outer box: top (with pocket opening), bottom, 4 sides = 6
        # pocket: floor + 4 walls = 5
        assert report.kinds() == ["plane"] * 11
        assert report.coverage == 1.0

    def test_top_face_has_rectangular_opening(self):
        _, patches = _run(pocketed_plate())
        top = [p for p in patches
               if p.kind == "plane" and p.z_dir @ Z > 0.999
               and np.isclose((p.origin @ Z), 5.0, atol=1e-6)]
        assert len(top) == 1
        assert len(top[0].holes) == 1
        hole = top[0].holes[0]
        w = hole[:, 0].max() - hole[:, 0].min()
        h = hole[:, 1].max() - hole[:, 1].min()
        assert sorted([round(w, 6), round(h, 6)]) == [12.0, 20.0]

    def test_pocket_floor_depth(self):
        _, patches = _run(pocketed_plate())
        floors = [p for p in patches
                  if p.kind == "plane" and p.z_dir @ Z > 0.999
                  and np.isclose(p.origin @ Z, 1.0, atol=1e-6)]
        assert len(floors) == 1
        assert floors[0].holes == []


class TestArbitraryOrientation:
    """The pipeline must not silently rely on canonical axes: a rotated
    part has no snappable directions, yet inventory, radii (value grids
    are orientation-independent), relative geometry, and hole loops must
    all survive."""

    ROT = trimesh.transformations.rotation_matrix(0.83, [1.0, 2.0, 0.7],
                                                  point=[3.0, -2.0, 1.0])

    def test_rotated_counterbore(self):
        mesh = counterbored_plate()
        mesh.apply_transform(self.ROT)
        report, patches = _run(mesh)
        assert report.kinds() == ["cylinder"] * 2 + ["plane"] * 7
        assert report.coverage == 1.0
        radii = sorted(s.fit.primitive.radius for s in report.by_kind("cylinder"))
        assert radii == [4.0, 6.0]
        small, big = sorted((s.fit.primitive for s in report.by_kind("cylinder")),
                            key=lambda c: c.radius)
        # coaxiality is enforced by clustering, not by canonical snapping
        assert abs(float(small.axis @ big.axis)) > 1.0 - 1e-12
        assert small.radial_distance([big.point])[0] < 1e-9
        # hole loops survive rotation
        holed = [p for p in patches if p.kind == "plane" and len(p.holes) == 1]
        assert len(holed) == 3  # top, bottom, shoulder
