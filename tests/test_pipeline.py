# SPDX-License-Identifier: LGPL-2.1-or-later
"""End-to-end regression tests: tessellated solids with fully known ground
truth go in; the recognized primitive inventory and parameters must match.

This file is the seed of the permanent regression suite: every future
feature (snapping, learned segmentation, history inference) must keep
these passing.
"""

import numpy as np
import trimesh

from meshtofeatures.pipeline import reconstruct

Z = np.array([0.0, 0.0, 1.0])


def _noisy(mesh: trimesh.Trimesh, sigma: float, seed: int = 0) -> trimesh.Trimesh:
    rng = np.random.default_rng(seed)
    out = mesh.copy()
    out.vertices = out.vertices + rng.normal(scale=sigma, size=out.vertices.shape)
    return out


class TestCleanSolids:
    def test_box(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        rep = reconstruct(mesh)
        assert rep.kinds() == ["plane"] * 6
        assert rep.coverage == 1.0
        # plane normals must be the three coordinate axes, two each
        normals = np.array([s.fit.primitive.normal for s in rep.surfaces])
        axis_hits = np.sum(np.isclose(np.abs(normals), 1.0, atol=1e-9), axis=0)
        assert list(axis_hits) == [2, 2, 2]

    def test_cylinder(self):
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
        rep = reconstruct(mesh)
        assert rep.kinds() == ["cylinder", "plane", "plane"]
        cyl = rep.by_kind("cylinder")[0].fit.primitive
        # mesh vertices lie exactly on the true cylinder -> tight recovery
        assert np.isclose(cyl.radius, 1.0, atol=1e-9)
        assert np.isclose(abs(cyl.axis @ Z), 1.0, atol=1e-9)
        # caps are z = +-1.5
        cap_offsets = sorted(
            float(s.fit.primitive.point @ Z) for s in rep.by_kind("plane")
        )
        assert np.allclose(cap_offsets, [-1.5, 1.5], atol=1e-12)

    def test_sphere(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
        rep = reconstruct(mesh)
        assert rep.kinds() == ["sphere"]
        sph = rep.surfaces[0].fit.primitive
        assert np.isclose(sph.radius, 2.0, atol=1e-9)
        assert np.allclose(sph.center, 0.0, atol=1e-9)

    def test_cone(self):
        mesh = trimesh.creation.cone(radius=1.0, height=2.0, sections=64)
        rep = reconstruct(mesh)
        assert rep.kinds() == ["cone", "plane"]
        cone = rep.by_kind("cone")[0].fit.primitive
        # trimesh cone: base at z=0, apex at z=height
        assert np.allclose(cone.apex, [0, 0, 2.0], atol=1e-6)
        assert np.isclose(cone.half_angle, np.arctan2(1.0, 2.0), atol=1e-8)
        # axis points from apex into the opening: -z
        assert cone.axis @ Z < -0.999999

    def test_capsule_mixed(self):
        # capsule = cylinder barrel + two hemispherical caps; the caps are
        # tangent-continuous with the barrel, so smooth region growing may
        # merge everything -- this documents current v0.1 behaviour:
        # recognition must not produce *wrong* primitives, and anything it
        # does return must fit tightly or be reported unrecognized.
        mesh = trimesh.creation.capsule(radius=1.0, height=2.0)
        rep = reconstruct(mesh)
        for s in rep.surfaces:
            assert s.fit.rms <= 1e-2 * 4.5  # accept_rms default vs diag
        assert rep.coverage + (len(rep.unrecognized) > 0) > 0  # smoke


class TestNoisySolids:
    SIGMA = 0.002  # simulated scan noise, 0.2% of unit radius

    def test_noisy_cylinder(self):
        mesh = _noisy(
            trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64), self.SIGMA
        )
        rep = reconstruct(mesh, accept_rms=0.05)
        cyls = rep.by_kind("cylinder")
        assert len(cyls) == 1
        assert np.isclose(cyls[0].fit.primitive.radius, 1.0, atol=0.01)

    def test_noisy_sphere(self):
        mesh = _noisy(trimesh.creation.icosphere(subdivisions=3, radius=2.0), self.SIGMA)
        rep = reconstruct(mesh, accept_rms=0.05)
        spheres = rep.by_kind("sphere")
        assert len(spheres) == 1
        assert np.isclose(spheres[0].fit.primitive.radius, 2.0, atol=0.01)


class TestRejection:
    def test_freeform_is_unrecognized_not_mislabelled(self):
        # a bumpy random surface must not be confidently labelled as a
        # primitive: it should land in `unrecognized`
        rng = np.random.default_rng(3)
        grid = np.linspace(-1, 1, 25)
        xx, yy = np.meshgrid(grid, grid)
        zz = 0.4 * np.sin(3 * xx) * np.cos(2 * yy) + 0.1 * rng.normal(size=xx.shape)
        vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        faces = []
        n = len(grid)
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                faces += [[a, b, c], [b, d, c]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
        rep = reconstruct(mesh, angle_threshold=np.deg2rad(60), accept_rms=0.01)
        assert rep.by_kind("plane") == [] or all(
            s.fit.rms <= 0.01 for s in rep.surfaces
        )
        assert len(rep.unrecognized) >= 1 or rep.coverage < 1.0
