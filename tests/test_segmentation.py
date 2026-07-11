# SPDX-License-Identifier: LGPL-2.1-or-later
"""Segmentation tests: tessellated analytic solids have a known number of
smooth regions; region growing must find exactly those."""

import numpy as np
import trimesh

from meshtofeatures.segmentation import segment_mesh


def test_box_has_six_planar_regions():
    mesh = trimesh.creation.box(extents=[2, 3, 4])
    segments = segment_mesh(mesh)
    assert len(segments) == 6
    # each region of a box is 2 triangles
    assert all(len(s) == 2 for s in segments)


def test_cylinder_has_three_regions():
    mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
    segments = segment_mesh(mesh)
    # barrel + 2 caps
    assert len(segments) == 3
    # largest region by area is the barrel: 2*pi*r*h vs pi*r^2
    barrel = segments[0]
    assert np.isclose(barrel.area, 2 * np.pi * 1.0 * 3.0, rtol=0.02)


def test_sphere_is_one_region():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    segments = segment_mesh(mesh)
    assert len(segments) == 1


def test_cone_has_two_regions():
    mesh = trimesh.creation.cone(radius=1.0, height=2.0, sections=64)
    segments = segment_mesh(mesh)
    # lateral surface + base cap
    assert len(segments) == 2


def test_coarse_tessellation_needs_wider_threshold():
    # 8 sections -> 45 deg between adjacent barrel facets: the default
    # 30 deg threshold must split the barrel, a 50 deg threshold must not.
    mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=8)
    coarse = segment_mesh(mesh)  # default threshold
    assert len(coarse) > 3
    merged = segment_mesh(mesh, angle_threshold=np.deg2rad(50))
    assert len(merged) == 3


def test_segment_normals_are_paired_and_unit():
    mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
    for seg in segment_mesh(mesh):
        assert seg.points.shape == seg.normals.shape
        assert np.allclose(np.linalg.norm(seg.normals, axis=1), 1.0)


def test_barrel_vertex_normals_perpendicular_to_axis():
    # segment-restricted normals on the barrel (including rim vertices!)
    # must be perpendicular to the cylinder axis (z)
    mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=64)
    barrel = segment_mesh(mesh)[0]
    assert np.max(np.abs(barrel.normals @ np.array([0.0, 0.0, 1.0]))) < 1e-6


def test_empty_mesh():
    mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
    assert segment_mesh(mesh) == []
