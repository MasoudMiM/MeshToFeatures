# SPDX-License-Identifier: LGPL-2.1-or-later
"""Cross-section profiling at arbitrary z-heights for lofted feature
reconstruction. The prismatic model (one constant profile extruded) fails
when the part's cross-section changes with height -- a cavity that widens
as bosses recede, a tapered wall, a step. This module extracts the 2D
polygon at a given z via mesh-plane intersection; where polygonize fails
on coarse STL tessellations (broken contour loops), the cavity polygon is
recovered by subtracting the wall-ring from the known outer pentagon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import polygonize


@dataclass
class ZProfile:
    """A 2D cross-section polygon at a given z-height."""
    z: float
    polygon: Polygon
    area: float


def profile_at(
    mesh: trimesh.Trimesh, z: float, outer_pentagon: Polygon | None = None,
) -> ZProfile | None:
    """2D cavity/inner polygon at height ``z``.

    The mesh-plane intersection produces a ring (wall cross-section)
    whose exterior is the outer outline and, on clean tessellations,
    whose interior holes trace the cavity boundary. On coarse STL meshes
    the inner boundary is often broken; when ``outer_pentagon`` is
    provided the cavity is recovered as ``outer_pentagon`` minus ``ring``.
    """
    segs = trimesh.intersections.mesh_plane(mesh, [0.0, 0.0, 1.0],
                                            [0.0, 0.0, z])
    if len(segs) == 0:
        return None
    lines = MultiLineString([tuple(map(tuple, s)) for s in segs])
    try:
        polys = list(polygonize(lines))
    except Exception:                                      # noqa: BLE001
        return None
    if not polys:
        return None
    ring = max(polys, key=lambda p: p.area)
    if ring.interiors:
        cavity = Polygon(ring.interiors[0])
        return ZProfile(z=z, polygon=cavity, area=float(cavity.area))
    if outer_pentagon is not None:
        diff = outer_pentagon.difference(ring)
        if diff.is_empty:
            return None
        cavity = (max(diff.geoms, key=lambda g: g.area)
                  if hasattr(diff, "geoms") else diff)
        return ZProfile(z=z, polygon=cavity, area=float(cavity.area))
    return None


def profile_stack(
    mesh: trimesh.Trimesh, z_range: tuple[float, float],
    outer_pentagon: Polygon, n_slices: int = 10,
) -> list[ZProfile]:
    """Cross-sections of the cavity/inner region at evenly spaced z."""
    z0, z1 = z_range
    profiles: list[ZProfile] = []
    for z in np.linspace(z0, z1, n_slices):
        p = profile_at(mesh, float(z), outer_pentagon=outer_pentagon)
        if p is not None and p.area > 1e-6:
            profiles.append(p)
    return profiles


def area_spread(profiles: list[ZProfile]) -> float:
    """Max relative area difference between the first and last profile.
    Zero when the cavity has the same cross-section at every height
    (a true prismatic pocket -- no taper, no loft needed)."""
    if len(profiles) < 2:
        return 0.0
    a0, a1 = profiles[0].area, profiles[-1].area
    return float(abs(a1 - a0) / max(a0, a1, 1e-12))
