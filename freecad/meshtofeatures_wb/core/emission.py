# SPDX-License-Identifier: LGPL-2.1-or-later
"""Emission planning: turn recognized (infinite) primitives into *bounded*
patch specifications a CAD kernel can realize as faces.

This module is pure numpy/scipy so the geometry of what gets emitted is
fully testable without FreeCAD. The FreeCAD adapter consumes
:class:`PatchSpec` objects and maps them 1:1 onto ``Part`` surfaces:

* plane    -> planar face over ``polygon`` (convex hull in the (x,y) frame)
* cylinder -> ``Part.Cylinder`` face, u in ``u_range``, v in ``v_range``
              (height above ``origin`` along ``z_dir``)
* cone     -> ``Part.Cone`` face; ``v_range`` is height above the apex
              along the axis (the OCC cone parameter is *slant* distance:
              divide by cos(half_angle) when realizing)
* sphere   -> ``Part.Sphere`` face, u = azimuth in ``u_range``,
              v = elevation in ``v_range`` (radians, in [-pi/2, pi/2])

Frames are right-handed orthonormal triads ``(x_dir, y_dir, z_dir)`` with
``z_dir`` the primitive's natural direction (plane normal, cylinder/cone
axis, spherical cap mean direction). Azimuth angles are measured from
``x_dir`` towards ``y_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .fitting import _axis_frame
from .pipeline import ReconstructionReport
from .primitives import Plane, Sphere, Cylinder, Cone, Primitive

__all__ = ["PatchSpec", "plan_patches"]

#: a revolution is "full" when the largest angular gap between adjacent
#: sample angles is below this (radians); tessellations of >= 12 sections
#: leave gaps of <= 30 deg
_FULL_REVOLUTION_GAP = np.deg2rad(45.0)


@dataclass
class PatchSpec:
    kind: str                       # 'plane' | 'cylinder' | 'cone' | 'sphere'
    primitive: Primitive
    origin: np.ndarray              # frame origin (see module docstring)
    x_dir: np.ndarray
    y_dir: np.ndarray
    z_dir: np.ndarray
    u_range: tuple[float, float]    # azimuth range; (0, 2*pi) when full_u
    full_u: bool
    v_range: tuple[float, float]    # see module docstring per kind
    polygon: np.ndarray | None      # (k,2) hull for planes, else None
    #: the segment points the patch must cover (kept for auditing/tests)
    primitive_points: np.ndarray
    label: str = ""
    #: interior hole loops for planes, each (k,2) in the (x,y) frame,
    #: clockwise (outer ``polygon`` is counter-clockwise)
    holes: list = field(default_factory=list)


def _frame(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = _axis_frame(z)
    z = z / np.linalg.norm(z)
    # _axis_frame guarantees cross(x, y) == z (right-handed)
    return x, y, z


def _angular_range(angles: np.ndarray) -> tuple[tuple[float, float], bool]:
    """Tight angular interval covering ``angles`` (radians, any values).

    Returns ``((u0, u1), full)`` with ``u1 > u0`` and ``u1 - u0 <= 2*pi``.
    The interval starts just after the largest gap in the circular
    distribution; when that gap is smaller than ``_FULL_REVOLUTION_GAP``
    the coverage is declared a full revolution.
    """
    a = np.sort(np.asarray(angles, dtype=float) % (2 * np.pi))
    gaps = np.diff(a, append=a[0] + 2 * np.pi)
    imax = int(np.argmax(gaps))
    if gaps[imax] < _FULL_REVOLUTION_GAP:
        return (0.0, 2 * np.pi), True
    u0 = a[(imax + 1) % len(a)] if imax + 1 < len(a) else a[0]
    span = 2 * np.pi - gaps[imax]
    return (float(u0), float(u0 + span)), False


def _signed_area(loop2d: np.ndarray) -> float:
    x, y = loop2d[:, 0], loop2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _plane_patch(prim: Plane, pts: np.ndarray,
                 boundary_loops: list[np.ndarray] | None = None,
                 outward: np.ndarray | None = None) -> PatchSpec:
    # orient the frame along the *outward* face normal (not the fitted
    # normal, whose sign is arbitrary) so boundary-loop winding is
    # predictable: outer CCW, holes CW
    normal = prim.normal
    if outward is not None and normal @ outward < 0:
        normal = -normal
    x, y, z = _frame(normal)
    origin = pts.mean(axis=0)
    # keep the origin exactly on the plane
    origin = origin - ((origin - prim.point) @ z) * z

    polygon = None
    holes: list[np.ndarray] = []
    if boundary_loops:
        loops2d = [np.column_stack([(lp - origin) @ x, (lp - origin) @ y])
                   for lp in boundary_loops if len(lp) >= 3]
        loops2d.sort(key=lambda l: abs(_signed_area(l)), reverse=True)
        if loops2d:
            outer = loops2d[0]
            if _signed_area(outer) < 0:      # winding/frame mismatch guard
                loops2d = [lp[::-1] for lp in loops2d]
            polygon = loops2d[0]
            for lp in loops2d[1:]:
                holes.append(lp[::-1] if _signed_area(lp) > 0 else lp)

    if polygon is None:                       # fallback: convex hull
        rel = pts - origin
        uv = np.column_stack([rel @ x, rel @ y])
        try:
            hull = ConvexHull(uv)
            polygon = uv[hull.vertices]
        except QhullError:
            lo, hi = uv.min(axis=0), uv.max(axis=0)
            polygon = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                                [hi[0], hi[1]], [lo[0], hi[1]]])

    return PatchSpec(
        kind="plane", primitive=prim, origin=origin,
        x_dir=x, y_dir=y, z_dir=z,
        u_range=(0.0, 0.0), full_u=False, v_range=(0.0, 0.0),
        polygon=polygon, primitive_points=pts,
        label=f"Plane n={np.round(z, 4)}" + (f" ({len(holes)} holes)" if holes else ""),
        holes=holes,
    )


def _cylinder_patch(prim: Cylinder, pts: np.ndarray) -> PatchSpec:
    x, y, z = _frame(prim.axis)
    rel = pts - prim.point
    h = rel @ z
    origin = prim.point + h.min() * z
    v_range = (0.0, float(h.max() - h.min()))
    angles = np.arctan2(rel @ y, rel @ x)
    u_range, full = _angular_range(angles)
    return PatchSpec(
        kind="cylinder", primitive=prim, origin=origin,
        x_dir=x, y_dir=y, z_dir=z,
        u_range=u_range, full_u=full, v_range=v_range,
        polygon=None, primitive_points=pts,
        label=f"Cylinder r={prim.radius:g}",
    )


def _cone_patch(prim: Cone, pts: np.ndarray) -> PatchSpec:
    x, y, z = _frame(prim.axis)
    rel = pts - prim.apex
    h = rel @ z
    v_range = (float(max(h.min(), 0.0)), float(h.max()))
    angles = np.arctan2(rel @ y, rel @ x)
    u_range, full = _angular_range(angles)
    return PatchSpec(
        kind="cone", primitive=prim, origin=prim.apex.copy(),
        x_dir=x, y_dir=y, z_dir=z,
        u_range=u_range, full_u=full, v_range=v_range,
        polygon=None, primitive_points=pts,
        label=f"Cone {np.rad2deg(prim.half_angle):.1f} deg",
    )


def _sphere_patch(prim: Sphere, pts: np.ndarray) -> PatchSpec:
    rel = pts - prim.center
    dirs = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    mean = dirs.mean(axis=0)
    n = np.linalg.norm(mean)
    zdir = mean / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    x, y, z = _frame(zdir)
    elev = np.arcsin(np.clip(dirs @ z, -1.0, 1.0))
    v_range = (float(elev.min()), float(elev.max()))
    angles = np.arctan2(dirs @ y, dirs @ x)
    # near the pole azimuths are meaningless; only use off-pole points
    off_pole = np.abs(dirs @ z) < 0.999
    if off_pole.any():
        u_range, full = _angular_range(angles[off_pole])
    else:
        u_range, full = (0.0, 2 * np.pi), True
    return PatchSpec(
        kind="sphere", primitive=prim, origin=prim.center.copy(),
        x_dir=x, y_dir=y, z_dir=z,
        u_range=u_range, full_u=full, v_range=v_range,
        polygon=None, primitive_points=pts,
        label=f"Sphere r={prim.radius:g}",
    )


def plan_patches(report: ReconstructionReport) -> list[PatchSpec]:
    """One bounded :class:`PatchSpec` per recognized surface, same order."""
    patches: list[PatchSpec] = []
    for surf in report.surfaces:
        prim = surf.fit.primitive
        pts = surf.segment.points
        if isinstance(prim, Plane):
            patches.append(_plane_patch(
                prim, pts,
                boundary_loops=surf.segment.boundary_loops,
                outward=surf.segment.face_normals.mean(axis=0)))
        elif isinstance(prim, Cylinder):
            patches.append(_cylinder_patch(prim, pts))
        elif isinstance(prim, Cone):
            patches.append(_cone_patch(prim, pts))
        elif isinstance(prim, Sphere):
            patches.append(_sphere_patch(prim, pts))
    return patches
