# SPDX-License-Identifier: LGPL-2.1-or-later
"""Design-intent recovery: snap fitted primitives to plausible exact values
and enforce geometric relations between them.

A raw fit says ``radius=0.99948, axis=(0.0031, -0.0002, 0.99999)``; the
designer almost certainly meant ``radius=1.0, axis=+Z``. This module
recovers that intent in ordered stages:

1. **direction unification** -- cluster all plane normals and cylinder /
   cone axes (sign-invariantly), replace each cluster with its weighted
   mean, and snap near-canonical means to exact +-X/+-Y/+-Z,
2. **coaxiality** -- among parallel axes, cluster the axis *lines* and
   merge lines closer than tolerance,
3. **value equalization** -- cluster radii and set each cluster to its
   weighted mean (two 8mm holes are the *same* 8mm),
4. **grid snapping** -- snap scalars (radii, plane offsets, positions) to
   the coarsest "nice" grid within tolerance, and cone half-angles to
   common angles,
5. **the guard** -- recompute each primitive's RMS on its segment's
   surface samples; any primitive whose fit degraded beyond
   ``max_extra_rms`` is reverted to the raw fit and the rejection logged.

The guard is what makes aggressive tolerances safe: a snap is a
*hypothesis about design intent*, and the mesh itself gets the veto.
Every decision is recorded as a :class:`SnapAction` so a UI can show the
user exactly what was inferred.

All functions are pure: the input report is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .fitting import FitResult
from .pipeline import RecognizedSurface, ReconstructionReport
from .primitives import Plane, Sphere, Cylinder, Cone, Primitive
from .segmentation import _UnionFind

__all__ = [
    "SnapConfig",
    "SnapAction",
    "SnapResult",
    "snap_value",
    "snap_angle",
    "snap_direction",
    "cluster_directions",
    "cluster_scalars",
    "snap_report",
]

_CANONICAL_AXES = np.array([
    [1.0, 0, 0], [-1.0, 0, 0],
    [0, 1.0, 0], [0, -1.0, 0],
    [0, 0, 1.0], [0, 0, -1.0],
])

_VALUE_GRIDS = (1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0,
                0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001)

_ANGLE_GRIDS = (np.pi / 4, np.pi / 6, np.pi / 12, np.pi / 36, np.pi / 180)


@dataclass
class SnapConfig:
    """Tolerances for snapping. ``None`` fields are resolved per report
    relative to its bounding-box diagonal."""

    angle_tol: float = np.deg2rad(1.0)
    value_atol: float | None = None       # default: 1e-3 * diagonal
    value_rtol: float = 1e-3
    max_extra_rms: float | None = None    # default: value_atol
    value_grids: tuple[float, ...] = _VALUE_GRIDS
    angle_grids: tuple[float, ...] = _ANGLE_GRIDS


@dataclass
class SnapAction:
    kind: str        # 'direction' | 'coaxial' | 'equal_value' | 'grid' | 'guard'
    detail: str
    accepted: bool


@dataclass
class SnapResult:
    report: ReconstructionReport
    actions: list[SnapAction] = field(default_factory=list)


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

def _standard_radius(radius: float, atol: float) -> float | None:
    """Nearest standard hole RADIUS within tolerance, or None."""
    from .standards import _CLEARANCE, _TAP
    best, best_err = None, atol
    for table in (_CLEARANCE, _TAP):
        for dia in table:
            err = abs(radius - dia / 2.0)
            if err <= best_err:
                best, best_err = dia / 2.0, err
    return best


def snap_value(x: float, atol: float,
               grids: tuple[float, ...] = _VALUE_GRIDS) -> float | None:
    """Snap ``x`` to the coarsest grid multiple within ``atol``; None if none.

    Grids finer than ``4 * atol`` are skipped: if the tolerance band covers
    a large fraction of the grid spacing, nearly *any* value would snap, so
    a match carries no evidence of design intent. Requiring
    ``g >= 4 * atol`` bounds the false-positive rate for a random value at
    ``2 * atol / g <= 50%`` -- every accepted snap is at least one bit of
    evidence.
    """
    for g in grids:
        if g < 4.0 * atol:
            continue
        s = round(x / g) * g
        if abs(x - s) <= atol:
            return float(s)
    return None


def snap_angle(a: float, atol: float,
               grids: tuple[float, ...] = _ANGLE_GRIDS) -> float | None:
    """Snap an angle (radians) to the coarsest common-angle grid within
    ``atol``; the same informativeness rule as :func:`snap_value` applies."""
    for g in grids:
        if g < 4.0 * atol:
            continue
        s = round(a / g) * g
        if abs(a - s) <= atol:
            return float(s)
    return None


def snap_direction(d: np.ndarray, angle_tol: float) -> np.ndarray | None:
    """Snap a unit vector to the nearest canonical axis within ``angle_tol``."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    dots = _CANONICAL_AXES @ d
    best = int(np.argmax(dots))
    if dots[best] >= np.cos(angle_tol):
        return _CANONICAL_AXES[best].copy()
    return None


def cluster_directions(
    dirs: np.ndarray, weights: np.ndarray, angle_tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sign-invariant clustering of unit vectors.

    Returns ``(labels, means)``: ``labels[i]`` in ``[0, k)`` and ``means``
    of shape ``(k, 3)`` holding weighted, unit-normalized cluster means.
    Anti-parallel vectors cluster together; each mean's sign follows the
    cluster's first member.
    """
    dirs = np.atleast_2d(np.asarray(dirs, dtype=float))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    n = len(dirs)
    uf = _UnionFind(n)
    cos_tol = np.cos(angle_tol)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dirs[i] @ dirs[j]) >= cos_tol:
                uf.union(i, j)
    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)

    means = np.zeros((len(unique_roots), 3))
    for k in range(len(unique_roots)):
        members = np.flatnonzero(labels == k)
        ref = dirs[members[0]]
        acc = np.zeros(3)
        for m in members:
            sgn = 1.0 if dirs[m] @ ref >= 0 else -1.0
            acc += weights[m] * sgn * dirs[m]
        means[k] = acc / np.linalg.norm(acc)
    return labels, means


def cluster_scalars(values: np.ndarray, atol: float) -> np.ndarray:
    """Cluster 1D values; pairs within ``atol`` merge (chains propagate)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    uf = _UnionFind(n)
    order = np.argsort(values)
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] <= atol:
            uf.union(int(a), int(b))
    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


# --------------------------------------------------------------------------
# report-level snapping
# --------------------------------------------------------------------------

def _line_anchor(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Point on the line closest to the origin (canonical line anchor)."""
    return point - (point @ direction) * direction


def _with_direction(prim: Primitive, new_dir: np.ndarray) -> Primitive:
    """Rebuild ``prim`` with a new direction, preserving orientation sign."""
    if isinstance(prim, Plane):
        d = new_dir if prim.normal @ new_dir >= 0 else -new_dir
        return Plane(point=prim.point.copy(), normal=d)
    if isinstance(prim, Cylinder):
        d = new_dir if prim.axis @ new_dir >= 0 else -new_dir
        return Cylinder(point=prim.point.copy(), axis=d, radius=prim.radius)
    if isinstance(prim, Cone):
        d = new_dir if prim.axis @ new_dir >= 0 else -new_dir
        return Cone(apex=prim.apex.copy(), axis=d, half_angle=prim.half_angle)
    return prim


def snap_report(report: ReconstructionReport,
                config: SnapConfig | None = None,
                progress=None) -> SnapResult:
    config = config or SnapConfig()
    actions: list[SnapAction] = []

    def _p(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, min(max(frac, 0.0), 1.0))

    _p("snapping directions", 0.0)

    surfaces = list(report.surfaces)
    if not surfaces:
        return SnapResult(report=ReconstructionReport(
            surfaces=[], unrecognized=list(report.unrecognized),
            coverage=report.coverage, mesh=report.mesh), actions=actions)

    # resolve tolerances relative to the model size
    all_pts = np.vstack([s.segment.points for s in surfaces])
    diag = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
    atol = config.value_atol if config.value_atol is not None else max(1e-3 * diag, 1e-12)
    max_extra = config.max_extra_rms if config.max_extra_rms is not None else atol

    prims: list[Primitive] = [s.fit.primitive for s in surfaces]
    orig_prims = list(prims)
    orig_rms = [
        float(np.sqrt(np.mean(p.distance(s.segment.samples) ** 2)))
        for p, s in zip(prims, surfaces)
    ]
    areas = np.array([s.segment.area for s in surfaces])

    def _dir_of(p: Primitive) -> np.ndarray | None:
        if isinstance(p, Plane):
            return p.normal
        if isinstance(p, (Cylinder, Cone)):
            return p.axis
        return None

    # ---- stage 1: direction unification ---------------------------------
    dir_idx = [i for i, p in enumerate(prims) if _dir_of(p) is not None]
    cluster_of: dict[int, int] = {}
    if dir_idx:
        dirs = np.array([_dir_of(prims[i]) for i in dir_idx])
        labels, means = cluster_directions(dirs, areas[dir_idx], config.angle_tol)
        for k in range(means.shape[0]):
            members = [dir_idx[m] for m in np.flatnonzero(labels == k)]
            target = means[k]
            canon = snap_direction(target, config.angle_tol)
            if canon is not None:
                target = canon
            for i in members:
                cluster_of[i] = k
                prims[i] = _with_direction(prims[i], target)
            if canon is not None or len(members) > 1:
                what = "canonical axis" if canon is not None else "shared direction"
                actions.append(SnapAction(
                    "direction",
                    f"surfaces {members}: unified to {what} {np.round(target, 6)}",
                    True))

    # ---- stage 2: coaxiality among parallel axes -------------------------
    axis_idx = [i for i, p in enumerate(prims) if isinstance(p, (Cylinder, Cone))]
    by_cluster: dict[int, list[int]] = {}
    for i in axis_idx:
        by_cluster.setdefault(cluster_of.get(i, -1 - i), []).append(i)
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        d = _dir_of(prims[members[0]])
        anchors = np.array([
            _line_anchor(prims[i].point if isinstance(prims[i], Cylinder)
                         else prims[i].apex, d)
            for i in members
        ])
        n = len(members)
        uf = _UnionFind(n)
        for a in range(n):
            for b in range(a + 1, n):
                if np.linalg.norm(anchors[a] - anchors[b]) <= atol:
                    uf.union(a, b)
        roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64)
        for root in np.unique(roots):
            grp = np.flatnonzero(roots == root)
            if len(grp) < 2:
                continue
            w = areas[[members[g] for g in grp]]
            common = np.average(anchors[grp], axis=0, weights=w)
            for g in grp:
                i = members[g]
                p = prims[i]
                if isinstance(p, Cylinder):
                    prims[i] = Cylinder(point=common, axis=p.axis, radius=p.radius)
                else:
                    h = (p.apex - common) @ p.axis
                    prims[i] = Cone(apex=common + h * p.axis, axis=p.axis,
                                    half_angle=p.half_angle)
            actions.append(SnapAction(
                "coaxial",
                f"surfaces {[members[g] for g in grp]}: merged onto common axis",
                True))

    # ---- stage 3 + 4a: radius equalization, then grid snap ---------------
    for kind, get_r, set_r in (
        (Cylinder, lambda p: p.radius, lambda p, r: Cylinder(point=p.point, axis=p.axis, radius=r)),
        (Sphere, lambda p: p.radius, lambda p, r: Sphere(center=p.center, radius=r)),
    ):
        idx = [i for i, p in enumerate(prims) if isinstance(p, kind)]
        if not idx:
            continue
        radii = np.array([get_r(prims[i]) for i in idx])
        eq_atol = max(atol, config.value_rtol * float(np.median(radii)))
        labels = cluster_scalars(radii, eq_atol)
        for k in np.unique(labels):
            grp = np.flatnonzero(labels == k)
            shared = float(np.average(radii[grp], weights=areas[[idx[g] for g in grp]]))
            tol_k = max(atol, config.value_rtol * shared)
            snapped = snap_value(shared, tol_k, config.value_grids)
            # standard hole sizes are snap AUTHORITIES: a d6.6 drill means
            # M6 clearance even though 3.3 sits on no informative grid;
            # the closer candidate wins, the guard still vets the result
            std = None
            if kind is Cylinder:
                std = _standard_radius(shared, tol_k)
            if std is not None and (snapped is None
                                    or abs(std - shared) <= abs(snapped - shared)):
                final = std
                if not np.isclose(std, shared, rtol=0, atol=0):
                    actions.append(SnapAction(
                        "standard",
                        f"surfaces {[idx[g] for g in grp]}: radius -> {std} "
                        f"(standard hole size)", True))
            else:
                final = snapped if snapped is not None else shared
            changed = [idx[g] for g in grp if not np.isclose(radii[g], final, rtol=0, atol=0)]
            for g in grp:
                i = idx[g]
                prims[i] = set_r(prims[i], final)
            if len(grp) > 1:
                actions.append(SnapAction(
                    "equal_value",
                    f"surfaces {[idx[g] for g in grp]}: radii equalized to {final}",
                    True))
            if snapped is not None and changed:
                actions.append(SnapAction(
                    "grid", f"surfaces {changed}: radius -> {final}", True))

    # ---- stage 4b: remaining scalar grid snaps ----------------------------
    # World-grid coordinates only encode design intent for WORLD-ALIGNED
    # entities: snapping the axis position of a hole in a rotated part
    # merely perturbs correct geometry within the guard band (and was
    # observed to push a hole's loop off its cylinder downstream).
    def _canonical(d) -> bool:
        return bool(np.max(np.abs(np.asarray(d))) > 1.0 - 1e-12)

    for i, p in enumerate(prims):
        if isinstance(p, Plane):
            if not _canonical(p.normal):
                continue
            off = float(p.point @ p.normal)
            s = snap_value(off, atol, config.value_grids)
            if s is not None and s != off:
                prims[i] = Plane(point=p.point + (s - off) * p.normal, normal=p.normal)
                actions.append(SnapAction("grid", f"surface {i}: plane offset -> {s}", True))
        elif isinstance(p, Cone):
            s = snap_angle(p.half_angle, config.angle_tol, config.angle_grids)
            if s is not None and s != p.half_angle and 0.0 < s < np.pi / 2:
                prims[i] = Cone(apex=p.apex, axis=p.axis, half_angle=s)
                actions.append(SnapAction(
                    "grid", f"surface {i}: half-angle -> {np.rad2deg(s):.1f} deg", True))

    def _snap_position(pos: np.ndarray) -> np.ndarray | None:
        out = pos.copy()
        hit = False
        for c in range(3):
            s = snap_value(float(pos[c]), atol, config.value_grids)
            if s is not None and s != pos[c]:
                out[c] = s
                hit = True
        return out if hit else None

    for i, p in enumerate(prims):
        if isinstance(p, Cylinder):
            if not _canonical(p.axis):
                continue
            anchor = _line_anchor(p.point, p.axis)
            s = _snap_position(anchor)
            if s is not None:
                prims[i] = Cylinder(point=s, axis=p.axis, radius=p.radius)
                actions.append(SnapAction("grid", f"surface {i}: axis position -> {s}", True))
        elif isinstance(p, Sphere):
            s = _snap_position(p.center)
            if s is not None:
                prims[i] = Sphere(center=s, radius=p.radius)
                actions.append(SnapAction("grid", f"surface {i}: center -> {s}", True))
        elif isinstance(p, Cone):
            if not _canonical(p.axis):
                continue
            s = _snap_position(p.apex)
            if s is not None:
                prims[i] = Cone(apex=s, axis=p.axis, half_angle=p.half_angle)
                actions.append(SnapAction("grid", f"surface {i}: apex -> {s}", True))

    _p("verifying snapped surfaces", 0.8)
    # ---- stage 5: the guard ----------------------------------------------
    new_surfaces: list[RecognizedSurface] = []
    for i, (surf, prim) in enumerate(zip(surfaces, prims)):
        d = prim.distance(surf.segment.samples)
        rms = float(np.sqrt(np.mean(d * d)))
        if rms > orig_rms[i] + max_extra:
            actions.append(SnapAction(
                "guard",
                f"surface {i} ({prim.kind}): snapped rms {rms:.3g} exceeds "
                f"{orig_rms[i]:.3g} + {max_extra:.3g}; reverted to raw fit",
                False))
            prim = orig_prims[i]
        new_surfaces.append(RecognizedSurface(
            segment=surf.segment,
            fit=FitResult.from_primitive(prim, surf.segment.samples)))

    _p("snapping done", 1.0)
    return SnapResult(
        report=ReconstructionReport(
            surfaces=new_surfaces,
            unrecognized=list(report.unrecognized),
            coverage=report.coverage,
            mesh=report.mesh,
        ),
        actions=actions,
    )
