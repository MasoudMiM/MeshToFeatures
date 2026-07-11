# SPDX-License-Identifier: LGPL-2.1-or-later
"""Least-squares fitting of analytic primitives to point sets.

Each ``fit_*`` function takes points (and, where useful, per-point unit
normals) and returns a fitted primitive. The strategy throughout is:

1. a *closed-form / linear* initial estimate (robust, no iteration), then
2. nonlinear refinement of the geometric (point-to-surface) distance with
   ``scipy.optimize.least_squares``.

Normals, when provided, are only used for direction estimation in the
initializers — final parameters always minimize geometric point distance,
so slightly inaccurate normals cannot bias the result.

``FitResult`` bundles the primitive with its residual statistics; ``fit_best``
performs tolerance-gated model selection preferring simpler primitives
(Occam's razor: a plane that explains the data wins over a giant sphere
that explains it equally well).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .primitives import Plane, Sphere, Cylinder, Cone, Primitive

__all__ = [
    "FitResult",
    "fit_plane",
    "fit_sphere",
    "fit_cylinder",
    "fit_cone",
    "fit_best",
]


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------

@dataclass
class FitResult:
    primitive: Primitive
    rms: float          # root-mean-square point-to-surface distance
    max_error: float    # worst point-to-surface distance

    @classmethod
    def from_primitive(cls, prim: Primitive, points: np.ndarray) -> "FitResult":
        d = prim.distance(points)
        return cls(primitive=prim, rms=float(np.sqrt(np.mean(d * d))),
                   max_error=float(d.max()))


def _trim_mask(distances: np.ndarray) -> np.ndarray | None:
    """Inlier mask via the MAD rule, or None when trimming is unwarranted.

    Points beyond ``median + 3.5 * 1.4826 * MAD`` are outliers. Trimming
    is skipped when the residuals are already tight (MAD ~ 0: exact data
    must never be perturbed), when nothing exceeds the cut, or when it
    would discard more than half the points (a fit that bad is a wrong
    model, not an outlier problem).
    """
    med = float(np.median(distances))
    mad = 1.4826 * float(np.median(np.abs(distances - med)))
    if mad <= 1e-12:
        return None
    mask = distances <= med + 3.5 * mad
    if mask.all() or mask.sum() < max(6, len(distances) // 2):
        return None
    return mask


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[1] != 3:
        raise ValueError(f"expected (n,3) points, got {pts.shape}")
    return pts


# --------------------------------------------------------------------------
# plane: exact closed form via PCA/SVD
# --------------------------------------------------------------------------

def fit_plane(points: np.ndarray, trim: bool = True) -> FitResult:
    pts = _as_points(points)
    centroid = pts.mean(axis=0)
    # normal = direction of least variance = last right singular vector
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    plane = Plane(point=centroid, normal=vt[-1])
    if trim:
        mask = _trim_mask(plane.distance(pts))
        if mask is not None:
            return fit_plane(pts[mask], trim=False)
    return FitResult.from_primitive(plane, pts)


# --------------------------------------------------------------------------
# sphere: algebraic linear least squares, then geometric refinement
# --------------------------------------------------------------------------

def fit_sphere(points: np.ndarray, trim: bool = True) -> FitResult:
    pts = _as_points(points)
    if len(pts) < 4:
        raise ValueError("sphere fit needs at least 4 points")

    # |p|^2 = 2 c.p + (r^2 - |c|^2)  ->  linear in (c, k)
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = np.einsum("ij,ij->i", pts, pts)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center, k = sol[:3], sol[3]
    r2 = k + center @ center
    if r2 <= 0:
        raise ValueError("degenerate sphere fit (non-positive radius)")
    radius = float(np.sqrt(r2))

    # geometric refinement (algebraic fit is biased for partial coverage)
    def residuals(x):
        return np.linalg.norm(pts - x[:3], axis=1) - x[3]

    res = least_squares(residuals, x0=[*center, radius], method="lm")
    c, r = res.x[:3], float(res.x[3])
    if r <= 0:
        raise ValueError("degenerate sphere fit after refinement")
    sphere = Sphere(center=c, radius=r)
    if trim:
        mask = _trim_mask(sphere.distance(pts))
        if mask is not None:
            return fit_sphere(pts[mask], trim=False)
    return FitResult.from_primitive(sphere, pts)


# --------------------------------------------------------------------------
# cylinder: axis from the Gauss map, circle in projection, refinement
# --------------------------------------------------------------------------

def _axis_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to ``axis``."""
    a = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(a[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return u, v


def _fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Kasa algebraic circle fit; returns (center(2,), radius)."""
    A = np.column_stack([2.0 * xy, np.ones(len(xy))])
    b = np.einsum("ij,ij->i", xy, xy)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center, k = sol[:2], sol[2]
    r2 = k + center @ center
    if r2 <= 0:
        raise ValueError("degenerate circle fit")
    return center, float(np.sqrt(r2))


def _spherical(axis: np.ndarray) -> tuple[float, float]:
    a = axis / np.linalg.norm(axis)
    theta = float(np.arccos(np.clip(a[2], -1.0, 1.0)))
    phi = float(np.arctan2(a[1], a[0]))
    return theta, phi


def _from_spherical(theta: float, phi: float) -> np.ndarray:
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def fit_cylinder(points: np.ndarray, normals: np.ndarray | None = None,
                 trim: bool = True) -> FitResult:
    pts = _as_points(points)
    if len(pts) < 6:
        raise ValueError("cylinder fit needs at least 6 points")

    # ---- initial axis ----
    if normals is not None and len(normals) > 0:
        nrm = np.atleast_2d(np.asarray(normals, dtype=float))
        # cylinder surface normals are all perpendicular to the axis:
        # the axis is the direction of least variance of the Gauss map.
        _, _, vt = np.linalg.svd(nrm - nrm.mean(axis=0) * 0.0, full_matrices=False)
        axis0 = vt[-1]
    else:
        # fall back: direction of *largest* point spread (works for elongated
        # segments; the refinement fixes moderate initialization error)
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
        axis0 = vt[0]

    # ---- initial center/radius from the projected circle ----
    u, v = _axis_frame(axis0)
    xy = np.column_stack([pts @ u, pts @ v])
    try:
        c2d, r0 = _fit_circle_2d(xy)
    except ValueError:
        c2d, r0 = xy.mean(axis=0), float(np.std(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
        r0 = max(r0, 1e-9)
    center0 = c2d[0] * u + c2d[1] * v  # point on axis (component along axis is free)

    # ---- nonlinear refinement over (theta, phi, cu, cv, r) ----
    theta0, phi0 = _spherical(axis0)

    def unpack(x):
        theta, phi, cu, cv, r = x
        axis = _from_spherical(theta, phi)
        uu, vv = _axis_frame(axis)
        point = cu * uu + cv * vv
        return point, axis, r

    def residuals(x):
        point, axis, r = unpack(x)
        w = pts - point
        h = w @ axis
        radial = np.linalg.norm(w - np.outer(h, axis), axis=1)
        return radial - r

    x0 = [theta0, phi0, center0 @ u, center0 @ v, r0]
    res = least_squares(residuals, x0=x0, method="lm")
    point, axis, r = unpack(res.x)
    if r <= 0:
        raise ValueError("degenerate cylinder fit (non-positive radius)")
    cyl = Cylinder(point=point, axis=axis, radius=float(r))
    if trim:
        mask = _trim_mask(cyl.distance(pts))
        if mask is not None:
            sub_n = None if normals is None else np.atleast_2d(normals)[mask] \
                if len(np.atleast_2d(normals)) == len(pts) else normals
            return fit_cylinder(pts[mask], sub_n, trim=False)
    return FitResult.from_primitive(cyl, pts)


# --------------------------------------------------------------------------
# cone: Gauss-map plane fit for (axis, half-angle), tangent-plane apex,
#       then refinement
# --------------------------------------------------------------------------

def fit_cone(points: np.ndarray, normals: np.ndarray | None = None,
             trim: bool = True) -> FitResult:
    pts = _as_points(points)
    if len(pts) < 6:
        raise ValueError("cone fit needs at least 6 points")

    if normals is not None and len(normals) > 0:
        nrm = np.atleast_2d(np.asarray(normals, dtype=float))
        # For an outward-oriented cone, every surface normal satisfies
        # n . axis = -sin(alpha) (a plane in the Gauss map). Solve
        # [N | -1] [axis; s] ~= 0 by SVD, normalize so |axis| = 1.
        M = np.column_stack([nrm, -np.ones(len(nrm))])
        _, _, vt = np.linalg.svd(M, full_matrices=False)
        sol = vt[-1]
        axis0 = sol[:3]
        norm_a = np.linalg.norm(axis0)
        if norm_a < 1e-12:
            raise ValueError("degenerate cone axis estimate")
        s = sol[3] / norm_a
        axis0 = axis0 / norm_a
        # orient the axis from apex into the opening: with outward normals
        # n . axis = -sin(alpha) < 0.
        if s > 0:
            axis0, s = -axis0, -s
        alpha0 = float(np.arcsin(np.clip(-s, 1e-3, 1 - 1e-6)))

        # Apex: every tangent plane passes through it: n_i . (apex - p_i) = 0
        b = np.einsum("ij,ij->i", nrm, pts)
        apex0, *_ = np.linalg.lstsq(nrm, b, rcond=None)
    else:
        raise ValueError("cone fitting currently requires normals for initialization")

    theta0, phi0 = _spherical(axis0)

    def unpack(x):
        ax, ay, az, theta, phi, alpha = x
        return np.array([ax, ay, az]), _from_spherical(theta, phi), alpha

    def residuals(x):
        apex, axis, alpha = unpack(x)
        v = pts - apex
        h = v @ axis
        radial = np.linalg.norm(v - np.outer(h, axis), axis=1)
        return radial * np.cos(alpha) - h * np.sin(alpha)

    x0 = [*apex0, theta0, phi0, alpha0]
    res = least_squares(residuals, x0=x0, method="lm")
    apex, axis, alpha = unpack(res.x)
    # normalize parametrization: keep half-angle in (0, pi/2)
    alpha = float(alpha)
    if alpha < 0:
        alpha, axis = -alpha, -axis
    if not (0.0 < alpha < np.pi / 2):
        raise ValueError(f"cone fit produced invalid half-angle {alpha}")
    cone = Cone(apex=apex, axis=axis, half_angle=alpha)
    if trim:
        mask = _trim_mask(cone.distance(pts))
        if mask is not None and normals is not None \
                and len(np.atleast_2d(normals)) == len(pts):
            return fit_cone(pts[mask], np.atleast_2d(normals)[mask], trim=False)
    return FitResult.from_primitive(cone, pts)


# --------------------------------------------------------------------------
# model selection
# --------------------------------------------------------------------------

def fit_best(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    tolerance: float | None = None,
    score_points: np.ndarray | None = None,
) -> FitResult:
    """Fit all primitive types and select the best model.

    Fitting uses ``points`` (typically mesh vertices: exact positions on
    the underlying surface). Candidate *ranking* uses ``score_points``
    when provided (typically dense face-interior samples): mesh vertices
    can occupy positions consistent with a wrong primitive (a cylinder
    barrel's two vertex rings lie exactly on a sphere), while
    face-interior samples expose the impostor.

    Selection rule: among candidates whose scoring RMS is below
    ``max(tolerance, 1.2 * best_rms)``, return the *simplest* (lowest
    complexity). The adaptive term keeps Occam's razor working in the
    presence of noise or tessellation chord error, where no candidate can
    reach an absolute tolerance but a plane within 20% of a giant
    sphere's residual is still the right answer. ``tolerance`` defaults to
    1e-4 of the point-set bounding-box diagonal.
    """
    pts = _as_points(points)
    score = pts if score_points is None else _as_points(score_points)
    if tolerance is None:
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        tolerance = max(1e-4 * diag, 1e-12)

    # nonlinear refinement cost scales with point count; a deterministic
    # subsample suffices for PARAMETER estimation (scoring below always
    # uses the full sample set, so selection quality is unaffected)
    fit_pts, fit_nrm = pts, normals
    if len(pts) > 600:
        step = int(np.ceil(len(pts) / 600))
        fit_pts = pts[::step]
        if normals is not None and len(np.atleast_2d(normals)) == len(pts):
            fit_nrm = np.atleast_2d(normals)[::step]

    candidates: list[FitResult] = []
    # complexity order enables short-circuiting: if a simpler model passes
    # the ABSOLUTE tolerance gate, no more complex candidate can win the
    # selection rule, so the (expensive) remaining fitters are skipped --
    # profiling showed fit_cone alone was ~75% of pipeline time, mostly
    # spent on segments that are trivially planes.
    fitters = (
        (0, lambda: fit_plane(fit_pts)),
        (1, lambda: fit_sphere(fit_pts)),
        (1, lambda: fit_cylinder(fit_pts, fit_nrm)),
        (2, lambda: fit_cone(fit_pts, fit_nrm)),
    )
    prev_complexity = -1
    for complexity, fitter in fitters:
        if candidates and complexity > prev_complexity                 and min(c.rms for c in candidates) <= tolerance:
            break
        prev_complexity = complexity
        try:
            fit = fitter()
        except (ValueError, np.linalg.LinAlgError):
            continue  # degenerate for this segment; skip
        # re-score on the surface-representative sample set
        candidates.append(FitResult.from_primitive(fit.primitive, score))

    if not candidates:
        raise ValueError("no primitive could be fitted to the segment")

    best_rms = min(c.rms for c in candidates)
    gate = max(tolerance, 1.2 * best_rms)
    passing = [c for c in candidates if c.rms <= gate]
    return min(passing, key=lambda c: (c.primitive.complexity, c.rms))
