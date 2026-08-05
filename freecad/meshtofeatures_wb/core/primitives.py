# SPDX-License-Identifier: LGPL-2.1-or-later
"""Analytic surface primitives and signed/unsigned distance functions.

All primitives implement ``distance(points) -> (n,) array`` giving the
unsigned distance from each point to the primitive *surface*. This single
interface drives least-squares refinement, residual scoring, model
selection, and the test harness.

Conventions
-----------
* All direction vectors (``normal``, ``axis``) are unit length.
* A primitive is orientation-agnostic: ``axis`` and ``-axis`` describe the
  same surface (except the cone, whose axis points from apex into the
  opening).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Plane", "Sphere", "Cylinder", "Cone", "Primitive"]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero-length direction vector")
    return v / n


@dataclass
class Plane:
    """Infinite plane through ``point`` with unit ``normal``."""

    point: np.ndarray
    normal: np.ndarray

    #: model complexity rank used by model selection (lower = simpler)
    complexity: int = field(default=0, init=False, repr=False)
    kind: str = field(default="plane", init=False)
    #: degrees of freedom of the surface parametrization
    dof: int = field(default=3, init=False, repr=False)

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float)
        self.normal = _unit(self.normal)

    def distance(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return np.abs((points - self.point) @ self.normal)


@dataclass
class Sphere:
    center: np.ndarray
    radius: float

    complexity: int = field(default=1, init=False, repr=False)
    kind: str = field(default="sphere", init=False)
    #: degrees of freedom of the surface parametrization
    dof: int = field(default=4, init=False, repr=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        self.radius = float(self.radius)
        if self.radius <= 0:
            raise ValueError("sphere radius must be positive")

    def distance(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return np.abs(np.linalg.norm(points - self.center, axis=1) - self.radius)


@dataclass
class Cylinder:
    """Infinite circular cylinder: ``point`` on axis, unit ``axis``, ``radius``."""

    point: np.ndarray
    axis: np.ndarray
    radius: float

    complexity: int = field(default=1, init=False, repr=False)
    kind: str = field(default="cylinder", init=False)
    #: degrees of freedom of the surface parametrization
    dof: int = field(default=5, init=False, repr=False)

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float)
        self.axis = _unit(self.axis)
        self.radius = float(self.radius)
        if self.radius <= 0:
            raise ValueError("cylinder radius must be positive")

    def radial_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from points to the axis line."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        v = points - self.point
        h = v @ self.axis
        return np.linalg.norm(v - np.outer(h, self.axis), axis=1)

    def distance(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.radial_distance(points) - self.radius)


@dataclass
class Cone:
    """Infinite one-sided cone.

    ``apex`` is the tip; unit ``axis`` points from the apex into the
    opening; ``half_angle`` (radians) is measured between axis and surface,
    in (0, pi/2).
    """

    apex: np.ndarray
    axis: np.ndarray
    half_angle: float

    complexity: int = field(default=2, init=False, repr=False)
    kind: str = field(default="cone", init=False)
    #: degrees of freedom of the surface parametrization
    dof: int = field(default=6, init=False, repr=False)

    def __post_init__(self) -> None:
        self.apex = np.asarray(self.apex, dtype=float)
        self.axis = _unit(self.axis)
        self.half_angle = float(self.half_angle)
        if not (0.0 < self.half_angle < np.pi / 2):
            raise ValueError("cone half-angle must be in (0, pi/2)")

    def distance(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        v = points - self.apex
        h = v @ self.axis                       # height along axis
        r = np.linalg.norm(v - np.outer(h, self.axis), axis=1)  # radial dist
        # Exact distance to the surface line in the (h, r) half-plane.
        # Surface direction in that plane: (cos a, sin a).
        ca, sa = np.cos(self.half_angle), np.sin(self.half_angle)
        # signed distance to the infinite surface line through origin
        d_line = np.abs(r * ca - h * sa)
        # points "behind" the apex project onto the apex itself
        t = h * ca + r * sa                     # parameter along surface line
        d_apex = np.sqrt(h * h + r * r)
        return np.where(t >= 0.0, d_line, d_apex)


Primitive = Plane | Sphere | Cylinder | Cone
