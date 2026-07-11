# SPDX-License-Identifier: LGPL-2.1-or-later
"""Pattern detection: circular, linear, and grid arrangements of features.

Designers create *patterns* -- a bolt circle is one decision, not six.
This module groups same-spec features (equal kind, diameter, depth,
through-flag, and axis direction) and tests whether their positions form:

* a CIRCULAR pattern -- on a common circle with uniform angular pitch
  (partial arcs allowed: all gaps equal except at most the wrap),
* a GRID -- the cartesian product of two uniformly spaced coordinate
  sets in the positions' principal frame,
* a LINEAR pattern -- collinear with uniform pitch.

A spec-group must match *wholly*; subset mining (a bolt circle plus two
stray holes of the same size) is deliberately out of scope for now and
such groups stay ungrouped rather than half-guessed. Positions come from
the snapped feature anchors, so tolerances can be tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import Feature, FeatureReport
from .fitting import _axis_frame, _fit_circle_2d
from .snapping import cluster_scalars

__all__ = ["Pattern", "PatternReport", "detect_patterns"]

_PATTERNABLE = ("hole", "counterbore", "boss")


@dataclass
class Pattern:
    members: list[Feature]
    params: dict
    description: str
    surface_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.surface_indices:
            # deduplicated union: members legitimately share interface
            # planes (every hole of a bolt circle opens onto the same
            # plate faces)
            seen: set[int] = set()
            for m in self.members:
                for i in m.surface_indices:
                    if i not in seen:
                        seen.add(i)
                        self.surface_indices.append(i)


@dataclass
class PatternReport:
    patterns: list[Pattern] = field(default_factory=list)
    #: patternable features that did not form a pattern
    ungrouped: list[Feature] = field(default_factory=list)


# --------------------------------------------------------------------------

_SPEC_SCALARS = ("diameter", "counterbore_diameter", "counterbore_depth",
                 "depth", "height")
_SPEC_RTOL = 2e-3


def _spec_compatible(a: Feature, b: Feature) -> bool:
    """Same manufacturing spec, judged with RELATIVE tolerance.

    Fixed decimal rounding split 9.12812 from 9.12811 (a 1e-6 relative
    difference -- the same drill) on real STL data; two scalars agree when
    within ``_SPEC_RTOL`` relative. Kind, through-flag, and axis direction
    must match exactly/angularly.
    """
    if a.kind != b.kind or a.params.get("through") != b.params.get("through"):
        return False
    ax = np.asarray(a.params["axis"], dtype=float)
    bx = np.asarray(b.params["axis"], dtype=float)
    if abs(float(ax @ bx)) < 1.0 - 1e-9:
        return False
    for name in _SPEC_SCALARS:
        va, vb = a.params.get(name), b.params.get(name)
        if (va is None) != (vb is None):
            return False
        if va is None:
            continue
        if abs(va - vb) > _SPEC_RTOL * max(abs(va), abs(vb), 1e-9):
            return False
    return True


def _positions_2d(members: list[Feature]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(members[0].params["axis"], dtype=float)
    u, v = _axis_frame(axis)
    pos = np.array([m.params["position"] for m in members], dtype=float)
    return np.column_stack([pos @ u, pos @ v]), u, v


def _try_circular(xy: np.ndarray, tol: float) -> dict | None:
    try:
        center, radius = _fit_circle_2d(xy)
    except (ValueError, np.linalg.LinAlgError):
        return None
    rel = xy - center
    spread = float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
    # collinear points "fit" a huge circle exactly; a real bolt circle has
    # radius comparable to the position spread
    if radius > 2.0 * max(spread, tol) or radius < tol:
        return None
    if np.max(np.abs(np.linalg.norm(rel, axis=1) - radius)) > tol:
        return None
    ang = np.sort(np.arctan2(rel[:, 1], rel[:, 0]))
    gaps = np.diff(ang, append=ang[0] + 2 * np.pi)
    order = np.sort(gaps)
    pitch = float(np.median(order[:-1])) if len(gaps) > 1 else float(gaps[0])
    ang_tol = max(tol / radius, 1e-4)
    uniform = np.all(np.abs(order[:-1] - pitch) < ang_tol)
    wrap_ok = abs(order[-1] - pitch) < ang_tol \
        or abs(order[-1] - (2 * np.pi - pitch * (len(xy) - 1))) < ang_tol
    if not (uniform and wrap_ok):
        return None
    return {"pattern": "circular", "count": len(xy),
            "bolt_circle_diameter": 2.0 * radius,
            "center_2d": center, "angular_pitch_deg": float(np.rad2deg(pitch))}


def _try_linear(xy: np.ndarray, tol: float) -> dict | None:
    c = xy.mean(axis=0)
    _, sv, vt = np.linalg.svd(xy - c, full_matrices=False)
    if len(sv) > 1 and sv[1] > tol * np.sqrt(len(xy)):
        return None                            # not collinear
    t = np.sort((xy - c) @ vt[0])
    steps = np.diff(t)
    if len(steps) == 0 or np.max(np.abs(steps - steps.mean())) > tol:
        return None
    return {"pattern": "linear", "count": len(xy),
            "pitch": float(steps.mean()), "direction_2d": vt[0]}


def _try_grid(xy: np.ndarray, tol: float) -> dict | None:
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    ab = (xy - c) @ vt.T                       # principal frame
    out_axes = []
    for col in (0, 1):
        labels = cluster_scalars(ab[:, col], atol=tol)
        levels = np.sort([ab[labels == k, col].mean()
                          for k in np.unique(labels)])
        if len(levels) < 2:
            return None
        steps = np.diff(levels)
        if np.max(np.abs(steps - steps.mean())) > tol:
            return None
        out_axes.append(levels)
    la, lb = out_axes
    if len(la) * len(lb) != len(xy):
        return None
    # every lattice site occupied
    grid = np.array([(a, b) for a in la for b in lb])
    for site in grid:
        if np.min(np.linalg.norm(ab - site, axis=1)) > 2 * tol:
            return None
    return {"pattern": "grid", "count": len(xy),
            "shape": [len(la), len(lb)],
            "pitches": [float(np.diff(la).mean()), float(np.diff(lb).mean())]}


def detect_patterns(feats: FeatureReport, tol: float | None = None) -> PatternReport:
    out = PatternReport()
    candidates = [f for f in feats.features if f.kind in _PATTERNABLE
                  and "axis" in f.params and "position" in f.params]

    groups: list[list[Feature]] = []
    for f in candidates:
        for g in groups:
            if _spec_compatible(g[0], f):
                g.append(f)
                break
        else:
            groups.append([f])

    for members in groups:
        if len(members) < 3:
            out.ungrouped.extend(members)
            continue
        xy, _, _ = _positions_2d(members)
        spread = float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
        eff_tol = tol if tol is not None else max(1e-3 * spread, 1e-6)

        hit = (_try_circular(xy, eff_tol)
               or _try_grid(xy, eff_tol)
               or _try_linear(xy, eff_tol))
        if hit is None:
            out.ungrouped.extend(members)
            continue
        m0 = members[0]
        hit["member"] = dict(m0.params)
        if hit["pattern"] == "circular":
            u, v = _axis_frame(np.asarray(m0.params["axis"], dtype=float))
            c2 = hit.pop("center_2d")
            hit["center"] = (c2[0] * u + c2[1] * v).tolist()
        name = {"circular": f"on BCD {hit.get('bolt_circle_diameter', 0):g}",
                "linear": f"pitch {hit.get('pitch', 0):g}",
                "grid": (f"grid {hit['shape'][0]}x{hit['shape'][1]}"
                         if hit["pattern"] == "grid" else "")}[hit["pattern"]]
        out.patterns.append(Pattern(
            members=members, params=hit,
            description=(f"{hit['count']}x {m0.description} "
                         f"({hit['pattern']} pattern, {name})")))
    return out
