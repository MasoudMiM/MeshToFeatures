# SPDX-License-Identifier: LGPL-2.1-or-later
"""History planning: from recognized features to an ordered, executable
build plan (base pad -> pads -> pockets -> holes).

Pure and FreeCAD-free, like every planning layer in this project: the
plan's semantics are pinned by executing it with boolean operations in
the test suite and comparing volumes against the input mesh. The FreeCAD
executor merely transliterates the plan into a PartDesign Body.

Plan conventions
----------------
* frame: right-handed (x, y, z), z is the extrusion direction, the base
  bottom plane is z = 0, the base occupies z in [0, length],
* pads stack on the base top; pockets, counterbores, and blind holes cut
  downward from the top face; through holes pierce everything,
* all 2D coordinates (profiles, hole positions) live in the frame's
  (x, y).

Sketch primitives carry exact parameters (a fitted arc keeps its center
and radius) so emitted sketches are *editable geometry*, not polylines.
Vertical fillets are absorbed for free: the base profile comes from the
face's real boundary loop, whose arcs `loop_to_sketch` recovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .emission import PatchSpec
from .features import FeatureReport
from .fitting import _axis_frame, _fit_circle_2d
from .patterns import PatternReport
from .pipeline import ReconstructionReport
from .primitives import Cylinder, Plane

__all__ = ["SketchLine", "SketchArc", "SketchCircle", "loop_to_sketch",
           "hole_op_properties", "FilletOp", "fillet_edge_matches",
           "ChamferOp", "CrossHoleOp", "blend_corner_profile",
           "chamfer_corner_profile",
           "BasePad", "HoleOp", "PocketOp", "PadOp", "BuildPlan",
           "plan_history", "MAX_STEP_LEVELS"]

#: Above this many intermediate step levels a part is treated as too
#: complex for the single-axis prismatic model and declined (a faceted,
#: terraced, or organic export -- e.g. the field's box.STL at 97). No
#: ordinary machined prismatic part has this many distinct shelves.
MAX_STEP_LEVELS = 32


# --------------------------------------------------------------------------
# sketch primitives
# --------------------------------------------------------------------------

@dataclass
class SketchLine:
    start: np.ndarray
    end: np.ndarray

    def sample(self, n: int | None = None) -> list[np.ndarray]:
        return [np.asarray(self.start, dtype=float)]


@dataclass
class SketchArc:
    center: np.ndarray
    radius: float
    start: np.ndarray
    end: np.ndarray
    sweep: float  # signed sweep angle (radians), + is CCW

    def sample(self, n: int | None = None) -> list[np.ndarray]:
        if n is None:
            n = max(4, int(abs(self.sweep) / np.deg2rad(2.0)))
        a0 = np.arctan2(self.start[1] - self.center[1],
                        self.start[0] - self.center[0])
        ts = a0 + self.sweep * np.arange(n) / n
        return [self.center + self.radius * np.array([np.cos(t), np.sin(t)])
                for t in ts]


@dataclass
class SketchCircle:
    center: np.ndarray
    radius: float

    @property
    def start(self):
        return self.center + np.array([self.radius, 0.0])

    end = start

    def sample(self, n: int | None = None) -> list[np.ndarray]:
        n = n or 96
        ts = 2 * np.pi * np.arange(n) / n
        return [self.center + self.radius * np.array([np.cos(t), np.sin(t)])
                for t in ts]


# --------------------------------------------------------------------------
# loop -> lines / arcs / circle
# --------------------------------------------------------------------------

def _line_residual(pts: np.ndarray) -> float:
    d = pts[-1] - pts[0]
    n = np.linalg.norm(d)
    if n == 0:
        return np.inf
    nrm = np.array([-d[1], d[0]]) / n
    return float(np.max(np.abs((pts - pts[0]) @ nrm)))


def _circle_residual(pts: np.ndarray):
    """Circle fit residual, gated on *arc-like sampling*.

    Cocircularity alone is not evidence of an arc: any rectangle's four
    corners lie exactly on its circumcircle. A genuine tessellated arc
    has at least two interior turns, all of one sign, each a modest step
    (<= 60 deg); sharp corners flunk, arc facets pass.
    """
    if len(pts) < 4:
        return np.inf, None, None
    e = np.diff(pts, axis=0)
    turns = []
    for a, b in zip(e[:-1], e[1:]):
        turns.append(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
    turns = np.array(turns)
    at = np.abs(turns)
    if np.any(at > np.deg2rad(60.0)) or np.any(np.sign(turns) != np.sign(turns[0])):
        return np.inf, None, None
    # constant curvature: uniform turning per vertex. A straight side plus
    # a gentle tangent transition curls consistently and can sit on a huge
    # circle within tolerance -- but its turning is wildly non-uniform,
    # which is the tell.
    if at.min() < 1e-12 or at.max() / at.min() > 1.5:
        return np.inf, None, None
    try:
        c, r = _fit_circle_2d(pts)
    except (ValueError, np.linalg.LinAlgError):
        return np.inf, None, None
    res = float(np.max(np.abs(np.linalg.norm(pts - c, axis=1) - r)))
    return res, c, r


def _make_arc(pts: np.ndarray, c: np.ndarray, r: float) -> SketchArc:
    ang = np.unwrap(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return SketchArc(center=np.asarray(c), radius=float(r),
                     start=pts[0].copy(), end=pts[-1].copy(),
                     sweep=float(ang[-1] - ang[0]))


def loop_to_sketch(loop: np.ndarray, tol: float | None = None,
                   arcs: bool = True, max_sweep_deg: float | None = None,
                   full_circle_deg: float = 270.0) -> list:
    """Decompose a closed 2D polyline into lines, arcs, or a full circle.

    Greedy: grow the current run while a line (preferred) or a circle
    explains all its points within ``tol``; emit and restart at the
    breaking point. A wrap-around pass merges the first and last
    primitives when they continue each other (the loop's start vertex may
    fall mid-edge or mid-arc).
    """
    loop = np.asarray(loop, dtype=float)
    n = len(loop)
    diag = float(np.linalg.norm(loop.max(axis=0) - loop.min(axis=0)))
    if tol is None:
        tol = max(1e-4 * diag, 1e-9)
    # An arc in a sketch profile is a fillet/round -- small relative to the
    # profile. A run of straight edges whose corners happen to be concyclic
    # (any rectangle's are; issue #5's L-bracket footprint diagonal is
    # exactly cocircular with its two adjacent corners) also passes the
    # arc-like-sampling gate with uniform turns, fitting a circle whose
    # radius is a large fraction of the whole profile. Such a "curve" bows
    # far off the true straight edges and inflates the extruded profile, so
    # cap the radius and let the run fall back to lines.
    arc_radius_cap = 0.5 * diag

    def _decompose(pts_closed: np.ndarray, m: int) -> list:
        prims: list = []
        i = 0
        while i < m:
            j = i + 2
            while j <= m and _line_residual(pts_closed[i:j + 1]) <= tol:
                j += 1
            if j - i >= 3:
                prims.append(SketchLine(start=pts_closed[i].copy(),
                                        end=pts_closed[j - 1].copy()))
                i = j - 1
                continue
            k = i + 3
            res, c, r = (_circle_residual(pts_closed[i:k + 1])
                         if arcs and k <= m else (np.inf, None, None))
            if res <= tol and r is not None and r <= arc_radius_cap:
                while k + 1 <= m:
                    res2, c2, r2 = _circle_residual(pts_closed[i:k + 2])
                    if res2 > tol or r2 > arc_radius_cap:
                        break
                    k, c, r = k + 1, c2, r2
                # Sweep gate: fillets are tight (< max_sweep deg); full
                # circles (> full_circle deg) become SketchCircle later;
                # anything in between is a spurious fit (e.g. arc bridging
                # a diagonal corner across boss outlines -- 90-270 deg).
                if max_sweep_deg is not None:
                    ang = np.unwrap(np.arctan2(pts_closed[i:k + 1, 1] - c[1],
                                               pts_closed[i:k + 1, 0] - c[0]))
                    sw = abs(np.degrees(ang[-1] - ang[0]))
                    if max_sweep_deg < sw < full_circle_deg:
                        prims.append(SketchLine(
                            start=pts_closed[i].copy(),
                            end=pts_closed[i + 1].copy()))
                        i += 1
                        continue
                prims.append(_make_arc(pts_closed[i:k + 1], c, r))
                i = k
            else:
                prims.append(SketchLine(start=pts_closed[i].copy(),
                                        end=pts_closed[i + 1].copy()))
                i += 1
        return prims

    prims = _decompose(np.vstack([loop, loop[:1]]), n)

    # The loop's arbitrary start vertex may split one entity in two (a
    # mid-arc start yields a big arc plus an orphaned sliver). Rotating
    # the loop to begin at the FIRST detected primitive boundary -- a
    # genuine corner -- and decomposing once more makes the result
    # start-invariant.
    if len(prims) > 1:
        bp = int(np.argmin(np.linalg.norm(loop - prims[0].end, axis=1)))
        rotated = np.roll(loop, -bp, axis=0)
        prims = _decompose(np.vstack([rotated, rotated[:1]]), n)

    # full circle: one arc consuming the whole loop
    if len(prims) == 1 and isinstance(prims[0], SketchArc) \
            and abs(abs(prims[0].sweep) - 2 * np.pi) < 0.2:
        a = prims[0]
        return [SketchCircle(center=a.center, radius=a.radius)]
    return prims


# --------------------------------------------------------------------------
# plan data model
# --------------------------------------------------------------------------

@dataclass
class BasePad:
    profile: list
    length: float
    #: through-openings in the base face (a frame/ring's central hole),
    #: each a list of sketch primitives; the base prism is cut by these so
    #: it does not fill solid (design note 37). Only genuine through-holes
    #: (open at both z-ends) belong here -- a blind pocket mouth does not.
    hole_profiles: list = field(default_factory=list)


@dataclass
class HoleOp:
    diameter: float
    through: bool
    depth: float
    positions: list                 # [(x, y), ...] in the plan frame
    counterbore_diameter: float | None = None
    counterbore_depth: float | None = None
    #: countersink (conical entry): the mouth diameter and the included
    #: (full tip) angle in DEGREES. When BOTH a countersink and a
    #: counterbore are set the hole is a counterdrill -- a cylindrical
    #: recess with a conical transition down to the drill -- and
    #: ``counterbore_depth`` is the CYLINDRICAL part only.
    #: ``countersink_angle`` defaults the executor to 90 deg.
    countersink_diameter: float | None = None
    countersink_angle: float | None = None
    #: blind depth / counterbore measured from the top face (z = length)
    #: when True, from the bottom face (z = 0) when False
    from_top: bool = True
    label: str = ""
    #: frame-z (measured from frame_origin along frame_z) of the face the
    #: hole OPENS on -- the counterbore mouth / drill mouth. When None the
    #: executor falls back to the global top (z = length). Placing the hole
    #: sketch here, not at the global top, is what stops a counterbore whose
    #: mouth sits BELOW the tallest level (e.g. bores on a base plate under
    #: a raised deck) from being modelled as an OUTWARD tower by
    #: PartDesign::Hole (a sketch floating above the material extrudes the
    #: counterbore outward instead of cutting it inward).
    surface_z: float | None = None


@dataclass
class PocketOp:
    profile: list
    depth: float
    from_top: bool = True
    through: bool = False
    #: interior hole loops (each a list of sketch primitives): material
    #: INSIDE these stays -- a ring-shaped shelf's holes are where higher
    #: decks stand
    hole_profiles: list = field(default_factory=list)
    label: str = ""
    #: mouth profile (same shape/semantics as ``profile``): when the
    #: cavity cross-section CHANGES with height (tapered walls, receding
    #: bosses) the pocket is built as a SUBTRACTIVE LOFT between the
    #: floor profile and the mouth profile instead of a prismatic
    #: extrusion. ``None`` keeps the prismatic path for constant cavities.
    mouth_profile: list | None = None


@dataclass
class PadOp:
    profile: list
    length: float
    #: pads grow upward from the top face when True, downward from the
    #: bottom face when False (the frame z sign is arbitrary)
    from_top: bool = True
    label: str = ""
    #: LATERAL pads (design note 36): material protruding horizontally off
    #: a wall -- a mounting flange, side rail, gusseted bracket. When
    #: ``axis`` is set the pad extrudes along it (a plan-frame unit vector)
    #: instead of z, and ``profile`` lives in the plane perpendicular to
    #: ``axis`` spanned by (``plane_u``, ``plane_v``): a profile 2D point
    #: (u, v) maps to the 3D plan-frame point
    #: ``plane_origin + u*plane_u + v*plane_v``. Vertical pads leave these
    #: None and behave exactly as before.
    axis: np.ndarray | None = None
    plane_origin: np.ndarray | None = None
    plane_u: np.ndarray | None = None
    plane_v: np.ndarray | None = None


@dataclass
class FilletOp:
    """A fillet to apply on the rebuilt sharp solid's edge.

    ``edge_start``/``edge_end`` bound the SHARP edge the fillet replaces
    (world coordinates): the fillet's axis line shifted by
    ``+- r * (n_a + n_b)`` (outward blend-plane normals; '+' for convex).
    """
    radius: float
    edge_start: np.ndarray
    edge_end: np.ndarray
    direction: np.ndarray
    n_a: np.ndarray
    n_b: np.ndarray
    convex: bool = True
    label: str = ""


def fillet_edge_matches(op: FilletOp, p0: np.ndarray, p1: np.ndarray,
                        tol: float) -> bool:
    """Does the straight edge (p0, p1) lie on ``op``'s sharp-edge line?

    Pure geometric matching (no topological names): direction parallel,
    both endpoints within ``tol`` of the sharp edge's infinite LINE, and the
    edge overlapping the detected segment along that line. Sub-edges are
    accepted because boolean rebuilds may split an edge into pieces, and
    SUPER-edges are accepted because the detected segment (one blend
    cylinder's span) is often shorter than the rebuilt sharp edge it dresses
    -- measuring endpoint distance to the clamped segment would reject the
    overhang. A parallel edge on a different line stays rejected because its
    endpoints sit more than ``tol`` from the sharp edge's line.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    e = p1 - p0
    n = np.linalg.norm(e)
    if n < tol:
        return False
    if abs(float((e / n) @ op.direction)) < 0.999:
        return False
    a, b = op.edge_start, op.edge_end
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return False
    ts = []
    for p in (p0, p1):
        t = float((p - a) @ ab) / L2
        ts.append(t)
        if np.linalg.norm(p - (a + t * ab)) > tol:
            return False
    lo, hi = min(ts), max(ts)
    pad = tol / float(np.sqrt(L2))
    if hi < -pad or lo > 1.0 + pad:
        return False
    return True


def blend_corner_profile(r: float, convex: bool,
                         bury: float = 0.0) -> list:
    """Cross-section of a fillet's corner tool in the blend plane.

    The plane is perpendicular to the edge ``direction`` through
    ``edge_start``, spanned by the blend-plane normals (n_a, n_b); the
    origin is the sharp edge. A CONVEX fillet (material removed) returns
    the corner sliver in the (-n_a, -n_b) quadrant: the square corner
    minus the quarter round -- the volume that must be cut away. A
    CONCAVE fillet (material added) returns the quarter disk in the
    (+n_a, +n_b) quadrant -- the volume that must be fused on.

    ``bury`` (concave only) extends the two straight legs past the origin
    into the material side by that amount: OCC will not reliably fuse a
    solid that only touches the base along faces (the lateral-pad
    doctrine), so the legs overlap into existing material while the arc
    -- the visible fillet surface -- stays exact.

    All loops are wound COUNTER-CLOCKWISE: ``trimesh``'s polygon
    extrusion yields a non-watertight mesh for clockwise input (the
    mesh-volume gate then rejects the whole plan), so loop orientation
    is part of this function's contract.

    This is the single definition of the corner geometry: the headless
    round-trip (``solidify._apply_fillets``) and the FreeCAD executor's
    geometric fallback both extrude it along the detected edge span.
    """
    r = float(r)
    if r <= 0.0:
        return []
    if convex:
        return [
            SketchLine(start=np.array([0.0, 0.0]),
                       end=np.array([0.0, -r])),
            SketchArc(center=np.array([-r, -r]), radius=r,
                      start=np.array([0.0, -r]),
                      end=np.array([-r, 0.0]), sweep=np.pi / 2),
            SketchLine(start=np.array([-r, 0.0]),
                       end=np.array([0.0, 0.0])),
        ]
    d = float(bury)
    return [
        SketchLine(start=np.array([-d, -d]), end=np.array([r, -d])),
        SketchLine(start=np.array([r, -d]), end=np.array([r, 0.0])),
        SketchArc(center=np.array([0.0, 0.0]), radius=r,
                  start=np.array([r, 0.0]), end=np.array([0.0, r]),
                  sweep=np.pi / 2),
        SketchLine(start=np.array([0.0, r]), end=np.array([-d, r])),
        SketchLine(start=np.array([-d, r]), end=np.array([-d, -d])),
    ]


def chamfer_corner_profile(s: float, convex: bool,
                           bury: float = 0.0) -> list:
    """Cross-section of an equal-leg chamfer's corner tool, same plane
    and conventions as :func:`blend_corner_profile` (a straight hypotenuse
    instead of an arc)."""
    s = float(s)
    if s <= 0.0:
        return []
    if convex:
        return [
            SketchLine(start=np.array([0.0, 0.0]),
                       end=np.array([-s, 0.0])),
            SketchLine(start=np.array([-s, 0.0]),
                       end=np.array([0.0, -s])),
            SketchLine(start=np.array([0.0, -s]),
                       end=np.array([0.0, 0.0])),
        ]
    d = float(bury)
    return [
        SketchLine(start=np.array([-d, -d]), end=np.array([s, -d])),
        SketchLine(start=np.array([s, -d]), end=np.array([s, 0.0])),
        SketchLine(start=np.array([s, 0.0]), end=np.array([0.0, s])),
        SketchLine(start=np.array([0.0, s]), end=np.array([-d, s])),
        SketchLine(start=np.array([-d, s]), end=np.array([-d, -d])),
    ]


@dataclass
class ChamferOp:
    """Equal-leg chamfer on the rebuilt sharp solid's edge; geometry
    mirrors FilletOp (the sharp edge is the intersection line of the two
    blended planes) and reuses `fillet_edge_matches` for edge lookup."""
    size: float
    edge_start: np.ndarray
    edge_end: np.ndarray
    direction: np.ndarray
    n_a: np.ndarray
    n_b: np.ndarray
    label: str = ""


@dataclass
class CrossHoleOp:
    """A hole whose axis is not the extrusion direction.

    Carried in WORLD coordinates (3D axis + anchors). A THROUGH hole is cut
    as a midplane through-all pocket on a sketch normal to the axis. A BLIND
    hole (``through=False``) is a depth-limited pocket: ``positions3d`` are
    the ENTRY points on the wall, ``entry_direction`` the unit vector into
    the part, and ``depth`` the drilled length from the wall to the flat
    floor."""
    diameter: float
    axis: np.ndarray
    positions3d: list
    label: str = ""
    through: bool = True
    depth: float | None = None
    entry_direction: np.ndarray | None = None


@dataclass
class ConeOp:
    """A conical recess (or stud) revolved about a frame-z axis.

    Rebuilt by the executor as a placed ``PartDesign::SubtractiveCone``
    (or ``AdditiveCone``) primitive: ``Radius1`` at the base, ``Radius2``
    at the top, ``Height`` along the axis. A primitive is preferred over a
    tapered pocket (whose taper SIGN is a convention to get wrong) and over
    a sketch-and-revolve (which needs a closed profile wire).

    ``r_mouth`` is the radius at the opening face, ``r_far`` the radius at
    the far end (0 for a recess running to a point), and ``depth`` the
    axial distance between them. ``surface_z`` is the frame-z of the
    opening face, as for HoleOp.
    """
    r_mouth: float
    r_far: float
    depth: float
    positions: list                 # [(x, y), ...] in the plan frame
    from_top: bool = True
    additive: bool = False
    through: bool = False
    surface_z: float | None = None
    label: str = ""


@dataclass
class BuildPlan:
    frame_origin: np.ndarray
    frame_x: np.ndarray
    frame_y: np.ndarray
    frame_z: np.ndarray
    base: BasePad
    holes: list[HoleOp] = field(default_factory=list)
    pockets: list[PocketOp] = field(default_factory=list)
    pads: list[PadOp] = field(default_factory=list)
    fillets: list[FilletOp] = field(default_factory=list)
    chamfers: list[ChamferOp] = field(default_factory=list)
    cross_holes: list[CrossHoleOp] = field(default_factory=list)
    cones: list[ConeOp] = field(default_factory=list)
    absorbed_features: int = 0
    unplanned: list[str] = field(default_factory=list)
    #: labels of PocketOps synthesized from exposed multi-level tops
    step_labels: list[str] = field(default_factory=list)
    #: deviation-correction patches (solidify.CorrectionOp): watertight
    #: patch solids to cut/add after the parametric features, covering
    #: freeform geometry no feature captured. Filled by
    #: :func:`.solidify.plan_corrections`; empty keeps legacy behaviour.
    corrections: list = field(default_factory=list)
    #: the source mesh the corrections were computed against; the executor
    #: intersects the finished body with it (removing all OVER at once)
    #: before fusing the UNDER patches. Set with ``corrections``.
    source_mesh: object = None


def hole_op_properties(op: HoleOp) -> dict:
    """Map a HoleOp onto PartDesign Hole feature properties (pure).

    Encodes through vs blind (DepthType), counterbore (HoleCutType +
    dimensions), and side-ness (Reversed; the executor also places the
    profile sketch on the corresponding face). Recognized blind holes are
    flat-bottomed cylinders, hence DrillPoint = Flat.
    """
    props: dict = {
        "Diameter": float(op.diameter),
        "DepthType": "ThroughAll" if op.through else "Dimension",
        "DrillPoint": "Flat",
        "Threaded": False,
        "Tapered": False,
        "HoleCutType": "None",
        "Reversed": not op.from_top,
    }
    if not op.through:
        props["Depth"] = float(op.depth)
    if op.countersink_diameter and op.counterbore_diameter:
        # a counterdrill: a cylindrical recess with a conical transition
        # down to the drill. PartDesign::Hole takes the BORE diameter, the
        # depth of the CYLINDRICAL part only (the cone below it is fixed by
        # the angle and the two diameters), and the included angle.
        props["HoleCutType"] = "Counterdrill"
        props["HoleCutDiameter"] = float(op.counterbore_diameter)
        props["HoleCutDepth"] = float(op.counterbore_depth)
        props["HoleCutCountersinkAngle"] = float(op.countersink_angle or 90.0)
    elif op.countersink_diameter:
        # a conical entry: PartDesign::Hole models it by the mouth diameter
        # plus the included angle (no depth -- the angle and the drill
        # diameter fix the cone).
        props["HoleCutType"] = "Countersink"
        props["HoleCutDiameter"] = float(op.countersink_diameter)
        props["HoleCutCountersinkAngle"] = float(op.countersink_angle or 90.0)
    elif op.counterbore_diameter:
        props["HoleCutType"] = "Counterbore"
        props["HoleCutDiameter"] = float(op.counterbore_diameter)
        props["HoleCutDepth"] = float(op.counterbore_depth)
    return props


def lateral_pad_world_frame(plan: "BuildPlan", pad: PadOp):
    """World-frame (origin, u, v, axis) for a lateral PadOp (pure).

    The pad's basis is stored in the PLAN frame (design note 36); the
    FreeCAD executor builds in world coordinates, so it composes the plan
    frame R = [frame_x frame_y frame_z] with the pad's plan-frame basis.
    A profile 2D point (uc, vc) then maps to the world point
    ``origin + uc*u + vc*v``, and the pad extrudes ``pad.length`` along
    ``axis`` -- identical to plan-frame construction followed by the frame
    placement, so the container round-trip and the FreeCAD body agree.
    """
    if pad.axis is None:
        raise ValueError("lateral_pad_world_frame requires a lateral pad")
    R = np.column_stack([plan.frame_x, plan.frame_y, plan.frame_z])
    origin = np.asarray(plan.frame_origin, dtype=float) + R @ pad.plane_origin
    u = R @ pad.plane_u
    v = R @ pad.plane_v
    axis = R @ pad.axis
    return origin, u, v, axis


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def _loop_area(lp: np.ndarray) -> float:
    """Planar polygon area: half the norm of the summed cross products.

    (Summing |components| instead is orientation-dependent by up to
    sqrt(3): wrong for comparing differently oriented loops.)
    """
    c = lp.mean(axis=0)
    return 0.5 * float(np.linalg.norm(
        np.cross(lp - c, np.roll(lp, -1, axis=0) - c).sum(axis=0)))


def _valid_loop(loop: np.ndarray, tol: float) -> bool:
    """True if a 2D loop can form a closed sketch profile: at least three
    vertices distinct beyond ``tol`` and a non-negligible enclosed area.

    Noisy meshes (subdivided cubes, scanned boxes) yield sliver segments
    whose boundary loop collapses to one or two points; emitting a pocket
    from such a loop makes a <3-point wire that crashes the boolean
    rebuild ("linearring requires 4 coordinates") and the FreeCAD sketch.
    Dropping the loop degrades the part by one feature instead.
    """
    loop = np.asarray(loop, dtype=float)
    if len(loop) < 3:
        return False
    keys = np.unique(np.round(loop / max(tol, 1e-12)).astype(np.int64), axis=0)
    if len(keys) < 3:
        return False
    px, py = loop[:, 0], loop[:, 1]
    area = 0.5 * abs(float(np.dot(px, np.roll(py, -1))
                           - np.dot(py, np.roll(px, -1))))
    return area > tol * tol


def _outer_loop(seg) -> np.ndarray | None:
    loops = seg.boundary_loops
    if not loops:
        return None
    return loops[int(np.argmax([_loop_area(lp) for lp in loops]))]


def plan_history(report: ReconstructionReport, feats: FeatureReport,
                 pats: PatternReport,
                 patches: list[PatchSpec] | None = None) -> BuildPlan:
    from .snapping import cluster_directions

    planes = [(i, s) for i, s in enumerate(report.surfaces)
              if isinstance(s.fit.primitive, Plane)]
    if not planes:
        raise ValueError("no planar surfaces: cannot infer a prismatic base")

    # extrusion direction: the plane-normal cluster with the largest area
    dirs = np.array([s.fit.primitive.normal for _, s in planes])
    areas = np.array([s.segment.area for _, s in planes])
    labels, means = cluster_directions(dirs, areas, np.deg2rad(1.0))
    best = int(np.argmax([areas[labels == k].sum()
                          for k in range(means.shape[0])]))
    z = means[best]
    # Canonicalize the axis SIGN. `means[best]` inherits its sign from the
    # tessellation's face-normal winding, so the SAME part can reconstruct
    # with z=+1 from one mesher and z=-1 from another (field-observed:
    # Python/trimesh gave +z, FreeCAD's mesh gave -z). The plan stays
    # self-consistent either way, but the FreeCAD executor's z_at/flip
    # placement assumes a canonical orientation and mis-places every
    # from-bottom feature under the inverted sign. Point z along the
    # positive direction of its dominant axis -- stable and mesher-
    # independent -- so the executor is never fed an inverted frame.
    if float(z[int(np.argmax(np.abs(z)))]) < 0.0:
        z = -z
    x, y = _axis_frame(z)

    members = [planes[i] for i in np.flatnonzero(labels == best)]
    # a boss's cap plane belongs to the boss, not the base: including it
    # would inflate the base extrusion to the top of the boss
    boss_caps = {f.surface_indices[1] for f in feats.features
                 if f.kind == "boss" and len(f.surface_indices) >= 2}
    extent_members = [(i, s) for i, s in members if i not in boss_caps]         or members
    offsets = [float(s.fit.primitive.point @ z) for _, s in extent_members]
    z0, z1 = min(offsets), max(offsets)
    if z1 - z0 <= 0:
        raise ValueError("degenerate base extent")

    # base profile: the member plane with the largest outer loop
    def outer_area(seg):
        lp = _outer_loop(seg)
        return -1.0 if lp is None else _loop_area(lp)

    base_i, base_s = max(members, key=lambda t: outer_area(t[1].segment))
    outer3d = _outer_loop(base_s.segment)

    _pts = np.vstack([s.segment.points for s in report.surfaces])
    centroid = _pts.mean(axis=0)
    origin = centroid - ((centroid @ z) - z0) * z
    _tol = max(1e-3 * float(np.linalg.norm(_pts.max(axis=0) - _pts.min(axis=0))),
               1e-9)

    def to2d(p3):
        rel = np.atleast_2d(p3) - origin
        return np.column_stack([rel @ x, rel @ y])

    profile = loop_to_sketch(to2d(outer3d))

    # A base face that is a frame/ring has inner loops; carry the ones that
    # are genuine THROUGH-openings (present at the opposite z-end too, not a
    # blind-pocket mouth) so the base prism is a ring, not solid (note 37).
    base_holes: list = []
    base_h = float((base_s.fit.primitive.point - origin) @ z)
    inner = [bl for bl in base_s.segment.boundary_loops
             if not (len(bl) == len(outer3d) and np.allclose(bl, outer3d))]
    if inner:
        opp_inner = [
            bl for _, s in members
            if abs(float((s.fit.primitive.point - origin) @ z) - base_h)
            > 0.5 * (z1 - z0)
            for bl in s.segment.boundary_loops
            if s.segment is not base_s.segment
            and not (_outer_loop(s.segment) is not None
                     and len(bl) == len(_outer_loop(s.segment))
                     and np.allclose(bl, _outer_loop(s.segment)))]

        def _match(a3, b3) -> bool:
            a2, b2 = to2d(a3), to2d(b3)
            if float(np.linalg.norm(a2.mean(0) - b2.mean(0))) > 5.0 * _tol:
                return False
            aa = 0.5 * abs(float(np.dot(a2[:, 0], np.roll(a2[:, 1], -1))
                                 - np.dot(a2[:, 1], np.roll(a2[:, 0], -1))))
            ab = 0.5 * abs(float(np.dot(b2[:, 0], np.roll(b2[:, 1], -1))
                                 - np.dot(b2[:, 1], np.roll(b2[:, 0], -1))))
            return abs(aa - ab) <= 0.15 * max(aa, ab, 1e-12)

        # An opening already recovered as a drilled hole/counterbore feature
        # is cut as that FEATURE, not carved into the base -- else the base
        # pre-punches it, the hole/counterbore pockets cut already-open space,
        # and the counterbore recesses are lost (field-observed on
        # featuretype: 8 counterbored through-holes pre-holed the base, so
        # the top counterbore sinks vanished). Only openings that are NOT a
        # feature (a frame's odd polygon, note 37) belong on the base.
        _drilled = []
        for ff in feats.features:
            if ff.kind in ("hole", "counterbore") and "position" in ff.params:
                fc = to2d(np.asarray(ff.params["position"], dtype=float))[0]
                fr = 0.5 * float(ff.params.get("diameter", 0.0))
                _drilled.append((fc, fr))
            elif ff.kind == "cone_pocket" and "position" in ff.params:
                # the mouth of a tapered through hole is cut by its own
                # cone primitive; carving it into the base too would punch
                # a STRAIGHT bore of the mouth radius clean through
                fc = to2d(np.asarray(ff.params["position"], dtype=float))[0]
                _drilled.append(
                    (fc, float(ff.params.get("mouth_radius", 0.0))))

        def _is_drilled_hole(loop2) -> bool:
            c = loop2.mean(0)
            lr = float(np.mean(np.linalg.norm(loop2 - c, axis=1)))
            for fc, fr in _drilled:
                if float(np.linalg.norm(c - fc)) < max(3.0 * _tol, 0.5 * fr) \
                        and abs(lr - fr) <= 0.35 * max(fr, _tol):
                    return True
            return False

        for il in inner:
            il2 = to2d(il)
            if _valid_loop(il2, _tol) and not _is_drilled_hole(il2) \
                    and any(_match(il, oil) for oil in opp_inner):
                base_holes.append(loop_to_sketch(il2, arcs=False))

    # profile orientation follows the source face; positive area expected
    plan = BuildPlan(frame_origin=origin, frame_x=x, frame_y=y, frame_z=z,
                     base=BasePad(profile=profile, length=z1 - z0,
                                  hole_profiles=base_holes))

    # ---- features -> ops ----------------------------------------------------
    #: surfaces that are a cone feature's own end faces -- excluded
    #: from terrace synthesis (see the cone_pocket branch below)
    _cone_faces: set[int] = set()

    def _aligned(f):
        return abs(float(np.asarray(f.params["axis"]) @ z)) > 0.999

    def _pos2d(f):
        return tuple(to2d(np.asarray(f.params["position"]))[0])

    def _side(f):
        """True when the feature's reference cylinder hugs the top face.

        For counterbores the reference is the BIG bore (provenance index
        1); for holes the wall cylinder (index 0). A cut whose axial span
        clings to z = L is machined from the top, to z = 0 from the
        bottom -- the plan frame's z sign is arbitrary, so this must be
        computed, not assumed. A countersunk hole's drill does not reach
        its mouth (the cone occupies that band), so the mouth face -- not
        the drill span -- fixes the machining side.
        """
        if f.params.get("countersink"):
            mz = float((np.asarray(f.params["mouth"]) - origin) @ z)
            L = plan.base.length
            return (L - mz) <= mz
        idx = f.surface_indices[1 if f.kind == "counterbore" else 0]
        hvals = (report.surfaces[idx].segment.points - origin) @ z
        L = plan.base.length
        return (L - float(hvals.max())) <= float(hvals.min())

    def _surface_z(f):
        """Frame-z of the face the hole opens on (counterbore mouth for a
        counterbore, drill mouth otherwise), measured from frame_origin. The
        executor places the hole sketch here so a bore whose mouth sits below
        the tallest level is cut inward, not modelled as an outward tower."""
        if f.params.get("countersink"):
            return float((np.asarray(f.params["mouth"]) - origin) @ z)
        idx = f.surface_indices[1 if f.kind == "counterbore" else 0]
        hvals = (report.surfaces[idx].segment.points - origin) @ z
        L = plan.base.length
        # from_top mouth is the high end of the opening cylinder; from_bottom
        # the low end (mirrors _side's top/bottom test)
        return float(hvals.max()) if (L - float(hvals.max())) <= \
            float(hvals.min()) else float(hvals.min())

    def _hole_op(spec_params, positions, label, from_top=True,
                 surface_z=None):
        return HoleOp(
            from_top=from_top,
            diameter=spec_params["diameter"],
            through=bool(spec_params.get("through", False)),
            depth=float(spec_params.get("depth", plan.base.length)),
            counterbore_diameter=spec_params.get("counterbore_diameter"),
            counterbore_depth=spec_params.get("counterbore_depth"),
            countersink_diameter=spec_params.get("countersink_diameter"),
            countersink_angle=spec_params.get("countersink_angle"),
            positions=positions, label=label, surface_z=surface_z)

    def _cross_hole_op(f):
        """A non-axis-aligned hole -> CrossHoleOp. A THROUGH hole carries an
        axis anchor (the executor cuts symmetrically through). A BLIND hole
        carries the ENTRY point on the wall, the inward direction, and the
        drilled depth (a one-sided Length pocket)."""
        axis = np.asarray(f.params["axis"], dtype=float)
        if f.params.get("through"):
            return CrossHoleOp(
                diameter=f.params["diameter"], axis=axis,
                positions3d=[list(f.params["position"])],
                through=True, label=f.description)
        pos = np.asarray(f.params["position"], dtype=float)
        entry = pos.copy()
        open_n = None
        if len(f.surface_indices) > 1:
            op = report.surfaces[f.surface_indices[1]].fit.primitive
            if isinstance(op, Plane):
                open_pt = np.asarray(op.point, dtype=float)
                open_n = np.asarray(op.normal, dtype=float)
                denom = float(axis @ open_n)
                if abs(denom) > 1e-9:              # pierce the wall plane
                    entry = pos + ((open_pt - pos) @ open_n) / denom * axis
        # inward direction: toward the floor plane if present, else opposite
        # the wall's outward normal
        if len(f.surface_indices) > 2 and isinstance(
                report.surfaces[f.surface_indices[2]].fit.primitive, Plane):
            floor_pt = np.asarray(
                report.surfaces[f.surface_indices[2]].fit.primitive.point,
                dtype=float)
            d = float((floor_pt - entry) @ axis)
        elif open_n is not None:
            d = -float(open_n @ axis)
        else:
            d = 1.0
        direction = axis * (1.0 if d >= 0 else -1.0)
        return CrossHoleOp(
            diameter=f.params["diameter"], axis=axis,
            positions3d=[list(entry)], through=False,
            depth=float(f.params["depth"]),
            entry_direction=list(direction), label=f.description)

    # horizontal cylinders protruding beyond the base become circular
    # lateral pads (design note 36); the surfaces they consume are skipped
    # by the pattern/feature dispatch and the planar lateral detector so
    # they are not double-handled as bosses, fillets, or crude boxes
    consumed = _plan_lateral_cylinders(plan, report, z, x, y, origin, to2d,
                                       outer3d)

    grouped = {id(m) for p in pats.patterns for m in p.members}
    for p in pats.patterns:
        if any(i in consumed for m in p.members for i in m.surface_indices):
            continue
        kind = p.members[0].kind
        if kind == "hole" and not _aligned(p.members[0]):
            for m in p.members:
                plan.cross_holes.append(_cross_hole_op(m))
        elif kind in ("hole", "counterbore") and _aligned(p.members[0]):
            plan.holes.append(_hole_op(p.members[0].params,
                                       [_pos2d(m) for m in p.members],
                                       p.description,
                                       from_top=_side(p.members[0]),
                                       surface_z=_surface_z(p.members[0])))
        elif kind == "boss" and _aligned(p.members[0]):
            for m in p.members:
                hvals = (report.surfaces[m.surface_indices[0]].segment.points
                         - origin) @ z
                plan.pads.append(PadOp(
                    profile=[SketchCircle(center=np.array(_pos2d(m)),
                                          radius=m.params["diameter"] / 2)],
                    length=m.params["height"],
                    from_top=float(hvals.mean()) > plan.base.length / 2,
                    label=m.description))
        else:
            plan.unplanned.append(p.description)

    for f in feats.features:
        if id(f) in grouped:
            continue
        if any(i in consumed for i in f.surface_indices):
            continue
        if f.kind in ("hole", "counterbore") and _aligned(f):
            plan.holes.append(_hole_op(f.params, [_pos2d(f)], f.description,
                                       from_top=_side(f),
                                       surface_z=_surface_z(f)))
        elif f.kind == "hole" and not _aligned(f):
            plan.cross_holes.append(_cross_hole_op(f))
        elif f.kind == "boss" and _aligned(f):
            hvals = (report.surfaces[f.surface_indices[0]].segment.points
                     - origin) @ z
            plan.pads.append(PadOp(
                profile=[SketchCircle(center=np.array(_pos2d(f)),
                                      radius=f.params["diameter"] / 2)],
                length=f.params["height"],
                from_top=float(hvals.mean()) > plan.base.length / 2,
                label=f.description))
        elif f.kind == "cone_pocket" and _aligned(f):
            plan.cones.append(ConeOp(
                r_mouth=float(f.params["mouth_radius"]),
                r_far=float(f.params["far_radius"]),
                depth=float(f.params["depth"]),
                positions=[_pos2d(f)],
                from_top=_side(f),
                additive=False,
                through=bool(f.params.get("through", False)),
                surface_z=_surface_z(f),
                label=f.description))
            # the floor disk is this cone's own end face: keep the terrace
            # pass off it, or it also emits a straight-walled substitute
            _cone_faces.update(f.surface_indices[2:])
        elif f.kind == "slot" and _aligned(f) and f.params.get("open"):
            # synthesize the closed notch profile: two lines + far arc +
            # a mouth-closing segment pushed OUTWARD so the cut clears the
            # boundary cleanly
            r = f.params["width"] / 2.0
            c2 = to2d(np.asarray(f.params["position"]))[0]
            d3 = np.asarray(f.params["direction"], dtype=float)
            d2 = np.array([float(d3 @ x), float(d3 @ y)])
            d2 = d2 / max(np.linalg.norm(d2), 1e-12)
            n2 = np.array([-d2[1], d2[0]])
            reach = f.params["length"] - r + 0.2 * r      # past the mouth
            a_pt, b_pt = c2 + r * n2, c2 - r * n2
            ma, mb = a_pt + reach * d2, b_pt + reach * d2
            profile = [
                SketchLine(start=ma, end=a_pt),
                SketchArc(center=c2, radius=r, start=a_pt, end=b_pt,
                          sweep=-np.pi if _cw(a_pt, b_pt, c2, d2) else np.pi),
                SketchLine(start=b_pt, end=mb),
                SketchLine(start=mb, end=ma),
            ]
            top_idx = f.surface_indices[0]
            top_off = float((report.surfaces[top_idx].fit.primitive.point
                             - origin) @ z)
            plan.pockets.append(PocketOp(
                profile=profile, depth=f.params["depth"],
                through=bool(f.params.get("through", False)),
                from_top=top_off > plan.base.length / 2,
                label=f.description))
        elif f.kind == "slot" and _aligned(f):
            top_idx = f.surface_indices[0]     # provenance: [top plane, ends]
            seg = report.surfaces[top_idx].segment
            loops = [lp for lp in seg.boundary_loops
                     if lp is not _outer_loop(seg)]
            match = None
            for lp in loops:
                mid2 = to2d(lp).mean(axis=0)
                ref2 = to2d(np.asarray(f.params["position"]))[0]
                if np.linalg.norm(mid2 - ref2) < max(f.params["width"], 1e-6):
                    match = lp
                    break
            if match is None:
                plan.unplanned.append(f.description)
                continue
            top_off = float((report.surfaces[top_idx].fit.primitive.point
                             - origin) @ z)
            if not _valid_loop(to2d(match), _tol):
                plan.unplanned.append(f.description)
                continue
            plan.pockets.append(PocketOp(
                profile=loop_to_sketch(to2d(match)),
                depth=f.params["depth"],
                through=bool(f.params.get("through", False)),
                from_top=top_off > plan.base.length / 2,
                label=f.description))
        elif f.kind == "pocket":
            top_idx = f.surface_indices[0]     # provenance order: [top, floor, walls...]
            floor_idx = (f.surface_indices[1]
                         if len(f.surface_indices) > 1 else None)
            floor_n = (report.surfaces[floor_idx].fit.primitive.normal
                       if floor_idx is not None else None)
            if floor_n is not None and abs(float(np.asarray(floor_n) @ z)) > 0.9:
                # HORIZONTAL floor: reconstructed by the height-field terrace
                # pass (_plan_terraces) to its exact projected footprint,
                # which is more faithful than the opening loop and also
                # captures the recesses this opening-driven detector misses.
                # (A filleted/chamfered floor -- inset from its opening -- is
                # the one case the flat footprint under-reaches; handling it
                # via the opening cut here regressed real terraced/counterbored
                # parts, so it is left as a documented limitation.)
                continue
            # side-wall pocket (vertical floor): keep the opening-driven cut
            seg = report.surfaces[top_idx].segment
            loops = [lp for lp in seg.boundary_loops
                     if lp is not _outer_loop(seg)]
            if not loops:
                plan.unplanned.append(f.description)
                continue
            opening = loops[int(np.argmax([_loop_area(lp) for lp in loops]))]
            top_off = float((report.surfaces[top_idx].fit.primitive.point
                             - origin) @ z)
            if not _valid_loop(to2d(opening), _tol):
                plan.unplanned.append(f.description)
                continue
            plan.pockets.append(PocketOp(
                profile=loop_to_sketch(to2d(opening)),
                depth=f.params["depth"],
                from_top=top_off > plan.base.length / 2,
                label=f.description))
        elif f.kind == "fillet" and _aligned(f):
            plan.absorbed_features += 1        # lives in the base profile arcs
        elif f.kind == "chamfer" and _aligned(f):
            plan.absorbed_features += 1        # a line segment in the profile
        elif f.kind == "chamfer":
            op = _chamfer_op(report, f)
            if op is not None:
                plan.chamfers.append(op)
            else:
                plan.unplanned.append(f.description)
        elif f.kind == "fillet":
            op = _fillet_op(report, patches, f, z)
            if op is not None:
                plan.fillets.append(op)
            else:
                plan.unplanned.append(f.description)
        else:
            plan.unplanned.append(f.description)

    _plan_lateral_pads(plan, report, z, x, y, origin, to2d, outer3d, consumed)
    _plan_terraces(plan, report, members, z, x, y, origin, to2d, outer3d,
                   skip=_cone_faces)
    _upgrade_to_tapered_pockets(plan, report, z, x, y, origin, to2d,
                                outer3d)
    _plan_bottom_pockets(plan, report, members, feats, z, x, y, origin,
                         to2d)
    _plan_gussets(plan, report, z, x, y, origin)
    _lateral_pad_mesh_veto(plan, report, x, y, z, origin, _tol)
    # Decline heavily faceted / organic exports. Count the intermediate
    # horizontal member SEGMENTS (a tessellated curve is dozens of thin
    # horizontal facets); terraces coalesce disjoint facets so the emitted
    # count hides them, and distinct-level count hides them too when facets
    # share z. Field-observed: box.STL has 97 facet segments (->16-solid
    # garbage) vs featuretype 15 / octagonal_pocket 16 for clean parts.
    _lvl_tol = max(1e-3 * plan.base.length, 1e-9)
    _intermediate = sum(
        1 for _, s in members
        if _lvl_tol < (float(s.fit.primitive.point @ z) - float(origin @ z))
        < plan.base.length - _lvl_tol)
    if _intermediate > MAX_STEP_LEVELS or len(plan.step_labels) > MAX_STEP_LEVELS:
        raise ValueError(
            f"part has {max(_intermediate, len(plan.step_labels))} "
            f"intermediate horizontal facets, beyond the single-axis "
            f"prismatic model (> {MAX_STEP_LEVELS}); it is likely a faceted, "
            f"terraced, or organic export")
    return plan


def _dist_to_polyline(pts: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Min distance from each 2D point to a closed 2D polyline."""
    a = loop
    b = np.roll(loop, -1, axis=0)
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    L2[L2 == 0.0] = 1e-30
    d = np.full(len(pts), np.inf)
    for p_i in range(len(pts)):
        t = np.clip(((pts[p_i] - a) * ab).sum(axis=1) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d[p_i] = float(np.min(np.linalg.norm(pts[p_i] - proj, axis=1)))
    return d


def _dedupe_loop(loop: np.ndarray, min_seg: float) -> np.ndarray:
    """Drop consecutive vertices closer than ``min_seg`` apart, removing the
    sub-tolerance sliver edges that a projected mesh footprint leaves in a
    loop. Manifold booleans tolerate these, but OCC's face builder can choke
    on them (field-observed: FreeCAD leaves thin walls at the corners of a
    curvy terrace, and one counterbore, when the profile carries slivers).
    Shape is preserved because only below-tolerance detail is merged."""
    loop = np.asarray(loop, dtype=float)
    if len(loop) < 4:
        return loop
    keep = [loop[0]]
    for pt in loop[1:]:
        if np.linalg.norm(pt - keep[-1]) >= min_seg:
            keep.append(pt)
    keep = np.asarray(keep)
    # a closing sliver (last ~ first) collapses the ring end onto its start
    while len(keep) > 3 and np.linalg.norm(keep[-1] - keep[0]) < min_seg:
        keep = keep[:-1]
    return keep if len(keep) >= 3 else loop


def _buffer_loop(loop: np.ndarray, dist: float) -> np.ndarray:
    """Grow a terrace exterior OUTWARD by ``dist`` with sharp (mitre)
    corners. Purpose: make adjacent terraces at different depths OVERLAP
    by a hair instead of sharing an exact boundary edge. OCC's Cut fuses a
    shared edge cleanly, but FreeCAD PartDesign leaves a thin standing
    'fin' along it (field-observed: the semicircular pocket and the wavy
    front). With a small overlap the deeper pocket's boundary falls INSIDE
    the shallower pocket's already-cut region, so there is no shared edge
    for PartDesign to fin -- the deeper cut simply extends under the
    shallower one (deepest-cut-wins keeps the result identical to the mesh
    to within ``dist``). Also subsumes the old outline nudge: outline-
    coincident edges move out too, clearing the boolean coincidence that
    the removed push handled."""
    from shapely.geometry import Polygon
    try:
        poly = Polygon(loop)
        if not poly.is_valid:
            poly = poly.buffer(0)
        grown = poly.buffer(dist, join_style=3)   # bevel: no spikes, no arcs
        if grown.is_empty or grown.geom_type != "Polygon":
            return loop
        out = np.asarray(grown.exterior.coords)[:-1]
        return out if len(out) >= 3 else loop
    except Exception:                               # noqa: BLE001
        return loop


def _push_outline_vertices(loop: np.ndarray, base_loop: np.ndarray,
                           delta: float, tol: float) -> np.ndarray:
    """Push loop vertices lying ON the base outline outward by ``delta``
    along the local outward bisector, so a cut profile clears the part
    boundary cleanly while interior (true step wall) vertices stay exact."""
    n = len(loop)
    nxt = np.roll(loop, -1, axis=0)
    area2 = float((loop[:, 0] * nxt[:, 1] - loop[:, 1] * nxt[:, 0]).sum())
    ccw = area2 > 0
    on_edge = _dist_to_polyline(loop, base_loop) < tol
    out = loop.copy()
    for i in np.flatnonzero(on_edge):
        e_prev = loop[i] - loop[(i - 1) % n]
        e_next = loop[(i + 1) % n] - loop[i]
        nrm = np.zeros(2)
        for e in (e_prev, e_next):
            ln = np.linalg.norm(e)
            if ln > 1e-12:
                perp = np.array([e[1], -e[0]]) / ln
                nrm += perp if ccw else -perp
        ln = np.linalg.norm(nrm)
        if ln > 1e-12:
            out[i] = loop[i] + delta * (nrm / ln)
    return out


def _merge_intervals(intervals, gap):
    """Union 1D intervals that overlap or lie within ``gap`` of each
    other; returns disjoint (lo, hi) spans sorted by lo. Used to split a
    side's protruding walls into separate flanges."""
    out: list[list[float]] = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def _inside_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crossing-number point-in-polygon: True where a 2D point lies
    strictly inside the closed polygon ``poly``."""
    px, py = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & \
            (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi)
        inside ^= cond
        j = i
    return inside


def _cylinder_convexity(points: np.ndarray, normals: np.ndarray,
                        axis: np.ndarray, axis_point: np.ndarray) -> float:
    """Mean alignment of a cylinder segment's vertex normals with the
    outward radial direction: ~+1 for a convex BOSS/rail (material inside,
    normals point away from the axis), ~-1 for a concave HOLE (material
    outside, normals point toward the axis). Lets the lateral detector
    reject a horizontal drilled hole that an incomplete base outline made
    look like it protrudes past the footprint (field-observed on the
    angle_block STL)."""
    a = np.asarray(axis, dtype=float)
    rel = np.asarray(points, dtype=float) - np.asarray(axis_point, dtype=float)
    radial = rel - np.outer(rel @ a, a)
    rn = np.linalg.norm(radial, axis=1, keepdims=True)
    ru = radial / (rn + 1e-12)
    return float(np.mean(np.sum(np.asarray(normals) * ru, axis=1)))


def _plan_lateral_cylinders(plan, report, z, x, y, origin, to2d, base_outer3d):
    """Horizontal cylinders protruding beyond the base footprint -> a
    circular lateral pad (design note 36): a peg boss (axis perpendicular
    to the wall) or a rounded rail (axis parallel to the wall). Both are
    the same primitive -- a cylinder whose body reaches past the outline
    -- and both extrude a circle of the cylinder's radius along its axis;
    for a rail the inner half of the circle simply fuses back into the
    base. Returns the consumed surface indices (the cylinder and its
    end-cap disks) so the planar detector and the feature dispatch skip
    them instead of emitting a crude box, a stray fillet, or an unplanned
    boss.
    """
    pts_all = np.vstack([s.segment.points for s in report.surfaces])
    diag = float(np.linalg.norm(pts_all.max(axis=0) - pts_all.min(axis=0)))
    tol = max(1e-3 * diag, 1e-9)
    base2d = to2d(base_outer3d)
    L = plan.base.length
    consumed: set[int] = set()

    for i, s in enumerate(report.surfaces):
        prim = s.fit.primitive
        if not isinstance(prim, Cylinder):
            continue
        a = np.asarray(prim.axis, dtype=float)
        if abs(float(a @ z)) > 0.15:
            continue                                # vertical: hole/boss
        pts = s.segment.points
        p2 = to2d(pts)
        outside = ~_inside_poly(p2, base2d)
        far = _dist_to_polyline(p2, base2d) > 3.0 * tol
        if not bool(np.any(outside & far)):
            continue                                # inside base: a cross-hole
        if _cylinder_convexity(pts, s.segment.normals, a, prim.point) < 0.3:
            continue                                # concave: a drilled hole,
            #  not a protruding boss/rail (an incomplete base outline can
            #  make an interior hole look like it reaches past the footprint)

        u, v = _axis_frame(a)                        # basis perp to the axis
        ac = (pts - origin) @ a
        a_min, a_max = float(ac.min()), float(ac.max())
        r = float(prim.radius)
        cu = float((prim.point - origin) @ u)
        cv = float((prim.point - origin) @ v)

        # bury an axial end into the base so the union fuses: an end whose
        # cross-section continues into base material (a peg's inner face)
        # is extended by a small overlap; ends that sit at a part boundary
        # (a rail spanning the full width) are left alone.
        overlap = 0.05 * (a_max - a_min)
        for end, step in ((a_min, -1.0), (a_max, +1.0)):
            probe = origin + (end + step * 3.0 * tol) * a + cu * u + cv * v
            hb = float((probe - origin) @ z)
            inside_base = bool(_inside_poly(to2d(probe[None, :]), base2d)[0]) \
                and -tol <= hb <= L + tol
            if inside_base:
                if step < 0:
                    a_min -= overlap
                else:
                    a_max += overlap

        length = a_max - a_min
        if length <= tol:
            continue
        a_pf = np.array([a @ x, a @ y, a @ z])
        u_pf = np.array([u @ x, u @ y, u @ z])
        v_pf = np.array([v @ x, v @ y, v @ z])
        plan.pads.append(PadOp(
            profile=[SketchCircle(center=np.array([cu, cv]), radius=r)],
            length=length, axis=a_pf, plane_origin=a_min * a_pf,
            plane_u=u_pf, plane_v=v_pf,
            label=f"Lateral boss (d{2 * r:g})"))
        consumed.add(i)

        # consume end-cap disks (planar, normal parallel to the axis, at an
        # axial end within the radius) so the planar detector cannot anchor
        # a crude box pad on them
        for j, s2 in enumerate(report.surfaces):
            pm = s2.fit.primitive
            if not isinstance(pm, Plane):
                continue
            if abs(float(np.asarray(pm.normal) @ a)) <= 0.9:
                continue
            c = s2.segment.points.mean(axis=0)
            acj = float((c - origin) @ a)
            radial = float(np.linalg.norm(
                (c - prim.point) - ((c - prim.point) @ a) * a))
            near_end = min(abs(acj - a_min), abs(acj - a_max))
            if radial < r + 3.0 * tol and near_end < 0.2 * length + 3.0 * tol:
                consumed.add(j)

    return consumed


def _plan_lateral_pads(plan, report, z, x, y, origin, to2d, base_outer3d,
                       consumed=frozenset()):
    """Material protruding horizontally beyond the base footprint -- a
    mounting flange, side rail, gusseted bracket (design note 36).

    The base profile is one horizontal plane's outline, so a sideways
    protrusion in a limited z-band has no vertical-extrusion
    representation and would be dropped. Anchor detection on a VERTICAL
    wall whose footprint sits beyond the base outline; the pad extrudes
    along that wall's length axis, and its profile is the convex hull of
    all protruding points in the plane perpendicular to that axis. The
    hull is exact for convex cross-sections (rectangular lugs AND
    ramped/gusseted brackets alike); the pockets that carve sub-levels
    into the flange top are emitted separately and cut it afterwards.
    Inner vertices are buried slightly into the base so the union fuses.
    Multiple protrusions sharing an outward direction are separated by
    clustering their walls along the pad axis, so two lugs on one side
    become two pads rather than one hull bridging the gap between them.
    """
    pts_all = np.vstack([s.segment.points for s in report.surfaces])
    diag = float(np.linalg.norm(pts_all.max(axis=0) - pts_all.min(axis=0)))
    tol = max(1e-3 * diag, 1e-9)
    base2d = to2d(base_outer3d)

    # candidate outer walls, grouped by outward horizontal direction; each
    # wall contributes its extent along the pad axis so co-directional but
    # spatially separate walls (two lugs on one side) can be split apart
    groups: list[dict] = []
    for idx, s in enumerate(report.surfaces):
        if idx in consumed:
            continue
        prim = s.fit.primitive
        if not isinstance(prim, Plane):
            continue
        n = np.asarray(prim.normal, dtype=float)
        if abs(float(n @ z)) > 0.15:
            continue                                # not a vertical wall
        nh = n - float(n @ z) * z
        nn = float(np.linalg.norm(nh))
        if nn < 1e-9:
            continue
        nh = nh / nn
        if float(s.segment.face_normals.mean(axis=0) @ nh) < 0:
            nh = -nh                                # orient outward
        d2 = np.array([float(nh @ x), float(nh @ y)])
        base_reach = float(np.max(base2d @ d2))
        wall_reach = float(np.max(to2d(s.segment.points) @ d2))
        if wall_reach - base_reach < 3.0 * tol:
            continue                                # wall sits on the outline
        a = np.cross(z, nh)
        a /= np.linalg.norm(a)
        ac = (s.segment.points - origin) @ a
        span = (float(ac.min()), float(ac.max()))
        g = next((g for g in groups if float(nh @ g["nh"]) > 0.999), None)
        if g is None:
            groups.append({"nh": nh, "a": a, "spans": [span]})
        else:
            g["spans"].append(span)

    for g in groups:
        nh, a = g["nh"], g["a"]
        u, v = nh, z
        base_reach_u = float(np.max((base_outer3d - origin) @ u))
        # split co-directional walls into separate flanges: a real gap (no
        # wall material) beyond ~2% of the part diagonal is a distinct
        # protrusion; adjacent wall pieces of one flange stay merged
        for (a_lo, a_hi) in _merge_intervals(g["spans"], gap=0.02 * diag):
            _emit_lateral_pad(plan, report, origin, x, y, z, u, v, a,
                              base_reach_u, a_lo, a_hi, tol, consumed)


def _emit_lateral_pad(plan, report, origin, x, y, z, u, v, a,
                      base_reach_u, a_lo, a_hi, tol, consumed=frozenset()):
    """Hull one flange's protruding faces (clipped to the axis window
    [a_lo, a_hi]) into a profile and append the PadOp (design note 36)."""
    from scipy.spatial import ConvexHull, QhullError

    # Gather the profile from PROTRUDING FACES (any surface reaching past
    # the base edge), keeping each face's points from the base edge
    # outward AND within this flange's axis window. Point-wise selection
    # alone fails on coarse meshes: a flange top/underside is often four
    # corner vertices, two of which sit exactly ON the base edge and would
    # be dropped, collapsing the profile to the outer wall line. Face-wise
    # keeps the inner edge; clipping to the beyond-region keeps a
    # part-spanning deck's protruding strip without its bulk.
    Pu: list[np.ndarray] = []
    Pv: list[np.ndarray] = []
    Pa: list[np.ndarray] = []
    for idx, s in enumerate(report.surfaces):
        if idx in consumed:
            continue
        prim = s.fit.primitive
        if isinstance(prim, Plane) and \
                abs(float(np.asarray(prim.normal) @ a)) > 0.9:
            continue                                # end-cap wall (normal
            #  parallel to the pad axis): spans the whole prism, does not
            #  define the cross-section -- its full-height inner points
            #  would corrupt the hull
        q = s.segment.points - origin
        uq = q @ u
        if float(uq.max()) <= base_reach_u + tol:
            continue                                # face does not protrude
        aq = q @ a
        keep = (uq >= base_reach_u - tol) & \
               (aq >= a_lo - tol) & (aq <= a_hi + tol)
        qk = q[keep]
        if len(qk) == 0:
            continue
        Pu.append(qk @ u)
        Pv.append(qk @ v)
        Pa.append(qk @ a)
    if not Pu:
        return
    prof = np.column_stack([np.concatenate(Pu), np.concatenate(Pv)])
    a_c = np.concatenate(Pa)
    if len(prof) < 3:
        return
    min_axis, max_axis = float(a_c.min()), float(a_c.max())
    length = max_axis - min_axis
    if length <= tol:
        return
    try:
        hull = ConvexHull(prof)
    except (QhullError, ValueError):
        return
    loop = prof[hull.vertices].astype(float)
    depth = float(loop[:, 0].max() - base_reach_u)
    if depth <= 3.0 * tol:
        return
    # Bury the inner edge WELL into the base. The overlap is inside
    # already-solid base material so it changes no final geometry, but
    # OCC's boolean fuse (unlike the manifold union the container round-
    # trip uses) leaves the pad a SEPARATE solid when the overlap is a
    # thin sliver -- field-observed on featuretype: a 0.025-deep overlap
    # rebuilt as a detached flange in FreeCAD. A depth-proportional
    # overlap with a part-relative floor gives OCC a solid volume to fuse.
    overlap = max(0.1 * depth, 12.0 * tol)
    # Bury SHAPE-PRESERVINGLY. Sliding an inner vertex along u is only safe
    # when its outward edge is u-parallel (a flat top/underside): sliding
    # the endpoint of a SLANTED edge (a gusset ramp / bevel underside)
    # PIVOTS that edge about its far end, lifting the pad's underside off
    # the true surface (field-observed on featuretype.STL: the bottom bevel
    # started 0.07 early at slope 0.88 instead of 1.0 -- a visible 'random
    # step at the bottom'). So each inner vertex is slid ONLY if every
    # adjacent edge leading outward is u-parallel (or leads to another
    # inner vertex); otherwise the TRUE vertex is kept and a u-parallel
    # stub bridges it into the base at its own v -- the stub region lies
    # inside already-solid base material, so the final geometry is
    # unchanged while OCC still gets its fusion volume.
    thr = base_reach_u + 3.0 * tol
    mu = base_reach_u - overlap
    n_lp = len(loop)
    buried: list[tuple[float, float]] = []
    for i in range(n_lp):
        pu, pv = float(loop[i, 0]), float(loop[i, 1])
        if pu >= thr:
            buried.append((pu, pv))
            continue
        prv = loop[(i - 1) % n_lp]
        nxt = loop[(i + 1) % n_lp]

        def _slide_safe(q, pv=pv):
            # neighbor is itself inner, or the edge to it is u-parallel
            return float(q[0]) < thr or abs(float(q[1]) - pv) <= tol

        if _slide_safe(prv) and _slide_safe(nxt):
            buried.append((mu, pv))            # flat/inner both sides: slide
        elif _slide_safe(prv):
            buried.append((mu, pv))            # bridge stub, then true corner
            buried.append((pu, pv))
        elif _slide_safe(nxt):
            buried.append((pu, pv))            # true corner, then bridge stub
            buried.append((mu, pv))
        else:
            buried.append((mu, pv))            # both edges slanted (rare):
            #  no u-parallel side to bridge from; slide as before
    loop = np.array(buried, dtype=float)
    keepm = np.ones(len(loop), dtype=bool)     # drop consecutive duplicates
    for i in range(len(loop)):
        if np.linalg.norm(loop[i] - loop[i - 1]) < 1e-12:
            keepm[i] = False
    loop = loop[keepm]

    # the profile scalars (u_c, v_c, a_c) are frame-invariant, but the
    # rebuild works in the PLAN frame then transforms to world -- so the
    # pad's basis and anchor must be expressed in the plan frame (u, v, a
    # are world directions here), else the frame transform is applied
    # twice.
    a_pf = np.array([a @ x, a @ y, a @ z])
    u_pf = np.array([u @ x, u @ y, u @ z])
    v_pf = np.array([v @ x, v @ y, v @ z])
    # Right-handed basis: FreeCAD's App.Placement cannot represent a
    # reflected (left-handed) frame -- it silently drops the reflection and
    # mis-places the pad, so the flange floats off the body (field-observed
    # on featuretype). The manifold round-trip applies the full matrix and
    # is unaffected, which is why the container never saw it. Flip v and
    # mirror the profile's v-coordinates: identical geometry, valid
    # placement.
    if float(np.linalg.det(np.column_stack([u_pf, v_pf, a_pf]))) < 0.0:
        v_pf = -v_pf
        loop = loop.copy()
        loop[:, 1] = -loop[:, 1]
    plan.pads.append(PadOp(
        profile=loop_to_sketch(loop, arcs=False),
        length=length, axis=a_pf, plane_origin=min_axis * a_pf,
        plane_u=u_pf, plane_v=v_pf,
        label=f"Lateral pad (depth {depth:g})"))


def _lateral_pad_mesh_veto(plan, report, x, y, z, origin, tol,
                           min_support=0.98, n_grid=12, n_stations=9):
    """The mesh gets the veto (design note 5), applied to lateral pads.

    A lateral pad's profile is the convex hull of every protruding point
    in the plane perpendicular to its axis, extruded over the whole axis
    window. That is only correct when the protrusion is PRISMATIC and
    CONVEX along the axis. A protruding face that surrounds a
    through-window (angle_block.STL's leg), two protrusions at different
    depths sharing one outward direction (which the interval merge hulls
    together), or the gap between disjoint bodies (multibody.stl) are all
    hulled into material the part does not have -- silently, which
    violates the loud-failure doctrine (stress-campaign bucket 2).

    So every emitted lateral pad is checked against the source mesh:
    at interior samples of its claimed volume (eroded off the boundary,
    so tessellation chord error and snap shifts do not vote), the FINAL
    plan's solidness -- the hull minus every planned cut that reaches the
    sample (pockets, holes, cones, cross-holes) -- must AGREE with
    ``mesh.contains``. This is why the veto runs after all planning: the
    sub-level pockets carved into a genuine flange's top
    (featuretype.STL) legitimately empty part of its hull. Agreement is
    two-sided on purpose: hulled air that no cut removes is invented
    material, and a planned cut through volume the mesh says is solid is
    a flange being carved away -- both mean the hull-plus-cuts model does
    not describe this protrusion. Pads under ``min_support`` agreement
    are dropped and reported in ``unplanned``; hand-built reports without
    a mesh keep their pads (nothing to veto against).
    """
    if report.mesh is None:
        return
    from shapely.geometry import Point, Polygon

    L = plan.base.length

    def _cut_reaches(r):
        """Does some planned cut remove plan-frame point r?

        Bounding shapes + tol; boundary effects on either side are
        absorbed by the near-surface leniency below.
        """
        rz = r[2]
        for op in plan.pockets:
            zlo, zhi = ((-tol, L + tol) if op.through
                        else (L - op.depth - tol, L + tol) if op.from_top
                        else (-tol, op.depth + tol))
            if zlo <= rz <= zhi:
                try:
                    poly = Polygon([q for e in op.profile
                                    for q in e.sample()]).buffer(tol)
                    if poly.contains(Point(r[0], r[1])):
                        return True
                except Exception:               # noqa: BLE001 - odd profile
                    pass
        for op in plan.holes:
            reach = op.depth + (op.counterbore_depth or 0.0)
            zlo, zhi = ((-tol, L + tol) if op.through
                        else (L - reach - tol, L + tol) if op.from_top
                        else (-tol, reach + tol))
            if not zlo <= rz <= zhi:
                continue
            rr = max(op.diameter, op.counterbore_diameter or 0.0,
                     op.countersink_diameter or 0.0) / 2.0 + tol
            if any((r[0] - px) ** 2 + (r[1] - py) ** 2 <= rr * rr
                   for px, py in op.positions):
                return True
        for op in plan.cones:
            if op.additive:
                continue
            zlo, zhi = ((-tol, L + tol) if op.through
                        else (L - op.depth - tol, L + tol) if op.from_top
                        else (-tol, op.depth + tol))
            if not zlo <= rz <= zhi:
                continue
            rr = max(op.r_mouth, op.r_far) + tol
            if any((r[0] - px) ** 2 + (r[1] - py) ** 2 <= rr * rr
                   for px, py in op.positions):
                return True
        return False

    def _cross_hole_forgiven(w):
        """Cross-holes are carried in WORLD coordinates."""
        for op in plan.cross_holes:
            a = np.asarray(op.axis, dtype=float)
            rr = op.diameter / 2.0 + tol
            for p3 in op.positions3d:
                rel = w - np.asarray(p3, dtype=float)
                if float(np.linalg.norm(rel - float(rel @ a) * a)) <= rr:
                    return True
        return False

    kept: list = []
    for pad in plan.pads:
        if pad.axis is None:
            kept.append(pad)
            continue
        prof = pad.profile
        if len(prof) == 1 and isinstance(prof[0], SketchCircle):
            poly = Point(*prof[0].center).buffer(float(prof[0].radius),
                                                 quad_segs=24)
        else:
            try:
                poly = Polygon([q for e in prof for q in e.sample()]).buffer(0)
            except Exception:                   # noqa: BLE001
                kept.append(pad)
                continue
        shrunk = poly.buffer(-2.5 * tol)
        if shrunk.is_empty:
            kept.append(pad)                    # too thin to judge
            continue
        minu, minv, maxu, maxv = shrunk.bounds
        pts_plan = []
        ts = (0.06 + 0.88 * np.arange(n_stations) / max(n_stations - 1, 1)) \
            * pad.length
        for uu in np.linspace(minu, maxu, n_grid):
            for vv in np.linspace(minv, maxv, n_grid):
                if not shrunk.contains(Point(uu, vv)):
                    continue
                for t in ts:
                    pts_plan.append(pad.plane_origin + uu * pad.plane_u
                                    + vv * pad.plane_v + t * pad.axis)
        if len(pts_plan) < 40:
            kept.append(pad)                    # insufficient evidence
            continue
        P = np.asarray(pts_plan)
        W = origin + P[:, 0:1] * x + P[:, 1:2] * y + P[:, 2:3] * z
        try:
            inside = report.mesh.contains(W)
        except Exception:                       # noqa: BLE001 - mesh not
            kept.append(pad)                    # queryable: cannot veto
            continue
        cut = np.array([_cut_reaches(P[i]) or _cross_hole_forgiven(W[i])
                        for i in range(len(P))])
        ok = (~cut) == inside                   # plan solidness == mesh
        if not ok.all():
            # near-surface leniency: snapping may shift a face by ~tol.
            # Some trimesh versions need the optional rtree package for
            # proximity queries; the leniency is belt-and-suspenders on
            # top of the 2.5-tol erosion, so degrade to the strict
            # verdict rather than failing the whole rebuild without it.
            try:
                import trimesh as _tm
                bad = np.flatnonzero(~ok)
                _, d, _ = _tm.proximity.closest_point(report.mesh, W[bad])
                ok[bad[d < 2.5 * tol]] = True
            except Exception:                   # noqa: BLE001
                pass
        support = float(ok.mean())
        if support >= min_support:
            kept.append(pad)
        else:
            plan.unplanned.append(
                f"{pad.label}: {1.0 - support:.0%} of the hulled volume is "
                f"not mesh material (windowed or non-prismatic lateral "
                f"protrusion beyond the base outline), skipped")
    plan.pads[:] = kept


def _footprint_polys(mesh, seg, to2d, tol):
    """Exact 2D footprint of a segment as clean shapely polygon(s).

    Projects the segment's own mesh triangles into the frame plane and
    unions them. Triangles that project edge-on (to a sliver) carry no
    area and would form a degenerate ring, so they are dropped -- this is
    what let a curved up-face crash the earlier prototype. Returns one
    polygon per disjoint region; empty list if the mesh is unavailable or
    the projection has no area (the caller then falls back to loops).
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import set_precision
    if mesh is None:
        return []
    try:
        tris = mesh.vertices[mesh.faces[seg.face_indices]]
    except Exception:                           # noqa: BLE001
        return []
    parts = []
    for t in tris:
        q = to2d(t)
        area2 = abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1])
                    - (q[2, 0] - q[0, 0]) * (q[1, 1] - q[0, 1]))
        if area2 < tol * tol:                   # edge-on: sliver, no footprint
            continue
        parts.append(Polygon(q).buffer(0))
    if not parts:
        return []
    try:
        u = set_precision(unary_union(parts), tol * 0.1).buffer(0)
    except Exception:                           # noqa: BLE001
        return []
    if u.is_empty:
        return []
    geoms = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    return [g for g in geoms if not g.is_empty and g.area > tol * tol]


def _plan_terraces(plan, report, members, z, x_ax, y_ax, origin, to2d,
                   base_outer3d, skip=frozenset()):
    """Height-field terrace reconstruction of horizontal faces.

    Each intermediate member plane is cut to its OWN exact projected
    footprint (exterior plus interior holes), so interlocked/nested
    terraces reconstruct without the over- or under-cut that a
    from-outline step or an opening-loop pocket produces. This replaces
    both the outline-touching step planner and the opening-driven
    prismatic-pocket reconstruction FOR HORIZONTAL faces, and closes the
    interior-recess gap (thin-wall edge, nested) that the opening detector
    misses. Cutting every terrace to its exact footprint is safe by
    construction: disjoint footprints don't interfere, deepest-cut-wins
    makes nesting correct, a cut over a raised deck/boss cap only removes
    the air above it, and interior loops become holes so towers survive.
    Proven: 0.0% on a synthetic interlocked-terrace fixture, and
    3.5% -> 0.1% max surface deviation on featuretype.
    """
    from shapely.geometry import Polygon, Point
    L = plan.base.length
    pts_all = np.vstack([s.segment.points for s in report.surfaces])
    tol = max(1e-3 * float(np.linalg.norm(np.ptp(pts_all, axis=0))), 1e-9)
    base2d = to2d(np.asarray(base_outer3d))

    def _exposed(poly, h, up) -> bool:
        """True when this terrace is the TOP (or bottom) surface over its
        footprint -- nothing above (below) it. A plane under an overhang or
        lip has material on its exposure side; cutting it from that side
        would raze the overhang. Verified against the mesh; if the mesh is
        unavailable or the containment test is unreliable, trust the
        terrace (the previous behaviour)."""
        if report.mesh is None:
            return True
        minx, miny, maxx, maxy = poly.bounds
        cand = [(minx + (maxx - minx) * fx, miny + (maxy - miny) * fy)
                for fx in (0.25, 0.5, 0.75) for fy in (0.25, 0.5, 0.75)]
        uv = [(px, py) for px, py in cand if poly.contains(Point(px, py))]
        rp = poly.representative_point()
        uv.append((rp.x, rp.y))
        # a point just off the terrace on its exposure side, in world coords
        off = (h + 5.0 * tol) if up else (h - 5.0 * tol)
        worlds = np.array([origin + u * x_ax + v * y_ax + off * z
                           for u, v in uv])
        try:
            inside = report.mesh.contains(worlds)
        except Exception:                       # noqa: BLE001
            return True
        return float(np.mean(inside)) < 0.3     # exposed if mostly air

    def _raised_loop(loop2d, h, up):
        """True if an interior loop bounds material RISING through the
        terrace (a tower / boss / frame wall -- keep it as a sketch hole so
        it survives the cut) rather than a hole DESCENDING through it (a
        drill / bore -- drop it, or the cut leaves a standing column that
        the hole feature bores into an annular chimney). Probes are taken
        just INSIDE the loop boundary and ray-cast toward the terrace, so a
        thin-walled or hollow frame (air at its centre) still reads as
        raised. Deterministic (fixed-direction ray cast), unlike a
        mesh.contains sample."""
        if report.mesh is None:
            return True
        try:
            poly = Polygon(loop2d)
            c = poly.representative_point()
            cx, cy = c.x, c.y
        except Exception:                       # noqa: BLE001
            return True
        probes = []
        for px, py in loop2d:
            d = float(np.hypot(cx - px, cy - py))
            if d < 1e-9:
                continue
            fr = min(max(2.0 * tol, 0.2) / max(d, 1e-9), 0.49)
            probes.append((px + (cx - px) * fr, py + (cy - py) * fr))
        if not probes:
            return True
        far = (L + 10.0 * tol) if up else (-10.0 * tol)
        oarr = np.array([origin + u * x_ax + v * y_ax + far * z
                         for u, v in probes])
        darr = np.tile((-z if up else z), (len(oarr), 1))
        try:
            locs, ridx, _ = report.mesh.ray.intersects_location(
                oarr, darr, multiple_hits=False)
        except Exception:                       # noqa: BLE001
            return True
        if len(locs) == 0:
            return False                        # nothing above/below: a hole
        hh = (locs - origin) @ z
        above = (np.sum(hh > h + 2.0 * tol) if up
                 else np.sum(hh < h - 2.0 * tol))
        return above > 0.5 * len(probes)

    for _mi, s in members:
        if _mi in skip:
            continue                        # a cone feature's own floor
        h = float(s.fit.primitive.point @ z) - float(origin @ z)
        if h < tol or h > L - tol:
            continue                            # base bottom/top: not a cut
        polys = _footprint_polys(report.mesh, s.segment, to2d, tol)
        if not polys:                           # fallback: boundary loops
            lp = _outer_loop(s.segment)
            if lp is None:
                continue
            try:
                poly = Polygon(
                    to2d(lp),
                    [to2d(il) for il in s.segment.boundary_loops
                     if il is not lp]).buffer(0)
            except Exception:                   # noqa: BLE001
                continue
            polys = ([poly] if (not poly.is_empty and poly.area > tol * tol)
                     else [])
        up = float(s.segment.face_normals.mean(axis=0) @ z) > 0.0
        depth = (L - h) if up else h
        for poly in polys:
            if not _exposed(poly, h, up):
                continue                        # under an overhang: not a
                #  height-field cut (field-observed: idler_riser's lip)
            ext = np.asarray(poly.exterior.coords)[:-1]
            # Grow the exterior OUTWARD a hair so adjacent different-depth
            # terraces OVERLAP rather than share an exact edge -- otherwise
            # FreeCAD PartDesign leaves a thin standing fin along that shared
            # edge (the semicircular pocket, the wavy front). Deepest-cut-wins
            # keeps the result mesh-accurate to within the buffer; this also
            # subsumes the old outline push (outline-coincident edges move out
            # too, clearing the boolean coincidence).
            ext = _buffer_loop(ext, 1.5 * tol)
            ext = _dedupe_loop(ext, 0.5 * tol)   # drop OCC-choking slivers
            if not _valid_loop(ext, tol):
                continue
            holes = [_dedupe_loop(np.asarray(r.coords)[:-1], 0.5 * tol)
                     for r in poly.interiors
                     if _valid_loop(np.asarray(r.coords)[:-1], tol)
                     and _raised_loop(np.asarray(r.coords)[:-1], h, up)]
            label = (f"Terrace to depth {depth:g}"
                     f"{'' if up else ' (from bottom)'}")
            plan.pockets.append(PocketOp(
                profile=loop_to_sketch(ext, arcs=False),
                depth=depth, from_top=up, label=label,
                hole_profiles=[loop_to_sketch(hh, max_sweep_deg=100)
                               for hh in holes]))
            plan.step_labels.append(label)


def _upgrade_to_tapered_pockets(plan, report, z, x, y, origin, to2d,
                                base_outer3d, spread_threshold=0.03):
    """Detect cavities whose cross-section changes with height and replace
    their prismatic PocketOp with a TaperedPocketOp (subtractive loft).

    When the cavity profile at the mouth is significantly different from
    the floor footprint (area spread > threshold), the prismatic cut
    either removes material it should leave (bosses that recede with
    height) or leaves material it should remove (walls that flare
    outward). A loft between the two profiles follows the actual geometry.

    Only top-side pockets are upgraded (the cross-section near z = plan
    length defines the mouth); through pockets, pads, and bottom-side
    pockets are unchanged.
    """
    L = plan.base.length
    mesh = report.mesh
    if mesh is None:
        return
    from shapely.geometry import Polygon
    from .cross_sections import profile_at, area_spread
    # outer pentagon in MESH xy (the base_outer3d is in mesh coords; to2d
    # maps to the plan frame but profile_at works in mesh coords)
    try:
        outer_2d = Polygon(base_outer3d[:, :2].tolist())
    except Exception:                                      # noqa: BLE001
        return

    for k, pk in enumerate(plan.pockets):
        if not getattr(pk, "from_top", True):
            continue
        if getattr(pk, "through", False):
            continue
        depth = float(pk.depth)
        floor_z = L - depth
        # Try several z near the mouth and floor -- coarse STL meshes only
        # yield clean cross-sections at a few discrete heights
        floor = mouth = None
        for dz in (0.0, 0.5, 1.0, -0.5):
            if floor is None:
                floor = profile_at(mesh, floor_z + dz,
                                   outer_pentagon=outer_2d)
        # Mouth: the top ~15 % of the cavity (well above the floor but
        # below the rim blend -- the bosses have fully evolved here).
        mouth_z = L - 0.15 * depth
        for dz in np.arange(-1.0, 1.01, 0.2):
            if mouth is None:
                mouth = profile_at(mesh, float(mouth_z + dz),
                                   outer_pentagon=outer_2d)
        if floor is None or mouth is None:
            continue
        spread = area_spread([floor, mouth])
        if spread < spread_threshold or spread > 0.25:
            # skip both trivially-identical profiles AND geometric
            # steps (cross-section change >25% is a step, not a taper)
            continue
        mouth_loop = np.array(mouth.polygon.exterior.simplify(0.1).coords)
        mouth_2d = to2d(mouth_loop)
        mouth_2d = _dedupe_loop(mouth_2d, 0.01)
        if not _valid_loop(mouth_2d, 1e-6):
            continue
        mpro = loop_to_sketch(mouth_2d, max_sweep_deg=100)
        pk.mouth_profile = mpro


def _plan_bottom_pockets(plan, report, members, feats, z, x, y, origin,
                         to2d):
    """Funnel-shaped through-passages opening on the BOTTOM face.

    The bracket's mounting funnels run from the bottom face up to the
    recess floor (the lowest intermediate plane), NOT to the far z-end, so
    the base through-opening matcher (which only compares opposite extremes
    and demands near-equal area) never sees them -- the funnel tapers, and
    its far side is an intermediate plane. Model each as a ``from_bottom``
    pocket spanning the base-plate thickness (bottom face -> recess floor);
    the recess pocket above already carves the cavity, so these only open
    the plate. Loops that coincide with a drilled hole/counterbore feature
    are skipped (that feature cuts them)."""
    from shapely.geometry import Polygon
    L = plan.base.length
    tol = max(1e-3 * L, 1e-9)
    horiz = []
    for _i, s in members:
        p = s.fit.primitive
        if not hasattr(p, "normal") or abs(float(p.normal @ z)) < 0.9:
            continue
        horiz.append((float((p.point - origin) @ z), s))
    if not horiz:
        return
    base_z = min(zz for zz, _ in horiz)
    above = sorted(zz for zz, _ in horiz if zz > base_z + tol)
    if not above:
        return
    plate_top = above[0]
    plate_thick = plate_top - base_z
    if plate_thick < 0.02 * L or plate_thick > 0.5 * L:
        return
    bottom_plane = next(s for zz, s in horiz if abs(zz - base_z) < tol)
    top_plane = next(s for zz, s in horiz if abs(zz - plate_top) < tol)

    def inner_loops(seg):
        loops = seg.boundary_loops
        if len(loops) < 2:
            return []
        areas = [abs(Polygon(to2d(l[:, :3])).area) for l in loops]
        outer_idx = int(np.argmax(areas))
        return [loops[i] for i in range(len(loops)) if i != outer_idx]

    bot_inners = inner_loops(bottom_plane.segment)
    top_inners = inner_loops(top_plane.segment)
    if not bot_inners or not top_inners:
        return
    top_centroids = [to2d(l[:, :3]).mean(0) for l in top_inners]

    # drilled holes / counterbores / cone pockets already cut their mouths
    drilled = []
    for ff in feats.features:
        if ff.kind in ("hole", "counterbore", "cone_pocket") \
                and "position" in ff.params:
            fc = to2d(np.asarray(ff.params["position"], dtype=float))[0]
            fr = 0.5 * float(ff.params.get("diameter",
                                           ff.params.get("mouth_radius", 0.0)))
            drilled.append((fc, fr))

    def is_drilled(loop2):
        c = loop2.mean(0)
        lr = float(np.mean(np.linalg.norm(loop2 - c, axis=1)))
        for fc, fr in drilled:
            if float(np.linalg.norm(c - fc)) < max(3.0 * tol, 0.5 * fr) \
                    and abs(lr - fr) <= 0.35 * max(fr, tol):
                return True
        return False

    for bl in bot_inners:
        bl2 = to2d(bl[:, :3])
        if not _valid_loop(bl2, tol) or is_drilled(bl2):
            continue
        bc = bl2.mean(0)
        if not any(float(np.linalg.norm(bc - tc)) < 2.0
                   for tc in top_centroids):
            continue
        plan.pockets.append(PocketOp(
            profile=loop_to_sketch(bl2, arcs=False),
            depth=float(plate_thick), from_top=False,
            label=f"Bottom funnel to depth {plate_thick:g}"))
    if any(not getattr(p, "from_top", True) for p in plan.pockets):
        top_pk = [p for p in plan.pockets if getattr(p, "from_top", True)]
        bot_pk = [p for p in plan.pockets
                  if not getattr(p, "from_top", True)]
        plan.pockets[:] = bot_pk + top_pk


def _signed_area(loop2d: np.ndarray) -> float:
    return 0.5 * float(np.dot(loop2d[:, 0], np.roll(loop2d[:, 1], -1))
                       - np.dot(loop2d[:, 1], np.roll(loop2d[:, 0], -1)))


def _plan_gussets(plan, report, z, x, y, origin):
    """Thin triangular reinforcing webs inside a recess.

    These are the UNDER regions the prismatic rebuild misses: a plate that
    rises from the recess floor, pointed at one end and tall at the other
    (a right-triangular side profile). Each is recovered as a LATERAL pad
    whose profile is that triangle and whose extrusion thickness is chosen
    so the prism volume equals the region's true volume -- matching both
    shape and mass without needing the (coarse-mesh-unreliable) loft
    cross-sections. Detection runs the headless rebuild and takes the
    connected components of ``mesh \\ rebuild`` that sit above the base
    plate; anything too small or too large is left to the correction pass.
    """
    from .solidify import plan_to_mesh
    mesh = report.mesh
    if mesh is None:
        return
    L = plan.base.length
    # a web lives on a recess floor: require a from-top pocket to exist and
    # use the lowest such floor as the seat the web must rise from
    recess_floor_z = None
    for pk in plan.pockets:
        if getattr(pk, "from_top", True) and not getattr(pk, "through", False):
            fz = L - float(pk.depth)
            if recess_floor_z is None or fz < recess_floor_z:
                recess_floor_z = fz
    if recess_floor_z is None:
        return
    try:
        solid = plan_to_mesh(plan)
    except Exception:                                      # noqa: BLE001
        return
    if solid is None:
        return
    try:
        under = mesh.difference(solid)
    except Exception:                                      # noqa: BLE001
        return
    span = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    z_floor_min = float(mesh.bounds[0][2]) + 0.1 * span
    for c in under.split():
        if c.volume < 100 or c.volume > 0.2 * float(mesh.volume):
            continue
        if c.bounds[0][2] < z_floor_min:
            continue                       # base-plate feature, not a web
        _emit_gusset(plan, c, z, x, y, origin, recess_floor_z, mesh)


def _emit_gusset(plan, c, z, x, y, origin, recess_floor_z, mesh=None):
    pts = c.vertices
    q = pts - origin
    v_all = q @ z
    floor_v, top_v = float(v_all.min()), float(v_all.max())
    # the web must rise FROM the recess floor (not float mid-cavity)
    if abs(floor_v - recess_floor_z) > 0.05 * plan.base.length:
        return
    # length vs thin axis: the larger horizontal extent is the plate's
    # length, the smaller its thickness direction (plan-frame axes x, y)
    exf = float((q @ x).max() - (q @ x).min())
    eyf = float((q @ y).max() - (q @ y).min())
    if exf > eyf:
        u, a = x, y
    else:
        u, a = y, x
    uq = q @ u
    vq = q @ z
    u_mid = 0.5 * (float(uq.min()) + float(uq.max()))
    lo = uq < u_mid
    hi = ~lo
    vext_lo = float(vq[lo].max() - vq[lo].min()) if lo.any() else 1e9
    vext_hi = float(vq[hi].max() - vq[hi].min()) if hi.any() else 1e9
    if vext_lo < vext_hi:
        pos_pointed_u, pos_tall_u = float(uq.min()), float(uq.max())
    else:
        pos_pointed_u, pos_tall_u = float(uq.max()), float(uq.min())
    base_len = abs(pos_tall_u - pos_pointed_u)
    height = top_v - floor_v
    tri_area = 0.5 * base_len * height
    if tri_area < 1e-6 or base_len < 1e-6:
        return
    eff_thick = float(c.volume) / tri_area
    # a web must be a PLATE: much thinner than it is long
    if eff_thick > 0.5 * base_len:
        return
    # Bury the base edge into the recess floor so the pad fuses with the
    # body instead of floating on a line contact (OCC leaves a line-touching
    # solid detached; field-observed ~40% volume loss without the bury).
    bury = max(0.2, 0.02 * plan.base.length)
    # Extend the tall end to the recess ceiling so the web reaches the
    # full cavity height (the UNDER region omits what the rebuild already
    # captured, so its max-z under-reports). The ceiling is the mesh top
    # in the web's xy region, clamped by the plan height.
    if mesh is not None:
        mx = float((pts @ x).min()), float((pts @ x).max())
        my = float((pts @ y).min()), float((pts @ y).max())
        region = ((mesh.vertices[:, 0] >= mx[0] - 0.1) &
                  (mesh.vertices[:, 0] <= mx[1] + 0.1) &
                  (mesh.vertices[:, 1] >= my[0] - 0.1) &
                  (mesh.vertices[:, 1] <= my[1] + 0.1))
        if region.any():
            ceiling_v = float(mesh.vertices[region, 2].max())
        else:
            ceiling_v = top_v
    else:
        ceiling_v = top_v
    ceiling_v = min(ceiling_v, float(plan.base.length))
    if ceiling_v > top_v + 0.1:
        top_v = ceiling_v
        height = top_v - floor_v
        tri_area = 0.5 * base_len * height
        eff_thick = float(c.volume) / tri_area
        if eff_thick > 0.5 * base_len:
            eff_thick = float(c.volume) / (0.5 * base_len * (top_v - floor_v + 1e-6))
    tri_uv = np.array([[pos_pointed_u, floor_v - bury],
                       [pos_tall_u, floor_v - bury],
                       [pos_tall_u, top_v]])
    if _signed_area(tri_uv) < 0:
        tri_uv = tri_uv[::-1]
    u_pf = np.array([float(u @ x), float(u @ y), float(u @ z)])
    v_pf = np.array([float(z @ x), float(z @ y), float(z @ z)])
    a_pf = np.array([float(a @ x), float(a @ y), float(a @ z)])
    if float(np.linalg.det(np.column_stack([u_pf, v_pf, a_pf]))) < 0.0:
        v_pf = -v_pf
        tri_uv = tri_uv.copy()
        tri_uv[:, 1] = -tri_uv[:, 1]
    thin_center_a = float((pts.mean(axis=0) - origin) @ a)
    min_axis = thin_center_a - 0.5 * eff_thick
    plan.pads.append(PadOp(
        profile=loop_to_sketch(tri_uv, arcs=False),
        length=float(eff_thick), axis=a_pf,
        plane_origin=min_axis * a_pf, plane_u=u_pf, plane_v=v_pf,
        label="Gusset web"))


def _plan_steps(plan, report, members, z, origin, to2d, base_outer3d):
    """Exposed intermediate-height planes touching the part outline are
    STEPS: the base pad must not fill above (or below) them. Exposure
    implies an empty column to the corresponding face, so a through-depth
    pocket over the shelf's own (outward-pushed) outer loop is valid by
    construction. Interior loops (pocket floors, counterbore shoulders)
    never touch the outline and cannot become phantom steps."""
    L = plan.base.length
    pts_all = np.vstack([s.segment.points for s in report.surfaces])
    diag = float(np.linalg.norm(pts_all.max(axis=0) - pts_all.min(axis=0)))
    tol = max(1e-3 * diag, 1e-9)
    base2d = to2d(base_outer3d)
    for _, s in members:
        h = float(s.fit.primitive.point @ z) - float(origin @ z)
        if h < tol or h > L - tol:
            continue                            # bottom/top level: not a step
        lp = _outer_loop(s.segment)
        if lp is None:
            continue
        lp2 = to2d(lp)
        if float(np.min(_dist_to_polyline(lp2, base2d))) > 3.0 * tol:
            continue                            # interior loop: not a step
        up = float(s.segment.face_normals.mean(axis=0) @ z) > 0.0
        depth = (L - h) if up else h
        pushed = _push_outline_vertices(lp2, base2d, 0.05 * depth, 3.0 * tol)
        if not _valid_loop(pushed, tol):
            continue                            # sliver on a noisy mesh:
            #  a degenerate loop would emit a <3-point wire that crashes
            #  the rebuild -- drop this phantom step
        # Interior loops of the shelf split two ways: footprints of
        # RAISED material (a tower in a ring shelf) must become sketch
        # holes or the cut razes them -- but openings of DOWNWARD
        # features (drills, pockets piercing the shelf) must NOT, or the
        # step cut leaves annular chimneys standing at every drill
        # (field-observed). Discriminator: does any surface sharing the
        # loop extend past the shelf on the EXPOSURE side?
        def _raised(hl3) -> bool:
            keys = {tuple(np.round(q, 9)) for q in hl3}
            for other in report.surfaces:
                if other.segment is s.segment:
                    continue
                okeys = {tuple(np.round(q, 9)) for q in other.segment.points}
                if len(keys & okeys) < 2:
                    continue
                oh = (other.segment.points - origin) @ z
                if up and float(oh.max()) > h + 3.0 * tol:
                    return True
                if not up and float(oh.min()) < h - 3.0 * tol:
                    return True
            return False

        hole_profiles = [loop_to_sketch(to2d(hl), arcs=False)
                         for hl in s.segment.boundary_loops
                         if hl is not lp and _raised(hl)]
        label = (f"Step to depth {depth:g}"
                 f"{'' if up else ' (from bottom)'}")
        # polyline profiles for steps: auxiliary cuts where guaranteed
        # closure beats pretty-but-fragile micro-arcs fitted to noisy
        # shelf boundaries (field-observed: "Wire is not closed")
        plan.pockets.append(PocketOp(
            profile=loop_to_sketch(pushed, arcs=False),
            depth=depth, from_top=up, label=label,
            hole_profiles=hole_profiles))
        plan.step_labels.append(label)


def _cw(a_pt, b_pt, c, d2) -> bool:
    """Arc from a to b around c must bulge AWAY from the mouth (-d2)."""
    far = c - 1.0 * d2 * 0.0  # midpoint of the far semicircle is c - r*d2
    # choose the sweep sign whose midpoint lies on the far side
    mid_ccw_ang = np.arctan2(a_pt[1] - c[1], a_pt[0] - c[0]) + np.pi / 2
    mid = c + np.linalg.norm(a_pt - c) * np.array([np.cos(mid_ccw_ang),
                                                   np.sin(mid_ccw_ang)])
    return bool((mid - c) @ d2 > 0)


def _keys(pts: np.ndarray) -> set:
    return {tuple(np.round(p, 9)) for p in np.asarray(pts)}


def _chamfer_op(report, f) -> ChamferOp | None:
    """Sharp edge of a chamfer: the intersection line of its two blended
    planes, using SNAPPED sign-oriented normals (design note 21 applies
    here identically)."""
    pi, ai, bi = f.surface_indices[:3]
    strip = report.surfaces[pi]

    def outward(idx):
        s = report.surfaces[idx]
        n = s.fit.primitive.normal.copy()
        if float(n @ s.segment.face_normals.mean(axis=0)) < 0:
            n = -n
        return n, s.fit.primitive.point

    n_a, p_a = outward(ai)
    n_b, p_b = outward(bi)
    d = np.cross(n_a, n_b)
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        return None
    d = d / nd
    c = strip.segment.points.mean(axis=0)
    try:
        x = np.linalg.solve(np.vstack([n_a, n_b, d]),
                            np.array([float(n_a @ p_a), float(n_b @ p_b),
                                      float(d @ c)]))
    except np.linalg.LinAlgError:
        return None
    n_p, _ = outward(pi)
    size = float(np.sqrt(2.0) * abs(n_p @ (c - x)))
    h = (strip.segment.points - x) @ d
    return ChamferOp(size=size,
                     edge_start=x + float(h.min()) * d,
                     edge_end=x + float(h.max()) * d,
                     direction=d, n_a=n_a, n_b=n_b, label=f.description)


def _fillet_op(report, patches, f, z_dir) -> FilletOp | None:
    """Build a FilletOp for a non-vertical fillet by locating the two
    blended planes and reconstructing the sharp edge they would form."""
    from .primitives import Plane

    ci = f.surface_indices[0]
    seg = report.surfaces[ci].segment
    cyl = report.surfaces[ci].fit.primitive
    d = cyl.axis
    fkeys = _keys(seg.points)

    neighbours = []
    for j, s in enumerate(report.surfaces):
        if not isinstance(s.fit.primitive, Plane):
            continue
        outward = s.fit.primitive.normal.copy()
        if float(outward @ s.segment.face_normals.mean(axis=0)) < 0:
            outward = -outward
        if abs(float(outward @ d)) > 0.75:
            continue  # end caps (near-parallel) or heavily tilted faces
        if len(fkeys & _keys(s.segment.points)) >= 2:
            neighbours.append(outward)
        elif float(np.min(
                np.linalg.norm(seg.points[:, :3, None]
                               - s.segment.points[:, :3].T, axis=1))) < 0.5:
            neighbours.append(outward)
        if len(neighbours) == 2:
            break
    if len(neighbours) != 2:
        return None

    n_a, n_b = neighbours
    sign = 1.0 if f.params.get("convex", True) else -1.0
    shift = sign * cyl.radius * (n_a + n_b)
    v0, v1 = patches[ci].v_range
    origin = patches[ci].origin                  # axis point at v = 0
    return FilletOp(
        radius=float(cyl.radius),
        edge_start=origin + shift,
        edge_end=origin + shift + (v1 - v0) * patches[ci].z_dir,
        direction=patches[ci].z_dir.copy(),
        n_a=n_a, n_b=n_b,
        convex=bool(f.params.get("convex", True)),
        label=f.description)
