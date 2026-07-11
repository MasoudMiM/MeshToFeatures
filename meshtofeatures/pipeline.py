# SPDX-License-Identifier: LGPL-2.1-or-later
"""End-to-end v0.1 pipeline: triangle mesh -> recognized analytic surfaces.

This module deliberately has no FreeCAD dependency: it maps a mesh to a
list of (segment, fitted primitive) pairs. Emission into a FreeCAD
document lives in the (separate) workbench adapter, keeping this core
testable headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from .conditioning import condition_mesh
from .fitting import FitResult, fit_best
from .segmentation import (Segment, adaptive_angle_threshold, segment_mesh,
                           split_by_curvature)

__all__ = ["RecognizedSurface", "ReconstructionReport", "reconstruct"]


@dataclass
class RecognizedSurface:
    segment: Segment
    fit: FitResult

    @property
    def kind(self) -> str:
        return self.fit.primitive.kind


@dataclass
class ReconstructionReport:
    surfaces: list[RecognizedSurface] = field(default_factory=list)
    unrecognized: list[Segment] = field(default_factory=list)
    #: fraction of total mesh area covered by recognized surfaces
    coverage: float = 0.0
    #: the (conditioned) mesh segmentation ran on, retained so downstream
    #: planning can project exact per-segment face footprints via
    #: ``segment.face_indices``. May be None for hand-built reports.
    mesh: "trimesh.Trimesh | None" = None

    def kinds(self) -> list[str]:
        return sorted(s.kind for s in self.surfaces)

    def by_kind(self, kind: str) -> list[RecognizedSurface]:
        return [s for s in self.surfaces if s.kind == kind]


def reconstruct(
    mesh: trimesh.Trimesh,
    angle_threshold: float | None = None,
    fit_tolerance: float | None = None,
    accept_rms: float | None = None,
    min_faces: int = 1,
    condition: bool = True,
    refine: bool = True,
    max_refine_depth: int = 1,
    refine_split_max: float | None = None,
    progress=None,
) -> ReconstructionReport:
    """Segment ``mesh`` and fit an analytic primitive to every segment.

    Parameters
    ----------
    mesh:
        Input triangle mesh.
    angle_threshold:
        Dihedral angle (radians) separating smooth regions; see
        :func:`meshtofeatures.segmentation.segment_mesh`.
    fit_tolerance:
        RMS gate for preferring simpler models during selection. Defaults
        to 1e-4 of the mesh bounding-box diagonal.
    accept_rms:
        Segments whose best fit exceeds this RMS are reported as
        ``unrecognized`` instead of being force-labelled. Defaults to 1e-2
        of the mesh bounding-box diagonal.
    min_faces:
        Segments with fewer faces are ignored entirely.
    refine_split_max:
        Cap (radians) on the dihedral threshold used by the refinement
        split. A tangent-merged blob -- a flat plane joined to a chamfer
        or curved perimeter -- has no internal edge as sharp as a loose
        feature-edge ``angle_threshold`` (up to 60 deg on a coarse
        export), so at that threshold it never breaks and its flat core is
        lost. Capping the split (default 30 deg) separates the plane from
        the curved remainder; parts whose ``angle_threshold`` is already
        below the cap are unchanged.
    """
    # progress reporting: stage + fraction, forced monotonic (the
    # refinement queue grows while draining, so a naive processed/total
    # would regress)
    _last = [0.0]

    def _p(stage: str, frac: float) -> None:
        if progress is None:
            return
        _last[0] = max(_last[0], min(max(frac, 0.0), 1.0))
        progress(stage, _last[0])

    _p("conditioning mesh", 0.0)
    if condition:
        mesh, _ = condition_mesh(mesh)
    _p("segmenting", 0.05)
    if angle_threshold is None:
        angle_threshold = adaptive_angle_threshold(mesh)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) if len(mesh.vertices) else 0.0
    if fit_tolerance is None:
        fit_tolerance = max(1e-4 * diag, 1e-12)
    if accept_rms is None:
        accept_rms = max(1e-2 * diag, 1e-9)

    report = ReconstructionReport()
    report.mesh = mesh
    recognized_area = 0.0
    unrecognized_area = 0.0

    # The refinement split must be TIGHTER than the feature-edge threshold:
    # a tangent-merged blob (a flat plane smoothly joined to a chamfer or
    # curved perimeter) has no internal edge as sharp as ``angle_threshold``
    # (which can be 60 deg on a coarse export), so at that threshold it
    # never breaks and its flat core is lost. Capping the split at 30 deg
    # separates the flat plane from the curved/chamfered remainder;
    # fine-tessellation parts (threshold already < 30 deg) are unchanged.
    refine_threshold = min(angle_threshold,
                           refine_split_max if refine_split_max is not None
                           else np.deg2rad(30.0))

    # (segment, depth) work queue: segments that fail primitive fitting at
    # depth < max_refine_depth are re-split by curvature (tangent blends
    # have no sharp edges; curvature contrast is the only signal) and their
    # children re-enter the queue.
    queue: list[tuple[Segment, int]] = [
        (s, 0) for s in segment_mesh(mesh, angle_threshold=angle_threshold,
                                     min_faces=min_faces)
    ]
    processed = 0
    while queue:
        seg, depth = queue.pop()
        processed += 1
        _p(f"fitting surfaces ({processed})",
           0.10 + 0.88 * processed / (processed + len(queue)))
        fit = None
        try:
            fit = fit_best(
                seg.points, seg.normals,
                tolerance=fit_tolerance,
                score_points=seg.samples,
            )
        except (ValueError, np.linalg.LinAlgError):
            pass
        # Acceptance is judged on the segment VERTICES, not the sampled
        # surface: a correct primitive interpolates the vertices of any
        # tessellation, however coarse (vertex rms ~ 0 while sample rms
        # carries the chord sag), whereas a mislabelled noisy patch has
        # vertex rms at the noise level. Model *selection* inside
        # fit_best still uses the samples, which is what disambiguates
        # vertex-degenerate impostors (two rings on a sphere).
        vertex_rms = None
        if fit is not None:
            d = fit.primitive.distance(seg.points)
            vertex_rms = float(np.sqrt(np.mean(d * d)))
        # A fit is only evidence when comfortably overdetermined. A single
        # triangle "fits" a plane exactly (3 points, 3 dof) by pure
        # interpolation, and a handful of noisy points lets a flexible
        # curved surface bend through them (vertex rms far below sample
        # rms is the overfit signature). Planes are rigid: dof + 1 points
        # suffice. Curved primitives must earn acceptance with 2 * dof.
        overdetermined = False
        if fit is not None:
            need = fit.primitive.dof + 1 if fit.primitive.kind == "plane" \
                else 2 * fit.primitive.dof
            overdetermined = len(seg.points) >= need
        if overdetermined and vertex_rms is not None and vertex_rms <= accept_rms:
            # Suspicious accept: a clean tessellation fits its true
            # primitive to ~machine precision, so a vertex rms far above
            # fit_tolerance on a *passing* fit smells like a compromise
            # model over tangent-merged surfaces (e.g. pocket floor +
            # concave fillets swallowed by one shallow cylinder). Try the
            # curvature split first; genuinely noisy meshes have
            # accept_rms >> 20 * fit_tolerance ratios too, but their
            # splits shatter and are refused, falling back to acceptance.
            if refine and depth < max_refine_depth \
                    and vertex_rms > 20.0 * fit_tolerance:
                children = split_by_curvature(
                    mesh, seg.face_indices, refine_threshold,
                    min_faces=min_faces)
                if len(children) > 1:
                    queue.extend((c, depth + 1) for c in children)
                    continue
            report.surfaces.append(RecognizedSurface(segment=seg, fit=fit))
            recognized_area += seg.area
            continue
        if refine and depth < max_refine_depth:
            children = split_by_curvature(
                mesh, seg.face_indices, refine_threshold, min_faces=min_faces)
            if len(children) > 1:
                queue.extend((c, depth + 1) for c in children)
                continue
        report.unrecognized.append(seg)
        unrecognized_area += seg.area

    total = recognized_area + unrecognized_area
    report.coverage = (recognized_area / total) if total > 0 else 0.0
    report.surfaces.sort(key=lambda s: s.segment.area, reverse=True)
    _p("done", 1.0)
    return report
