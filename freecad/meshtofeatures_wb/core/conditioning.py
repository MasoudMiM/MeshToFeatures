# SPDX-License-Identifier: LGPL-2.1-or-later
"""Mesh conditioning: make real-world meshes safe for the pipeline.

STL files routinely arrive as *triangle soup* -- every face owning three
private vertices -- in which case the face-adjacency graph is empty and
segmentation sees disconnected confetti. Booleans and scans additionally
produce degenerate (zero-area) faces. Conditioning welds coincident
vertices, drops degenerate faces, and removes orphaned vertices, and
reports what it did so the operation is auditable.

:func:`repair_solid` is the sibling pass for *boolean output* meshes
(solidify): those carry ~zero-volume sliver components and cracked edges
that make the resulting OCC shell invalid (issue #7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

__all__ = ["ConditioningReport", "RepairReport", "condition_mesh",
           "repair_solid"]


@dataclass
class ConditioningReport:
    vertices_merged: int = 0
    faces_removed: int = 0

    @property
    def touched(self) -> bool:
        return self.vertices_merged > 0 or self.faces_removed > 0


def condition_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, ConditioningReport]:
    """Return a conditioned copy of ``mesh`` plus a report.

    The input is never mutated.
    """
    out = mesh.copy()
    n_vertices = len(out.vertices)
    n_faces = len(out.faces)

    out.merge_vertices()
    vertices_merged = n_vertices - len(out.vertices)

    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    faces_removed = n_faces - len(out.faces)

    out.remove_unreferenced_vertices()
    return out, ConditioningReport(
        vertices_merged=vertices_merged, faces_removed=faces_removed)


@dataclass
class RepairReport:
    """What :func:`repair_solid` changed (auditable)."""

    normalized: bool = False           # round-tripped through the engine
    components_found: int = 0
    components_dropped: int = 0
    volume_before: float = 0.0
    volume_after: float = 0.0


def _manifold_normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Round-trip a mesh through the manifold3d engine.

    The engine's constructor is exact: it welds float32 cracks, resolves
    coincident shells by exact interpretation, and guarantees a watertight
    manifold output -- the debris that makes an OCC shell invalid cannot
    survive the round-trip.
    """
    from manifold3d import Manifold, Mesh
    mf = Manifold(mesh=Mesh(
        vert_properties=np.array(mesh.vertices, dtype=np.float32),
        tri_verts=np.array(mesh.faces, dtype=np.uint32)))
    back = mf.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(back.vert_properties, dtype=float),
        faces=np.asarray(back.tri_verts), process=False)


def repair_solid(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, RepairReport]:
    """Normalize a boolean-output mesh so it converts cleanly to a BRep.

    Mesh booleans round-trip through float32 (trimesh's manifold engine),
    which leaves debris on the result: ~zero-volume sliver components
    touching the real solid, cracked edges where coincident shells share
    a geometric edge, and micro-degenerate faces. A shell built from such
    a mesh is topologically invalid for OCC (issue #7): ``Part.Solid``
    fails and the raw shell -- installed as a Compound -- integrates to a
    wrong volume.

    The pass: (1) round-trip through the manifold engine when available
    (watertight, manifold, exact solid interpretation; without the engine
    the input is returned as-is -- best effort), (2) drop micro-debris
    components with |volume| below 5e-5 of the largest component --
    ``plan_corrections`` never emits patches smaller than 1e-4 of the
    source volume, so legitimate material always survives -- and (3)
    union the surviving components into one solid. The input is never
    mutated.
    """
    out = mesh.copy()
    report = RepairReport(volume_before=float(out.volume))
    try:
        out = _manifold_normalize(out)
        report.normalized = True
    except Exception:                                  # noqa: BLE001
        pass
    comps = out.split(only_watertight=False)
    report.components_found = len(comps)
    if comps:
        biggest = max(abs(float(c.volume)) for c in comps)
        floor = max(5e-5 * biggest, 1e-6)
        keep = [c for c in comps if abs(float(c.volume)) >= floor]
        if not keep:                                   # all slivers: keep
            keep = [max(comps, key=lambda c: abs(float(c.volume)))]
            # the largest, even if tiny -- never an empty result
        report.components_dropped = len(comps) - len(keep)
        out = keep[0]
        for c in keep[1:]:
            out = out.union(c)
    report.volume_after = float(out.volume)
    return out, report
