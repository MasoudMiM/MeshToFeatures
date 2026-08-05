# SPDX-License-Identifier: LGPL-2.1-or-later
"""Mesh conditioning: make real-world meshes safe for the pipeline.

STL files routinely arrive as *triangle soup* -- every face owning three
private vertices -- in which case the face-adjacency graph is empty and
segmentation sees disconnected confetti. Booleans and scans additionally
produce degenerate (zero-area) faces. Conditioning welds coincident
vertices, drops degenerate faces, and removes orphaned vertices, and
reports what it did so the operation is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh

__all__ = ["ConditioningReport", "condition_mesh"]


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
