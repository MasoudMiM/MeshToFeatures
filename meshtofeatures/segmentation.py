# SPDX-License-Identifier: LGPL-2.1-or-later
"""Mesh segmentation into smooth regions separated by sharp edges.

v0.1 strategy: region growing on the face-adjacency graph. Two adjacent
faces belong to the same region iff the dihedral angle between them is
below ``angle_threshold``. On tessellated analytic surfaces this groups:

* every planar face set into one region (dihedral ~ 0),
* smooth curved surfaces (cylinder barrels, spheres, cones) into one
  region, because adjacent facets of a reasonable tessellation deviate by
  only a few degrees,

while genuine feature edges (e.g. the 90-degree cap/barrel junction) stay
region boundaries.

The angle threshold must exceed the tessellation's facet-to-facet angle
(360/sections for a revolved surface) and stay below the smallest real
feature angle. The default of 30 degrees handles tessellations of >= 12
sections and features >= 45 degrees; it is exposed as a parameter for
tuning and for the future learned segmenter to replace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

__all__ = ["Segment", "segment_mesh", "build_segment",
           "adaptive_angle_threshold", "face_curvature",
           "split_by_curvature"]


@dataclass
class Segment:
    """A contiguous set of mesh faces presumed to sample one surface.

    ``points`` and ``normals`` are *paired* per-vertex arrays: each normal
    is the average of the normals of the segment's own faces incident to
    that vertex. Restricting the averaging to the segment is essential:
    ordinary mesh vertex normals are blended across sharp edges and would
    corrupt direction estimation.
    """

    face_indices: np.ndarray   # (k,) indices into mesh.faces
    points: np.ndarray         # (m,3) unique vertex positions of the segment
    normals: np.ndarray        # (m,3) segment-restricted vertex normals
    face_normals: np.ndarray   # (k,3) raw face normals
    #: dense on-surface samples (face centroids + edge midpoints). Mesh
    #: vertices can occupy degenerate positions (e.g. a cylinder barrel's
    #: vertices lie on two circles, hence on a common *sphere*); samples in
    #: face interiors disambiguate model selection.
    samples: np.ndarray        # (4k,3)
    #: closed boundary loops (outer outline + one per hole), each (n,3),
    #: winding inherited from the mesh faces; [] for closed surfaces
    boundary_loops: list[np.ndarray]
    area: float

    def __len__(self) -> int:
        return len(self.face_indices)


def _segment_vertex_normals(
    mesh: trimesh.Trimesh, faces: np.ndarray, vertex_ids: np.ndarray
) -> np.ndarray:
    """Average face normals per vertex, using only ``faces`` of the segment."""
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[vertex_ids] = np.arange(len(vertex_ids))
    acc = np.zeros((len(vertex_ids), 3))
    tri = mesh.faces[faces]                    # (k, 3) vertex ids
    fn = mesh.face_normals[faces]              # (k, 3)
    for corner in range(3):
        np.add.at(acc, remap[tri[:, corner]], fn)
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return acc / norms


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        # path halving
        p = self.parent
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _boundary_loops(mesh: trimesh.Trimesh, faces: np.ndarray) -> list[np.ndarray]:
    """Closed boundary loops of a face subset, as (k,3) vertex arrays.

    A *directed* edge is a boundary edge of the segment iff its reverse is
    not used by any segment face. Chaining these preserves the face
    winding, so (for consistently wound meshes) the outer loop runs
    counter-clockwise and hole loops clockwise when viewed along the
    outward normal -- the orientation the emission planner relies on.
    """
    tri = mesh.faces[faces]
    directed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    fwd = set(map(tuple, directed.tolist()))
    nxt: dict[int, int] = {}
    for a, b in fwd:
        if (b, a) not in fwd:
            nxt[a] = b  # pinch vertices (two outgoing) keep one arbitrarily

    loops: list[np.ndarray] = []
    visited: set[int] = set()
    for start in list(nxt):
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        cur = nxt[start]
        while cur != start and cur in nxt and cur not in visited:
            chain.append(cur)
            visited.add(cur)
            cur = nxt[cur]
        if cur == start and len(chain) >= 3:
            loops.append(mesh.vertices[np.array(chain)])
    return loops


def segment_mesh(
    mesh: trimesh.Trimesh,
    angle_threshold: float = np.deg2rad(30.0),
    min_faces: int = 1,
) -> list[Segment]:
    """Partition ``mesh`` into smooth regions.

    Parameters
    ----------
    mesh:
        Input triangle mesh.
    angle_threshold:
        Maximum dihedral angle (radians) between adjacent faces for them to
        be merged into the same region.
    min_faces:
        Regions with fewer faces are discarded (tessellation debris).

    Returns
    -------
    Segments sorted by area, largest first.
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []

    uf = _UnionFind(n_faces)
    adjacency = mesh.face_adjacency            # (e, 2) face index pairs
    angles = mesh.face_adjacency_angles        # (e,) dihedral angles, >= 0
    for (fa, fb) in adjacency[angles < angle_threshold]:
        uf.union(int(fa), int(fb))

    roots = np.fromiter((uf.find(i) for i in range(n_faces)), dtype=np.int64)
    segments: list[Segment] = []
    for root in np.unique(roots):
        faces = np.flatnonzero(roots == root)
        if len(faces) < min_faces:
            continue
        segments.append(build_segment(mesh, faces))
    segments.sort(key=lambda s: s.area, reverse=True)
    return segments


def build_segment(mesh: trimesh.Trimesh, faces: np.ndarray) -> Segment:
    """Construct a :class:`Segment` from a face-index subset of ``mesh``."""
    vertex_ids = np.unique(mesh.faces[faces].ravel())
    tri = mesh.vertices[mesh.faces[faces]]          # (k, 3, 3)
    samples = np.concatenate([
        tri.mean(axis=1),                           # centroids
        0.5 * (tri[:, 0] + tri[:, 1]),              # edge midpoints
        0.5 * (tri[:, 1] + tri[:, 2]),
        0.5 * (tri[:, 2] + tri[:, 0]),
    ])
    return Segment(
        face_indices=faces,
        points=mesh.vertices[vertex_ids],
        normals=_segment_vertex_normals(mesh, faces, vertex_ids),
        face_normals=mesh.face_normals[faces],
        samples=samples,
        boundary_loops=_boundary_loops(mesh, faces),
        area=float(mesh.area_faces[faces].sum()),
    )


def adaptive_angle_threshold(
    mesh: trimesh.Trimesh,
    floor: float = np.deg2rad(15.0),
    ceiling: float = np.deg2rad(60.0),
    default: float = np.deg2rad(30.0),
) -> float:
    """Choose a segmentation angle threshold from the mesh's own dihedral
    distribution.

    On tessellated CAD parts the dihedral angles form two populations:
    small angles inside smooth surfaces (the tessellation angle -- 5.6 deg
    for 64 sections, 45 deg for 8) and large angles at feature edges. The
    threshold belongs in the gap between them; a fixed value cannot serve
    both fine and coarse exports. Coplanar angles (< 2 deg) are ignored,
    the largest remaining gap is located, and its midpoint is returned,
    clamped to [floor, ceiling]. Without a significant gap (featureless or
    single-population meshes) ``default`` is returned.
    """
    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    angles = np.sort(angles[angles > np.deg2rad(2.0)])
    if len(angles) < 4:
        return default
    gaps = np.diff(angles)
    i = int(np.argmax(gaps))
    if gaps[i] < np.deg2rad(8.0):
        return default
    return float(np.clip(0.5 * (angles[i] + angles[i + 1]), floor, ceiling))


def _internal_adjacency(
    mesh: trimesh.Trimesh, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjacency pairs internal to a face subset, with dihedral angles and
    the *edge-perpendicular span* of each pair.

    The span is the sum of the two centroids' perpendicular distances to
    the shared edge line -- the width over which the surface actually
    bends. Dividing angles by centroid *distance* instead is wrong on
    anisotropic triangulations (tall thin wall quads dilute curvature by
    an order of magnitude, hiding fillets); the perpendicular span is
    invariant to triangle aspect ratio.
    """
    in_subset = np.zeros(len(mesh.faces), dtype=bool)
    in_subset[faces] = True
    adj = mesh.face_adjacency
    both = in_subset[adj[:, 0]] & in_subset[adj[:, 1]]
    pairs = adj[both]
    angles = mesh.face_adjacency_angles[both]
    edges = mesh.face_adjacency_edges[both]          # (e, 2) vertex ids
    p = mesh.vertices[edges[:, 0]]
    u = mesh.vertices[edges[:, 1]] - p
    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
    u_norm[u_norm == 0.0] = 1.0
    u = u / u_norm
    centers = mesh.triangles_center
    span = np.zeros(len(pairs))
    for col in (0, 1):
        v = centers[pairs[:, col]] - p
        span += np.linalg.norm(v - np.einsum("ij,ij->i", v, u)[:, None] * u, axis=1)
    return pairs, angles, span


def face_curvature(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    """Discrete curvature proxy per face of the subset: the mean, over
    adjacent subset faces, of dihedral angle divided by centroid distance.

    Planes give ~0; a surface of radius r gives O(1/r), because adjacent
    facets around the curved direction turn by ``arc / r`` over a centroid
    spacing of ``arc``. The proxy needs no parameters and is what lets a
    tangent fillet (kappa = 1/r) be separated from the plane it blends
    into (kappa = 0) when dihedral angles alone cannot.
    """
    pairs, angles, span = _internal_adjacency(mesh, faces)
    span = span.copy()
    span[span == 0.0] = np.inf
    k_edge = angles / span

    remap = np.full(len(mesh.faces), -1, dtype=np.int64)
    remap[faces] = np.arange(len(faces))
    acc = np.zeros(len(faces))
    cnt = np.zeros(len(faces))
    for col in (0, 1):
        np.add.at(acc, remap[pairs[:, col]], k_edge)
        np.add.at(cnt, remap[pairs[:, col]], 1.0)
    cnt[cnt == 0.0] = 1.0
    return acc / cnt


def split_by_curvature(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    angle_threshold: float,
    min_faces: int = 1,
) -> list[Segment]:
    """Second-pass split of a segment that failed primitive fitting.

    Faces are clustered by the curvature proxy (1D gap clustering), then
    connected components are formed among adjacent faces of the same
    curvature cluster (still respecting the dihedral threshold). Tangent
    blends have no sharp edge, but their curvature differs from their
    neighbours' -- that difference is the only signal available.
    """
    from .snapping import cluster_scalars  # local import to avoid a cycle

    faces = np.asarray(faces)
    if len(faces) < 2:
        return []
    # Face curvature for CLASSIFICATION uses only tessellation-scale
    # edges (< 20 deg): a feature edge that slipped the merge threshold
    # (e.g. a 45-deg chamfer edge under an adaptive threshold of 60)
    # would otherwise pollute its border faces with high k, spawning
    # thin "curved" ribbons that trip the fragmentation refusal.
    pairs_a, angles_a, span_a = _internal_adjacency(mesh, faces)
    small = angles_a < np.deg2rad(20.0)
    remap = np.full(len(mesh.faces), -1, dtype=np.int64)
    remap[faces] = np.arange(len(faces))
    sp = span_a[small].copy()
    sp[sp == 0.0] = np.inf
    ke = angles_a[small] / sp
    acc = np.zeros(len(faces))
    cnt = np.zeros(len(faces))
    for col in (0, 1):
        np.add.at(acc, remap[pairs_a[small][:, col]], ke)
        np.add.at(cnt, remap[pairs_a[small][:, col]], 1.0)
    cnt[cnt == 0.0] = 1.0
    k = acc / cnt

    pts = mesh.vertices[np.unique(mesh.faces[faces].ravel())]
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    k_floor = 2.0 / max(diag, 1e-12)
    curved = k[k > k_floor]
    k_atol = max(k_floor, 0.25 * float(np.percentile(curved, 90)) if len(curved) else k_floor)
    labels = cluster_scalars(k, atol=k_atol)

    # A blob of merged PLANES is invisible to curvature (k ~ 0 across the
    # board) yet still needs splitting -- 45-deg chamfer edges vs 45-deg
    # coarse-tessellation facets are inherently ambiguous by dihedral
    # angle, and the disambiguation lives here (a coarse cylinder FITS
    # and never reaches this splitter). Sub-cluster the lowest-k
    # population by face-normal direction.
    from .snapping import cluster_directions
    means = {int(lab): float(k[labels == lab].mean()) for lab in np.unique(labels)}
    low = min(means, key=means.get)
    flat = (labels == low) & np.array([means[low] <= 3.0 * k_floor] * len(k))
    if np.count_nonzero(flat) >= 2:
        nrm = mesh.face_normals[faces][flat]
        w = mesh.area_faces[faces][flat]
        nlabels, _ = cluster_directions(nrm, w, np.deg2rad(10.0))
        combined = labels.astype(np.int64).copy()
        offset = int(labels.max()) + 1
        combined[flat] = offset + nlabels
        labels = combined

    remap = np.full(len(mesh.faces), -1, dtype=np.int64)
    remap[faces] = np.arange(len(faces))
    all_pairs, all_angles, _ = _internal_adjacency(mesh, faces)
    pairs = all_pairs[all_angles < angle_threshold]

    uf = _UnionFind(len(faces))
    for fa, fb in pairs:
        a, b = remap[fa], remap[fb]
        if labels[a] == labels[b]:
            uf.union(int(a), int(b))
    roots = np.fromiter((uf.find(i) for i in range(len(faces))), dtype=np.int64)
    groups = [faces[np.flatnonzero(roots == root)] for root in np.unique(roots)]
    groups = [g for g in groups if len(g) >= min_faces]

    # Fragmentation refusal: if the split shatters the segment instead of
    # finding a few coherent sub-surfaces, it has found noise, not
    # structure -- refuse, so the parent is reported honestly
    # unrecognized rather than emitted as primitive confetti.
    if len(groups) > 1:
        sizes = np.array([len(g) for g in groups])
        too_many = len(groups) > max(16, len(faces) // 25)
        too_small = float(np.median(sizes)) < 2.0
        if too_many or too_small:
            return []

    out = [build_segment(mesh, g) for g in groups]
    out.sort(key=lambda s: s.area, reverse=True)
    return out
