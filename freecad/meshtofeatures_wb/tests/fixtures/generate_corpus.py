# SPDX-License-Identifier: LGPL-2.1-or-later
"""Regenerate the regression corpus STLs in this directory.

Each part is built by exact CSG (boxes, cylinders, cones, an extruded
stepped profile) so its ground-truth feature inventory and volume are
known. The emitted binary STLs are committed; run this script only to
reproduce them after an intentional generator change:

    .venv/bin/python freecad/meshtofeatures_wb/tests/fixtures/generate_corpus.py

The corpus is the target of ``tests/test_corpus.py``: every part must
flow through the full pipeline, reproduce its analytic feature set, and
close the volume gate (corrected mesh volume within 1% of the mesh
volume).
"""

from pathlib import Path

import numpy as np
import shapely.geometry as sg
import trimesh

_OUT = Path(__file__).parent
_D2R = np.pi / 180.0


def _save(mesh, name):
    mesh.export(str(_OUT / name), file_type="stl")
    print(f"{name}: watertight={mesh.is_watertight} volume={mesh.volume:.1f}")


def _cyl(r, h, center, axis="z"):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=64)
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(_D2R * 90, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(_D2R * 90, [1, 0, 0]))
    m.apply_translation(center)
    return m


def _box(sx, sy, sz, center):
    m = trimesh.creation.box(extents=[sx, sy, sz])
    m.apply_translation(center)
    return m


def _profile_mesh(poly2d, depth, y_center=0.0):
    """Extrude an XZ profile (2D polygon, y = height) along Y to `depth`,
    centered on y = y_center."""
    m = trimesh.creation.extrude_polygon(poly2d, depth)
    # extrusion ran along +Z: (x, h, t) -> (x, t, h)
    m.apply_transform(trimesh.transformations.rotation_matrix(_D2R * 90, [1, 0, 0]))
    m.apply_translation([0.0, y_center, 0.0])
    return m


def plate_counterbored():
    """60x40x10 plate, four Ø8 through bores, two of them counterbored
    Ø14 x 5 deep from the top face."""
    plate = _box(60, 40, 10, [0, 0, 0])
    tool = trimesh.util.concatenate([_cyl(4.0, 30, [x, y, 0])
                                     for x, y in [(20, 12), (20, -12),
                                                  (-20, 12), (-20, -12)]])
    for x, y in [(20, 12), (20, -12)]:
        tool = tool.union(_cyl(7.0, 5.0, [x, y, 2.5]))
    return plate.difference(tool)


def block_countersunk():
    """50x50x20 block, two Ø10 through bores with 90° Ø18 countersinks on
    the top face, one Ø12 blind bore 8 deep."""
    block = _box(50, 50, 20, [0, 0, 0])
    tool = trimesh.util.concatenate([_cyl(5.0, 40, [x, 0, 0]) for x in (15, -15)])
    for x in (15, -15):
        sink = trimesh.creation.cone(radius=9.0, height=9.0, sections=64)
        sink.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        sink.apply_translation([x, 0, 10.0])     # base (Ø18) at z=10, apex at z=1
        tool = tool.union(sink)
    tool = tool.union(_cyl(6.0, 8.0, [0, 15, 6.0]))
    return block.difference(tool)


def bossed_plate():
    """80x50x8 plate, a 30x20x12 boss on the top face, four Ø6 through
    bores (two through boss + plate, two through plate only) and a
    36x10 through slot."""
    part = _box(80, 50, 8, [0, 0, 0]).union(_box(30, 20, 12, [0, 0, 10]))
    tool = trimesh.util.concatenate([
        _cyl(3.0, 40, [7, 0, 0]), _cyl(3.0, 40, [-7, 0, 0]),
        _cyl(3.0, 40, [33, 0, 0]), _cyl(3.0, 40, [-33, 0, 0]),
        _box(36, 10, 40, [0, 17, 0]),
    ])
    return part.difference(tool)


def cross_holed_block():
    """40x40x40 block: one Ø8 Z through bore, one Ø10 Y through bore
    (crossing it), one Ø8 blind X bore 15 deep from the x=-20 face."""
    block = _box(40, 40, 40, [0, 0, 0])
    tool = _cyl(4.0, 60, [0, 0, 0])
    tool = tool.union(_cyl(5.0, 60, [0, 0, 20], axis="y"))
    tool = tool.union(_cyl(4.0, 15.0, [-12.5, 0, 10], axis="x"))
    return block.difference(tool)


def stepped_chamfered():
    """Stepped block (60/40/20 wide, 6 high each) extruded 40 in Y, with
    2 mm chamfers on the top corners of the highest step and one Ø6 Z
    through bore at the center."""
    poly = sg.Polygon([
        (-30, 0), (30, 0), (30, 6), (20, 6), (20, 12), (10, 12),
        (10, 16), (8, 18), (-8, 18), (-10, 16), (-10, 12), (-20, 12),
        (-20, 6), (-30, 6),
    ])
    part = _profile_mesh(poly, 40, y_center=20.0)
    return part.difference(_cyl(3.0, 60, [0, 0, 0]))


def main():
    _save(plate_counterbored(), "corpus_plate_counterbored.stl")
    _save(block_countersunk(), "corpus_block_countersunk.stl")
    _save(bossed_plate(), "corpus_bossed_plate.stl")
    _save(cross_holed_block(), "corpus_cross_holed_block.stl")
    _save(stepped_chamfered(), "corpus_stepped_chamfered.stl")


if __name__ == "__main__":
    main()
