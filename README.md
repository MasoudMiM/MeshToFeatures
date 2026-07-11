# MeshToFeatures

**A FreeCAD workbench that reverse-engineers STL meshes of prismatic parts
into editable PartDesign bodies.**

> **Status: beta (0.16.x).** The pipeline is well tested on machined-style
> prismatic parts, but this is a first public release — expect rough edges.
> Bug reports with the offending STL attached are very welcome.

Point it at a triangle mesh of a plate-like machined part and it will:

1. **Segment & recognize** the mesh into analytic surfaces (planes,
   cylinders) by region-growing + geometric fitting,
2. **Snap** the fitted parameters to inferred design intent — unify
   near-parallel directions to canonical axes, merge coaxial cylinders,
   equalize near-equal radii, round values to grid-friendly numbers —
   with a full audit trail of every snap decision in the report view,
3. **Detect features** — holes, counterbores, pockets, steps, pads,
   patterns — and infer a plausible build history,
4. **Rebuild** the part as a native **PartDesign Body**: real sketches,
   pads, pockets, and holes you can edit parametrically.

## Capabilities

Recognized and rebuilt on the current beta:

- **Base solid** from the part footprint (arbitrary polygonal outline,
  including interior cutouts), on parts in any orientation (the working
  frame is detected, not assumed axis-aligned).
- **Terraces / stepped pockets** at multiple depths, including adjacent
  regions at different depths, curved (polygonal-approximated)
  boundaries, and raised islands inside pockets.
- **Drilled holes**: through and blind, including grid **patterns**
  (detected and labeled, e.g. "8x counterbored hole, grid 4x2").
- **Counterbored holes**, including bores whose mouth opens *below* the
  part's top face (e.g. on a plate under a raised deck).
- **Cross-axis holes** (horizontal through-holes).
- **Vertical bosses** and **lateral pads** — flanges, rails, and gusseted
  or beveled wedges protruding sideways off a wall, with slanted
  undersides reproduced at true slope.
- **Fillet detection** with partial rebuild support; **chamfer
  detection** (reported, not yet rebuilt — see limitations).
- **Robust executor**: FreeCAD/OCC boolean failures on individual
  features are retried, deferred to the end of the build, and — only if
  unrecoverable — skipped with a loud, named report instead of silently
  degrading the part.
- **Diagnostics**: `scripts/probe_bore_placement.py` runs the full
  pipeline headlessly inside FreeCAD's Python console and prints
  quantitative per-feature checks (useful when reporting bugs).

## Limitations (please read before filing bugs)

- **Prismatic parts only.** The reconstruction targets parts made of
  planes and cylinders (plates, brackets, housings, fixtures). Organic /
  sculpted / scanned shapes will not reconstruct meaningfully. Spheres,
  cones, and tori are fitted by the core but not yet rebuilt as features.
- **Chamfers and some blends are not rebuilt.** They are detected and
  listed as "surfaces belonging to no recognized feature"; the rebuilt
  body has sharp edges there.
- **Dimensional fidelity is bounded by the mesh.** Snapping tolerance is
  ~0.1% of the part diagonal; a coarse tessellation limits what can be
  recovered. Some internal recess boundaries are intentionally oversized
  by ~0.15% of the part diagonal to avoid degenerate zero-thickness
  shells in OCC booleans — within mesh tolerance, but worth knowing if
  you measure the result.
- **Input mesh quality matters.** Best results on watertight,
  single-solid, CAD-exported STLs. Noisy 3D-scan meshes are untested and
  expected to perform poorly in this release.
- **FreeCAD 1.1+** is the tested target (developed and field-tested on
  1.1.x). Older versions may work but are unverified.
- Very large meshes (millions of faces) have not been performance-tuned.

## Installation

### 1. Python dependencies

The geometry core needs a few packages available **inside FreeCAD's
Python interpreter** (not just your system Python):

```
numpy scipy trimesh shapely
```

See [docs/VERIFY.md](docs/VERIFY.md) for per-platform instructions and a
verification checklist.

### 2. The workbench

**Via the Addon Manager (custom repository):** Edit → Preferences →
Addon Manager → add `https://github.com/MasoudMiM/MeshToFeatures` as a
custom repository, then install *MeshToFeatures* from the Addon Manager
and restart FreeCAD.

**Manually:** download the release zip and extract so that the folder
lands at:

```
~/.local/share/FreeCAD/v1-1/Mod/MeshToFeatures      (Linux)
%APPDATA%/FreeCAD/Mod/MeshToFeatures                (Windows)
~/Library/Application Support/FreeCAD/Mod/...       (macOS)
```

Restart FreeCAD fully after installing or upgrading (it caches Python
modules for the whole session).

## Usage

1. Import your STL (File → Open, or drag-and-drop).
2. Switch to the **MeshToFeatures** workbench.
3. Select the mesh object, then run one of:
   - **Rebuild as PartDesign body** — the full pipeline; creates a
     `Rebuilt_<name>` Body plus a `Reconstruction of <name>` group with
     the recognized surfaces.
   - **Reconstruct surfaces (snapped)** — recognition + snapping only,
     with the audit trail in the Report view.
   - **Reconstruct surfaces (raw fits)** — recognition without snapping,
     for comparing against the snapped result.
   - **MeshToFeatures panel…** — task panel with options and progress.
4. Watch the **Report view** (View → Panels → Report view): it lists
   every recognized pattern/feature, every snap decision, and any
   feature the executor had to retry or skip.

## Project layout

```
meshtofeatures/          geometry core (FreeCAD-free: numpy/scipy/
                         trimesh/shapely) — fitting, segmentation,
                         snapping, feature detection, build planning
freecad/meshtofeatures_wb/  the FreeCAD workbench: commands, task
                         panel, and the PartDesign executor
scripts/                 headless diagnostics (probe)
tests/                   pytest suite (300+ tests, runs without FreeCAD)
docs/                    design notes, development history, verification
```

The core is deliberately FreeCAD-free so the entire planning pipeline is
unit-testable anywhere; only the thin executor layer touches FreeCAD.

## Running the tests

```bash
pip install numpy scipy trimesh shapely manifold3d mapbox-earcut rtree pytest
python -m pytest tests/ -q
```

## Contributing & bug reports

Issues and PRs are welcome. For reconstruction bugs, please attach the
STL (or a minimal reproduction), your FreeCAD version, and the Report
view output — the probe script's output is even better.

## License

LGPL-2.1-or-later. See [LICENSE](LICENSE).
