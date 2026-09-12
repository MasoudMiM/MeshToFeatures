# Changelog

## 0.17.4 — unreleased

Issue #6 follow-up: the corner bracket's freeform diagonal band and base
scoops stay carried by the deviation-correction patches (documented as
the intended answer for this geometry class), and the r1.5 band fillets
that were detected but dropped are now rebuilt.

### Added

- **Geometric blend fallback in the executor.** A detected fillet whose
  sharp edge matches no edge of the parametric body (the freeform-band
  class: the tree approximates the band, so no edge exists at the
  mesh-fit position) was previously skipped with a warning. The executor
  now dresses it geometrically: the corner-tool cross-section shared with
  the headless round-trip (`blend_corner_profile` in `core/history.py` --
  one definition consumed by both the manifold verification gate and the
  FreeCAD executor, so they cannot drift) is sketched on the plane
  through the detected sharp edge, perpendicular to the edge direction,
  and extruded along the detected span. Convex blends become a Pocket
  (corner sliver removed), concave blends a Pad (quarter round fused,
  legs buried 0.02·radius into the material per the lateral-pad fusion
  doctrine; the arc itself exact). Direction is encoded in the sketch
  placement (mirrored profile), never a `Reversed` boolean. The op is
  labelled "(geometric)" and the Report view records the fallback
  loudly. The terminal deviation-correction pass reconciles any residual
  against the source mesh. Chamfers keep the loud skip (their tool side
  needs the headless solid probe).
- **Corner-tool cross-section builders** `blend_corner_profile` /
  `chamfer_corner_profile` in `core/history.py`, replacing the
  hand-built shapely polygons inside `solidify._apply_fillets`.
  All loops are wound counter-clockwise: `trimesh`'s polygon extrusion
  yields a non-watertight mesh for clockwise input, which the volume
  gate then rejects wholesale.
- **Documentation of the freeform-geometry boundary** (README
  limitations, design note 50): freeform transition bands and scoops fit
  no analytic primitive by definition; the parametric tree approximates
  them, the correction patches carry the mesh-accurate shape, and blends
  detected on such surfaces are dressed geometrically at their detected
  positions.

### Fixed

- **Geometric fallback sketch-plane handedness.** `App.Placement`
  silently negates a left-handed rotation matrix (R -> R*(-I)) instead of
  failing, so a fillet whose detected normals happened to order as
  `n_a x n_b = -direction` (the `_fillet_op` neighbour order is
  arbitrary) would have been dressed on the mirrored plane: the convex
  pocket cut air and the concave pad fused its quarter round off the
  part, both without error. The executor now swaps `n_a`/`n_b` when the
  frame is left-handed (the corner profiles are symmetric under the
  swap, so the dressed region is unchanged) and orthonormalizes `n_a`
  before projecting `n_b`, matching the headless `_edge_cutter`.

### Tests

- `test_fillet_ops.py::TestCornerProfiles` -- closed-loop, extent, area,
  and arc-radius checks for both builders (convex sliver, concave
  quarter disk, burial legs, straight-leg chamfer).
- `test_solidify.py::TestCornerToolVolumes` -- analytic volume gates for
  a single blend applied headlessly to a plain box (convex fillet,
  concave fillet, convex chamfer), pinning the refactored
  `_apply_fillets`.
- `test_blend_fallback.py` -- stubbed-FreeCAD executor wiring: unmatched
  convex fillet -> Pocket with the detected span and the
  `(n_a, -n_b, -d)` placement; unmatched concave fillet -> Pad with
  `(n_a, n_b, d)` and buried legs; matched edge -> primary
  `PartDesign::Fillet` path wins; unmatched chamfer and degenerate
  blends still report loudly without creating ops; two-fillet chain
  continues after fallbacks; both neighbour orderings of the same
  physical fillet yield the identical right-handed plane.

## 0.17.3 — 2026-08-22

Reconstruction-robustness release, prompted by a field report of a
3D-printed corner bracket that failed outright (#5, thanks @amoose136).
Includes the follow-up executor fixes that bring the bracket's editable
feature body in line with the mesh-accurate corrected shape (verified
against the mesh with headless FreeCAD builds).

### Added

- **Gusset web recovery.** Thin triangular reinforcing webs inside a recess
  (pointed at one end, tall at the other -- the corner gussets of a
  bracket) were dropped entirely by the prismatic rebuild: they are not
  holes, pads, or terraces, so no planner stage claimed them and the
  rebuilt body was missing ~5% of the part. `_plan_gussets` now takes the
  headless rebuild, isolates the connected components of
  ``mesh \\ rebuild`` that sit on a recess floor, and re-emits each as a
  LATERAL pad whose profile is the web's right-triangular side and whose
  extrusion thickness is chosen so the prism volume equals the region's
  true volume (matching mass without trusting coarse-mesh loft sections).
  The pads are built AFTER the recess pockets so the pocket that carves
  the cavity does not immediately delete them, and their base edge is
  buried slightly into the floor to give OCC a real fusion volume. On
  issue #5's bracket this recovers both corner gussets and moves the
  parametric rebuild from -5% (webs missing) to within ~1-2% of the mesh.
- **Bottom-face funnel pockets.** Mounting funnels that open on the BOTTOM
  face and run up to the recess floor (an INTERMEDIATE plane, not the far
  z-end) were invisible to the base through-opening matcher, which only
  compares opposite extremes and demands near-equal area -- the funnel
  tapers, and its far side is intermediate. `_plan_bottom_pockets` now
  detects each bottom-face inner loop that centroid-matches a recess-floor
  loop and emits a `from_bottom` pocket spanning the base-plate thickness,
  skipping loops already cut by a drilled hole/counterbore/cone feature.
  On issue #5's bracket this recovers the three mounting funnels.
- **Constrained arc/circle detection in `loop_to_sketch`.** New
  `max_sweep_deg` / `full_circle_deg` gates: runs whose fitted arc sweeps
  between the two thresholds are spurious (a curve bowing across a diagonal
  corner over unrelated boss outlines) and fall back to lines; tight arcs
  (<= max_sweep, e.g. fillets) and near-full circles (>= full_circle, e.g.
  boss outlines) are kept. The recess-mouth profiles use this so a 147-deg
  impostor arc no longer rounds a 90-deg corner.
- **Deviation correction (patch-boolean fallback).** New core module
  `solidify`: `plan_to_mesh` executes a plan headlessly with mesh
  booleans (base, pads incl. lateral, pockets, holes with counterbore/
  countersink, conical pockets, cross-axis holes, fillets, chamfers),
  `plan_corrections` reports the connected components of rebuild-vs-mesh
  (OVER = what the feature tree over-built, UNDER = what it missed), and
  `apply_corrections` closes the gap by intersecting with the source mesh
  and fusing the UNDER patches back. The FreeCAD executor emits every
  patch as its own tree object (a "<name> correction patches" group of
  labelled patch solids) and builds the corrected shape as a visible
  boolean chain (one Cut/Fuse per patch) validated against the headless
  reference -- like the fitted-surface features of commercial hybrid
  tools, every correction has a tree equivalent; where OCC's booleans
  degenerate on the faceted input the chain is dropped and the exact
  headless shape is installed instead, so the output is mesh-accurate
  either way: the body stays the editable parametric history, the
  corrected feature is the mesh-accurate output. This is the
  hybrid-modelling fallback for freeform geometry no analytic primitive
  captures: on issue #5's corner bracket (diagonal transition band, base
  scoops) it closes the rebuild from +25% to within ~0.1% of the mesh
  volume. Requires `manifold3d`; without it the plan is untouched and
  behaves exactly as before.
- **Blend-channel peeling (`split_by_channels`).** Refinement now peels
  straight-spine channels -- constant-cross-section blend bands such as a
  fillet strip along a straight edge -- out of a failed blob, after the
  planar peel and before the curvature split. Curvature clustering cannot
  separate adjacent strips of equal radius (their curvature proxy is
  identical); the invariant that does is the spine direction. Candidate
  spines come from adjacent-facet normal cross products, filtered by
  dihedral angle rather than cross-product norm so finely
  chord-tessellated holes (~0.1-1 deg per facet) still propose their
  axis; a candidate is accepted only if its faces' normals share one
  great circle, its on-surface samples project onto a single
  cross-section circle, and its ruling edges are mutually parallel. The
  last two gates reject the two impostor classes that pass a naive
  great-circle test: a narrow cone or sphere sector (rulings converge)
  and a flat slab wrap (cocircular corners). Full-turn channels are
  accepted -- a channel wrapping 360 deg around its spine IS a cylinder,
  and peeling one splits coaxial hole stacks (drill wall + cone +
  counterbore chained by the curvature bridge) that no other split
  separates. On the bracket this lifts recognized coverage 0.80 -> 0.85
  and turns rim blend bands into real fillet/chamfer operations instead
  of leaving them unrecognized (the rebuild kept every sharp corner).

### Fixed

- **"degenerate base extent" on parts whose flat faces are chained by
  gently faceted transition blends.** The bracket's base, recess floor,
  walls and top all fused into one ~3000-face segment whose curvature
  proxy is continuous (flat faces' per-edge curvature sits at the noise
  floor), so the refinement split could not break it and only a single
  plane of the dominant normal cluster was recognized -- a zero-thickness
  base the planner refuses. `reconstruct` now peels exactly-planar
  sub-regions out of a failed blob first (`split_by_planes`:
  vertex-coplanar region growing, each candidate validated by a global
  plane fit), recovering the parallel flats at their distinct offsets.
  Sub-significant slivers are deliberately left in the remainder so the
  peel yields a few coherent surfaces, not primitive confetti.
- **Compromise fits over near-flat blends.** A chamfer strip spanning the
  part can "fit" a cylinder/sphere whose radius dwarfs the part; the
  feature layer then read it as a giant fillet and the round-trip unioned
  part-sized junk (the bracket rebuilt at ~5000x its volume). `fit_best`
  now drops curved candidates whose radius exceeds 4x the segment
  diagonal, and a suspicious fit (vertex rms far above tolerance) that is
  itself a product of refinement splitting is reported honestly
  unrecognized instead of accepted.
- **Straight edges misread as a giant sketch arc.** A run of straight
  edges whose corners happen to be concyclic (the bracket's footprint
  diagonal is exactly cocircular with its two adjacent corners) passed the
  arc-sampling gate with uniform turns and bowed a huge arc off the true
  outline, inflating the extruded profile. `loop_to_sketch` now caps arc
  radius at half the loop diagonal and falls back to lines.
- **Executor deleted the terrace pocket when a bottom-side hole fell back
  to pocket cuts.** The hole fallback's cleanup used `'op' in dir()`,
  which saw the `op` variable leaked from the pocket loop and removed the
  last terrace pocket instead of the never-created `PartDesign::Hole`.
  Any part combining a recess with a hole that opens off the outer top
  face lost its recess cut (the bracket rebuilt ~3x its true volume). The
  cleanup now removes only a Hole object the same iteration created.
- **Coaxial hole stacks fused into one unrecognized blob on fine
  tessellations.** A drill wall, its taper and a counterbore chain into a
  single segment whose curvature proxy is continuous (the cone bridges the
  two cylinder radii), so neither the dihedral nor the curvature split
  separates it and the whole hole is dropped (field-observed with OCC
  0.05-chord tessellation, where the facet step is a fraction of a
  degree). The channel peel now splits the stack into its cylinders and
  the remaining cone, restoring counterdrill/counterbore detection.
- **Partial cylinders without two blend faces claimed as fillets.** A
  fillet is relational -- a blend between two faces. A compromise cylinder
  fitted over a curved transition band has no two planar neighbours
  perpendicular to its axis, yet the feature layer claimed it as a fillet
  and the planner reported it unplanned. The fillet detector now requires
  the two blend planes before claiming, so such strips stay recognized
  surfaces without spawning an unplannable feature.
- **Bottom-face countersunk hole now actually cuts.** A from-bottom THROUGH
  bore opening on an internal face (the recess floor) was sketched on the
  bottom face with `flip=False`, so the fallback Pocket extruded DOWN into
  air and removed nothing -- the rebuilt body stayed solid where the hole
  belongs. The drill now uses `flip=True` (sketch normal pointing down),
  which a PartDesign Pocket cuts opposite, i.e. UP into the part (the same
  convention the working from-bottom funnel pockets already use). The
  countersink cone for a from-bottom hole was also placed ABOVE the opening
  face, in the already-empty recess, cutting nothing; it now sits BELOW the
  mouth -- widening at the opening face and narrowing toward the drill --
  mirroring the top-side convention.
- **Fillet edge matching robust to rebuild approximation.** All 7 detected
  fillets were skipped ("no body edge matched") for two reasons: the matcher
  required the body edge's endpoints to fall within the detected segment,
  rejecting rebuilt edges LONGER than the one blend cylinder's span, and the
  executor tolerance (0.09 mm) was tighter than the coarse-mesh fit error
  that offsets the reconstructed sharp edge from the body's true edge.
  `fillet_edge_matches` now accepts any collinear edge overlapping the
  segment (sub- and super-edges), and the executor tolerance scales with the
  blend size. On the bracket this applies the 3 rim fillets whose edges the
  rebuild reproduces; the 4 fillets on the freeform diagonal band remain
  skipped because the feature tree has no sharp edge there to dress (that
  geometry is only recovered by the correction pass).

### Removed

- Leftover debug prints and a redundant duplicate hole-cutting loop in
  `solidify.plan_to_mesh`.

### Tests

- Vendored the reported bracket (`tests/fixtures/`, see the test module
  for provenance) and added `test_issue5_corner_bracket.py`: the plan
  succeeds with the true 20 mm extent, the flats are recovered at several
  distinct offsets, the countersink plus four cross-axis holes are
  detected, and at least one blend feature reaches the build plan.
- Added `test_blend_channels.py`: the peel recovers a fillet strip from
  curved junk end to end, and refuses a cone wall and a planar slab wrap.
- Added `test_solidify.py`: frustum cutters are watertight with exact
  volumes, a fully captured part needs ~no patches, and the bracket's
  corrections close the volume gap to within 2%. Suite grows from 452 to
  465 passing.
- The FreeCAD smoke test gains a terrace-plus-bottom-hole section pinning
  the executor fix (the terrace pocket must survive the bottom-side
  hole's pocket fallback, and the volume must match), and a bracket
  deviation-correction section (corrected shape exists, volume within 3%
  of the mesh).

## 0.17.2 — 2026-08-16

Dependency documentation and robustness release, prompted by a field
report from a Flatpak user (#3, thanks @kizzard).

### Fixed

- The missing-dependency gate now checks `shapely` (a hard planning
  dependency): with it absent, the rebuild command previously started
  and failed mid-pipeline instead of naming the missing package up
  front. The gate's message now includes the Flatpak install variant.
- The lateral-pad veto's near-surface leniency no longer aborts the
  rebuild on trimesh builds whose proximity queries need the optional
  `rtree` package; without it the veto degrades to its strict verdict
  (the 2.5-tol erosion already absorbs boundary noise).

### Docs

- README and VERIFY.md document the complete runtime dependency set
  (now including `rtree`), the test-only extras (`manifold3d`,
  `mapbox-earcut`, `pytest`), and Flatpak-specific pip/symlink/pytest/
  smoke-test commands. VERIFY.md previously listed only
  `numpy scipy trimesh`.
- `requirements.txt` gains `rtree`.

## 0.17.1 — 2026-08-05

Repository restructure requested by the FreeCAD addon-index review
(FreeCAD/Addons#108): nothing but metadata files may live at the addon's
top level, because FreeCAD puts the addon directory itself on
`sys.path` — a top-level `tests/` (or `scripts/`, or the core package)
becomes globally importable and collides with every other addon.

### Changed

- The geometry core moved from top-level `meshtofeatures/` to
  `freecad/meshtofeatures_wb/core/` (the `lib/`-subpackage pattern used
  by CurvesWB). Its import path is now
  `freecad.meshtofeatures_wb.core.*`; the modules themselves are
  unchanged (the core already used only relative imports internally).
- `tests/`, `scripts/`, and `docs/` moved under
  `freecad/meshtofeatures_wb/` as well; run the suite with
  `python -m pytest freecad/meshtofeatures_wb/tests/ -q` (or plain
  `pytest`, via `testpaths`).
- `package.xml`: `freecadmin` raised to 1.1.0 and the readme URL now
  serves the raw file (merged upstream from the addon review).
- The headless scripts re-extend the `freecad` namespace after putting
  the checkout on `sys.path`, since `freecadcmd` imports FreeCAD's own
  `freecad` package before the script runs.

## 0.17.0 — 2026-08-02

Four new rebuilt feature types, plus a cone-fitting robustness fix that
made them possible. This release also retires the *beta* label: the
planner regressions found by the 25-part real-STL stress campaign are
now fixed (spurious lateral pads, this release) or pinned as documented
limitations with loud reporting.

### Added

- **Conical pockets.** A concave cone that no drill or bore claims — a
  tapered recess, a conical seat, a tapered through hole — is now
  rebuilt as a placed `PartDesign::SubtractiveCone`. Handles truncated
  and pointed recesses, through tapers, off-centre and rotated parts,
  and top- and bottom-face machining. Previously the cone was fitted and
  dropped, and the terrace pass substituted a straight-walled pocket
  built from the floor loop — the right depth at the wrong radius, with
  nothing reported as unplanned.

- **Counterdrilled holes.** A hole carrying *both* a counterbore and a
  countersink — a cylindrical recess with a conical transition down to
  the drill — is now recognized as one feature and rebuilt as a
  `PartDesign::Hole` with `HoleCutType = Counterdrill` (bore diameter,
  the depth of the cylindrical part, and the included angle). Handles
  through and blind holes, off-centre and rotated parts, grid patterns,
  and top- and bottom-face machining. Previously the bore was dropped by
  the countersink pass and re-emitted as a spurious blind hole of the
  bore diameter, with the countersink's sketch plane placed on the bore
  floor instead of the part face.
- **Countersunk holes.** A concave cone capping a coaxial drilled
  cylinder is recognized as a countersink and rebuilt as a
  `PartDesign::Hole` with `HoleCutType = Countersink` (mouth diameter +
  included angle). Handles through and blind holes, arbitrary included
  angles, off-centre and rotated parts, grid patterns, top- and
  bottom-face machining, and coexistence with plain and counterbored
  holes. Previously the conical entry of every flat-head-screw hole was
  fitted but dropped as an unassigned surface.
- **Blind cross-axis holes.** A side hole that stops inside the part is
  now rebuilt as a depth-limited `CrossHoleOp` (entry point on the wall,
  inward direction, drilled depth), cut as a one-sided `Length` pocket.
  Previously only *through* cross-axis holes were rebuilt; blind ones
  were reported as unplanned.

### Fixed

- **Spurious lateral pads on non-flange geometry** (stress-campaign
  bucket 2). The lateral-pad hull assumes a prismatic convex protrusion;
  a protruding face that surrounds a through-window (angle_block.STL's
  leg), stacked protrusions sharing one outward direction, or the gap
  between disjoint bodies were silently hulled into invented material.
  Every lateral pad is now verified against the source mesh after
  planning (hull interior must agree with the mesh once planned cuts are
  subtracted); unsupported pads are dropped with a loud `unplanned`
  report instead of silently overfilling the part. Genuine flanges --
  including ones whose hulls are legitimately emptied by sub-level
  pockets (featuretype.STL) and thin mesh-true edge lands
  (octagonal_pocket.stl) -- are untouched.
- **Conical entries on the pocket-fallback path.** A countersunk or
  counterdrilled hole that does not open on the outer top face (bottom
  face, or a bore under a raised deck) is rebuilt by pocket cuts rather
  than `PartDesign::Hole`. That path cut cylinders only, so the conical
  entry was silently dropped; it is now cut as a `SubtractiveCone`.
- **Cone fitting** could diverge or mis-select on short two-ring cone
  segments (tessellated countersinks): the half-angle parameter wandered
  along the periodic residual valley to a wrapped value the `(0, π/2)`
  guard then rejected, and the normal-based initialization assumed a
  convex cone, sending concave (hole) cones to a degenerate solution.
  The half-angle is now recovered from the converged geometry and the
  axis orientation from the point cloud, so both convex and concave
  cones fit robustly. Without this, countersink cones were mis-fitted as
  spheres (two rings lie on a common sphere).

### Docs

- README: countersinks and blind cross-axis holes added to the supported
  list; corrected the stale note that chamfers are detected-but-not-
  rebuilt (they are rebuilt as `PartDesign::Chamfer`).

### Tests

- Suite grows from 318 to 452 passing (one expected failure documents
  a terrace-model limitation), including countersink
  detection/planning/property-mapping/round-trip, blind cross-hole
  planning and round-trips, convex/concave cone recovery, and a
  network-guarded robustness pass over a real machined part
  (`featuretype.STL`), plus non-prismatic-protrusion veto fixtures
  (windowed flange, stacked rails) and network-guarded
  angle_block/octagonal_pocket false-positive pins.

## 0.16.0 (beta) — 2026-07-11

First public beta release.

### Highlights

- End-to-end pipeline: STL → surface recognition (planes, cylinders) →
  design-intent snapping with audit trail → feature detection → editable
  PartDesign Body.
- Features rebuilt: base solid from footprint, multi-depth terraces and
  stepped pockets (with islands and curved boundaries), through/blind
  drilled holes, hole grid patterns, counterbored holes (including
  below-top openings), cross-axis holes, vertical bosses, lateral pads
  (flanges/gussets/bevels with true-slope undersides), partial fillets.
- Robust PartDesign executor: per-feature failure recovery (Refine
  fallback, epsilon retries, end-of-chain deferral for order-dependent
  OCC boolean flakes) with loud reporting — a failed cut is never
  silently dropped.
- Snapping: canonical-axis direction unification, coaxial merging,
  equal-value equalization, grid rounding — every decision logged.
- Works on parts in arbitrary orientation (frame detection).
- Distinct toolbar icons per command; task panel with options/progress.
- 318-test pytest suite (runs without FreeCAD) plus a 25-part corpus
  regression gate; headless in-FreeCAD probe script for field
  diagnostics.

### Known limitations

See README "Limitations": prismatic parts only; chamfers detected but
not rebuilt; dimensional fidelity bounded by mesh tessellation (~0.1% of
part diagonal, with some recess boundaries intentionally oversized by
~0.15% to avoid degenerate OCC shells); scan-quality meshes untested;
FreeCAD 1.1+ target.
