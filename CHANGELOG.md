# Changelog

## Unreleased

Four new rebuilt feature types, plus a cone-fitting robustness fix that
made them possible.

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

- Suite grows from 318 to 379 (all passing), including countersink
  detection/planning/property-mapping/round-trip, blind cross-hole
  planning and round-trips, convex/concave cone recovery, and a
  network-guarded robustness pass over a real machined part
  (`featuretype.STL`).

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
