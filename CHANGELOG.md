# Changelog

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
