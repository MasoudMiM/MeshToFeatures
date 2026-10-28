# Roadmap

Working plan for MeshToFeatures after v0.17.7. Each phase is developed on
its own branch (see the branch name in each phase's header) and released
when its exit criteria are met. The order is deliberate: the process
guarantee first, then feature depth, then trust, then editability.

## Phase 1 — FreeCAD executor CI (`infra/freecad-executor-ci`)

The last five field fixes (issues #3–#8) were all executor regressions the
headless core suite cannot see. The core is in CI; the PartDesign executor
(`build.py`, ~1000 lines) is verified only by a manual smoke test. This
phase puts the executor in CI.

Scope:

- A second CI job (`freecad-smoke`) runs the existing
  `scripts/freecad_smoke_test.py` (17 parts, full pipeline through
  `build_body`, including the manifold3d deviation-correction case) inside
  a real headless FreeCAD 1.1.x build (AppImage, sha256-verified) on every
  push and PR.
- The smoke test's `check()` output is flushed per check: `freecadcmd`
  drops unflushed Python stdout on `sys.exit()`, which would otherwise
  hide all `[PASS]`/`[FAIL]` lines in CI logs.
- Dependency metadata brought in line with the runtime dependency gate
  (`commands._missing_dependencies`): `shapely` (and `rtree`) are runtime
  requirements, not dev-only; README / requirements.txt / pyproject.toml
  stay mutually consistent.

Exit criteria:

- `freecad-smoke` job green on the FreeCAD 1.1.x stable AppImage with all
  smoke checks visible in the log.
- A deliberate executor regression (e.g. a wrong volume tolerance) fails
  the job — verified once during development.
- Headless suite still green; dependency docs accurate.

Status: **in progress**.

## Phase 2 — Feature depth: convex cones, spheres, tori

The README's core limitation is that the core *fits* spheres, tori, and
convex cones but the planner/executor do not *rebuild* them. Each item
reuses the existing pattern (feature rule in `features.py` → plan op in
`history.py` → placed primitive or dressup in `build.py` → corpus + smoke
case), gated by the mesh-veto doctrine and the unplanned doctrine.

Order (smallest documented gap first):

1. **Convex cones** (tapered bosses, draft-angled studs) as
   `PartDesign::AdditiveCone` — the `SubtractiveCone` path (ConeOp)
   already exists end to end; the convex rule, plan op, and executor
   branch are the delta.
2. **Spheres** (spherical bosses and concave seats) — PartDesign has no
   sphere primitive; the executor builds a placed `Part::Sphere` revolve
   (circle sketch) fused/cut through the body, or declines loudly.
3. **Tori** (round blends on edges the fillet rule cannot claim) — the
   largest of the three; revolve-based, and only where a full-revolution
   torus fit is unambiguous.

Exit criteria: corpus parts carrying each new feature class with analytic
ground truth; smoke case per executor path; no unplanned/silent drops of
newly recognized features.

Status: not started.

## Phase 3 — Per-feature deviation report

The pipeline already computes the trust metrics internally (two-way p99
surface distance in the adversarial verification, the volume gate in
`_shape_from_mesh`, per-feature checks in `probe_bore_placement.py`) but
never shows them to the user. This phase surfaces them: a Report-view
table per rebuild listing, for every planned feature, its status
(rebuilt / deferred / skipped / undressable) and its measured deviation
from the source mesh, plus the overall volume ratio. The hybrid-modelling
story (edit the tree, *measure* the corrected feature) needs this to be
usable rather than asserted.

Exit criteria: the report renders for a corpus part and for the corner
bracket; numbers agree with the probe script's output; no new pipeline
math — plumbing only.

Status: not started.

## Phase 4 — Dimensioned, constrained sketches

`loop_to_sketch` currently emits geometry only. PartDesign::Hole carries
full parameters (resize-a-hole works), but pocket/pad/step *profiles* are
unconstrained: dragging an entity has no dimensions to preserve, which
undermines the "editable body" promise. This phase makes rebuilt profile
sketches fully constrained with named, snap-informed dimensions.

Exit criteria: corpus parts' rebuilt sketches report fully-constrained
status; a scripted dimension edit (e.g. deepen a pocket) survives the
solve; under/over-constrained sketches are reported, never silently
emitted.

Status: not started.

## Quick wins (any time, small branches)

- **One-click STEP export** of the rebuilt body (the common "I need the
  STEP" end of the workflow).
- **Adoption content**: a short demo video, a before/after gallery, and a
  FreeCAD forum thread. The addon pulls ~500 installs/month from the
  official index; discoverability is the current constraint.

## Deferred (with reasons)

- **Noisy 3D-scan meshes.** Explicitly out of scope for the prismatic
  target; the conditioning and snapping tolerances assume
  CAD-exported tessellations. Revisit only after the prismatic core is
  rock-solid and real scan feedback exists.
- **Multi-axis / revolve / swept parts.** An architectural leap beyond
  the single-frame prismatic model; the terrace/step machinery is
  frame-axis-specific.
- **Interactive recognition override.** Valuable for robustness, but a
  large UX investment; the automatic-pipeline-plus-report identity is the
  differentiator from the manual/assisted tools (e.g. Detessellate).

## Process

- One branch per phase/slice; branch names prefixed `infra/`, `feature/`,
  or `fix/` per the existing convention.
- Every new executor path lands with a smoke-test case in the same change
  (Phase 1 makes that check run automatically).
- Every new feature class lands with a corpus part and analytic ground
  truth in the same change.
- `pyproject.toml`, `requirements.txt`, `package.xml`, `README.md`, and
  this file are checked for staleness in every phase's final pass.
