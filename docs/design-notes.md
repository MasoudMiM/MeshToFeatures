# meshtofeatures (working title)

Geometry core for a FreeCAD reverse-engineering addon: converts triangle
meshes (STL and friends) into recognized analytic surfaces — the first
stage toward editable parametric models.

**License:** LGPL-2.1-or-later (add canonical license text from
https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html before publishing).

## Architecture

- `meshtofeatures.primitives` — Plane / Sphere / Cylinder / Cone with
  point-to-surface distance functions (the single interface that drives
  fitting, scoring and testing).
- `meshtofeatures.fitting` — closed-form initializers + nonlinear geometric
  refinement per primitive; `fit_best` does tolerance-gated model
  selection preferring simpler primitives.
- `meshtofeatures.segmentation` — region growing on the face-adjacency graph
  with a dihedral-angle smoothness criterion; produces segments with
  paired vertex positions / segment-restricted normals and dense
  face-interior sample points.
- `meshtofeatures.pipeline` — `reconstruct(mesh)` end to end; returns
  recognized surfaces, unrecognized segments, and area coverage.

The core is deliberately FreeCAD-free (numpy/scipy/trimesh only) so it
runs headlessly in CI. The FreeCAD workbench will be a thin adapter that
maps `ReconstructionReport` into document objects.

## Design notes (hard-won, do not regress)

1. **Fit on vertices, score on face samples.** Mesh vertices can occupy
   degenerate positions consistent with the *wrong* primitive: a cylinder
   barrel's vertices form two rings that lie exactly on a sphere. Dense
   face-interior samples expose the impostor. See
   `tests/test_pipeline.py::TestCleanSolids::test_cylinder`.
2. **Segment-restricted vertex normals.** Ordinary vertex normals are
   blended across sharp edges; normals averaged only within a segment keep
   rim vertices usable for axis estimation.
3. **Adaptive Occam gate.** Model selection accepts the simplest
   primitive within `max(tolerance, 1.2 x best_rms)` — absolute tolerance
   alone breaks under noise / tessellation chord error.

- `meshtofeatures.snapping` — design-intent recovery: direction
  unification (with canonical-axis snapping), coaxiality, radius
  equalization, and grid snapping of scalars, each logged as a
  `SnapAction` audit trail. `snap_report(reconstruct(mesh))` is the full
  v0.2 pipeline.

Additional design notes:

4. **Informative snapping only.** A grid is considered only when its
   spacing is >= 4x the tolerance, so a random value snaps with <= 50%
   probability — every accepted snap carries at least one bit of evidence
   of design intent. Without this rule, fine grids snap everything.
5. **The mesh gets the veto.** Every snapped primitive is re-scored
   against its segment's surface samples; if the RMS degrades beyond
   `max_extra_rms`, the snap is reverted and the rejection logged. Snaps
   are hypotheses about intent; the data can refuse them.

- `meshtofeatures.emission` — pure planner turning recognized primitives into
  bounded patch specs (frames, parameter ranges, hull polygons).
- `freecad/meshtofeatures_wb/` — the FreeCAD workbench: two commands
  (reconstruct snapped / raw), color-coded Part faces, audit trail stored
  on the result group. Kept deliberately thin; its math (placements, OCC
  cone slant parametrization) is covered by stub-based tests, and real
  FreeCAD behaviour by `scripts/freecad_smoke_test.py` -- see VERIFY.md.

6. **Planes are bounded by real boundary loops.** A segment's boundary
   (directed edges used by exactly one segment face) chains into an outer
   outline plus one loop per hole; convex hulls are only a fallback.
   Without this, plane patches paper over drilled holes and concave
   outlines (observed in the field on 1002_tray_bottom.STL, v0.3).

Robustness design notes (v0.5):

7. **Condition first.** STLs arrive as unwelded triangle soup; meshes are
   welded and degenerate faces dropped before anything else
   (`meshtofeatures.conditioning`).
8. **The angle threshold adapts.** The dihedral distribution's largest
   gap separates tessellation angles from feature edges; a fixed
   threshold cannot serve both 8-section and 64-section exports.
9. **Curvature proxy uses edge-perpendicular span.** Dividing dihedral by
   centroid *distance* dilutes curvature by 10x on anisotropic wall
   quads, hiding fillets.
10. **Tangent blends are found by a second pass.** Segments that fail
    fitting are re-split by curvature contrast (plane k~0 vs fillet
    k~1/r) -- the only signal where no sharp edge exists. Splits that
    shatter (median child < 2 faces) are refused: that is noise, not
    structure.
11. **Acceptance is judged on vertices; selection on samples.** A correct
    primitive interpolates any tessellation's vertices (coarse meshes
    stay recognizable); a noisy mislabel has vertex rms at the noise
    level. Sample scoring still picks *which* model.
12. **Interpolation is not evidence.** Planes need dof+1 points; curved
    primitives need 2*dof. A triangle is not a recognized plane, and 7
    noisy points are not a cylinder.

- `meshtofeatures.features` — rule-based feature inference: holes (through /
  blind), counterbores, bosses, fillets, prismatic pockets, each with
  parameters, a human-readable description, and provenance (the surface
  indices it explains). Surfaces are consumed at most once; anything no
  rule explains stays honestly unassigned. Design note 13: **opening
  detection uses interior loops only** -- a cap or shoulder whose outer
  rim lies on a cylinder is that cylinder's end disk, not a surrounding
  opening; conflating them turns base bodies into phantom bosses.

- `meshtofeatures.patterns` — groups same-spec features into circular (bolt
  circles), linear, and grid patterns with pitch/BCD parameters. A spec
  group must match wholly; partial matches stay ungrouped rather than
  half-guessed (subset mining and relative-tolerance spec keys are known
  next steps). Collinear positions fitting a huge circle are guarded by
  requiring the circle radius to be comparable to the position spread.

- `meshtofeatures.history` + `freecad/meshtofeatures_wb/build.py` — build-plan
  inference and PartDesign emission: base pad from the largest face's
  real boundary loop (decomposed into editable lines/arcs/circles),
  then pads, pockets, and hole ops (patterns as multi-position ops,
  counterbores as stacked cuts). Plan semantics are pinned by executing
  every fixture plan with booleans and matching the mesh volume within
  1%. Design notes 14-15: an arc claim requires arc-like sampling
  (>= 2 same-sign turns, each <= 60 deg) AND constant curvature
  (turn ratio <= 1.5) -- rectangle corners are cocircular, and a straight
  side plus a tangent transition sits on a huge circle; both are
  impostors that residuals alone cannot reject.

Adversarial hardening (v0.8.1):

16. **Cuts are side-aware.** Volume round-trips cannot see "right material
    removed from the wrong face"; verification is a two-way p99 surface
    distance in world coordinates, and blind holes / counterbores carry
    `from_top` computed from where the cut's cylinder hugs the base extent
    (the plan frame's z sign is arbitrary).
17. **Ownership doctrine, amended.** A feature's defining cylinder is
    exclusive; planes are shared interfaces. A top annulus is
    simultaneously a through-hole's opening and a boss's cap -- treating
    planes as consumable made hollow bosses silently vanish.

v0.9 (slots + tolerant spec keys):

18. **Sketch decomposition is start-invariant.** A loop's arbitrary start
    vertex can split one entity in two (mid-arc start -> big arc plus an
    orphaned sliver); decompose, rotate the loop to the first detected
    primitive boundary, decompose again.
19. **Slots are stadium openings** (2 parallel lines + 2 semicircular
    arcs) with concave half-cylinder end walls, claimed before the fillet
    rule so slot ends are not mislabelled as blends. Open-ended slots
    (running off the part edge) are a known unhandled variant.
20. **Spec matching is relative.** Fixed decimal rounding split two holes
    differing by 1e-6 relative (the same drill, observed on real STL
    data); scalars now agree within 2e-3 relative.

v0.10 (UX): the pipeline reports progress via an optional
``progress(stage, fraction)`` callback (fractions forced monotonic --
the refinement queue grows while draining). In the GUI, commands run the
pure pipeline on a worker thread with a progress dialog, and a task
panel (options + progress + browsable results) is available; document
objects are only ever touched from the main thread via a poll timer.

v0.11 (semantic holes): rebuilt bodies use PartDesign::Hole features
(one editable feature carrying diameter, depth mode, counterbore) via a
pure, tested HoleOp -> properties mapping, with pocket-cut fallback.
`meshtofeatures.standards` identifies ISO metric clearance and tap-drill
sizes (d6.6 -> "M6 clearance") and annotates hole features; only
unambiguous matches are reported. Known future step: standards-aware
snapping (a d6.6 drill can currently be grid-snapped to 6.5 and miss the
window).

v0.12 (horizontal fillets): non-vertical fillets become FilletOps: the
sharp edge each fillet replaces is its axis line shifted by
``+- r * (n_a + n_b)`` (snapped, sign-oriented blend-plane normals; raw
face-normal means are tilted by boolean slivers and shift the edge by
r*sin(tilt) -- design note 21). The executor matches straight body edges
against the op with a pure, tested predicate (sub-edges accepted, since
boolean rebuilds split edges) and applies PartDesign::Fillet, chained,
with compute-failure rollback. Round-trip verification applies convex
fillets with corner-tool booleans, so the whole plan is geometry-checked
headlessly.

v0.12.1 (rigor sweep fixes -- found by random transform x scale sweeps
with two-way surface-distance verification):

22. **Position snapping requires a canonical direction.** World-grid
    coordinates only encode intent for world-aligned entities; snapping a
    rotated hole's axis position merely perturbs correct geometry within
    the guard band (and pushed a hole's loop off its cylinder, silently
    killing feature detection).
23. **Pads are side-aware** (`from_top`), like holes and pockets: the
    plan frame's z sign is arbitrary, and a boss can hug either face.
24. **Loop areas via ||sum of cross products||/2** -- summing absolute
    components is orientation-dependent by up to sqrt(3).

v0.13 (polish batch):

25. **fit_best short-circuits by complexity** and refines on a
    deterministic subsample (scoring stays full): profiling showed cone
    fitting was ~75% of pipeline time, mostly on segments that are
    trivially planes. 2-25x faster on real parts, behaviour unchanged.
26. **Standard hole sizes are snap authorities** -- a d6.6 drill means M6
    clearance even though 3.3 sits on no informative grid; the closer of
    grid/standard wins and the guard still vets it.
27. **Suspicious accepts get the curvature split first.** A passing fit
    whose vertex rms is >> fit_tolerance on a clean mesh smells like a
    compromise over tangent-merged surfaces (pocket floor + concave
    fillets as one shallow cylinder); noisy meshes' splits shatter and
    are refused, falling back to acceptance.
28. **Open-ended slots** are semicircular notches in OUTER boundaries
    (arc flanked by parallel lines + a concave end wall); planned as
    pockets whose mouth segment is pushed past the part edge.
29. Concave fillets are round-trip verified (union corner tools).

v0.13.1: renamed **MeshToFeatures** (was the working title "mesh2part",
retired after a collision review: Mesh2Surface is an established
commercial product in the same category). The name states the
differentiator: everyone else converts meshes to dumb solids; this
recovers the *features*.

v0.14 (completeness before release):

30. **Chamfers.** A chamfer is a narrow plane strip bisecting two
    ~perpendicular neighbours; the NARROWNESS rule is essential (a top
    face flanked by two chamfer strips bisects them identically). The
    sharp edge is the neighbours' intersection line; PartDesign::Chamfer
    via the same edge matcher as fillets.
31. **Flat blobs split by face normals.** 45-deg chamfer edges vs 45-deg
    coarse-tessellation facets are inherently ambiguous by dihedral
    angle; disambiguation lives in the failure-path splitter (a coarse
    cylinder FITS and never reaches it), whose classification curvature
    uses only tessellation-scale (< 20 deg) edges so slipped feature
    edges cannot spawn ribbon children.
32. **Cross-axis through holes** are rebuilt as midplane through-all
    pockets on axis-normal sketches (CrossHoleOp, world-frame). Blind
    side holes remain honestly unplanned.

v0.15 (steps / multi-level tops):

33. **Exposed intermediate planes touching the outline are steps.**
    Exposure implies an empty column to the corresponding face, so a
    pocket over the shelf's own outer loop (outline-coincident vertices
    pushed outward; the true step wall exact) cut to that face is valid
    by construction -- and orientation-symmetric, since the plan frame's
    z sign is arbitrary. Interior loops (pocket floors, counterbore
    shoulders) never touch the outline: no phantom steps.

v0.15.1 (field fixes from the step rollout):

34. **Sketch arcs are built through three points** (start, on-arc mid,
    end): fitted-arc endpoints sit up to ~1e-4 off the shared raw chain
    points, and angle-parametrized arcs land ON the circle, opening the
    wire. Step profiles additionally use pure polylines (arcs=False) --
    guaranteed closure beats micro-arcs fitted to noisy shelf boundaries.
35. **One broken feature degrades the body by one feature.** Every
    chained op is validated and rolled back on failure; previously a
    single open-wire step pocket invalidated every downstream feature
    and blanked the part.

v0.15.x (step hardening, field-driven): pocket floors must lie BELOW
their opening (phantom inverted pockets razed raised decks); ring
shelves carry raised-material hole loops as sketch holes, but ONLY
raised ones -- drill/pocket openings piercing a shelf must be cut
through, or annular chimneys stand at every drill; from-bottom
operations encode direction via flipped sketch placements (mirrored
profiles), not direction booleans.

v0.15.7 (robustness, stress-test driven): a 25-STL stress campaign
(clean prismatic, multibody, degenerate, organic) drove a degenerate-
loop guard. Noisy or subdivided faces (subdivided cubes, scanned boxes)
segment into sliver planes whose boundary loop collapses to one or two
points; a pocket/step built from such a loop made a <3-point wire that
crashed the boolean rebuild ("linearring requires 4 coordinates") and
the FreeCAD sketch. `_valid_loop` (>= 3 distinct vertices AND non-
negligible area) now gates every op-emission site, so a sliver drops ONE
feature to `unplanned` instead of blanking the part. Organic/degenerate
inputs (spheres, tori, triangle soup, empty meshes) already decline
cleanly ("no planar surfaces / degenerate base extent"); very complex
faceted parts that over-segment (box.STL's 97 steps) still lean on the
executor's per-feature rollback rather than reconstructing well. A
second stress finding: the lateral-cylinder detector was reading a
horizontal drilled HOLE as a protruding boss when an incomplete base
outline made the hole look like it reached past the footprint
(angle_block). `_cylinder_convexity` now gates it -- a boss's vertex
normals point away from the axis (~+1), a hole's point inward (~-1), so
only clearly convex cylinders (> 0.3) become pads. The PLANAR lateral
false-positives on the same parts are not truly spurious: they capture
real material beyond a base outline that determination fit too small
(an L-block's rectangle, an octagon's sub-polygon), so the fix is better
base-outline determination, not suppressing the pad.

v0.15.6 (lateral pads):

36. **Material protruding sideways off a wall is a lateral pad** -- a
    mounting flange, side rail, or gusseted bracket. The base profile is
    one horizontal plane's outline, so a protrusion in a limited z-band
    (not touching the base's top or bottom face) had no vertical-
    extrusion representation and was silently dropped (7% of volume and
    an exact-`depth` p99 plateau on the field's featuretype.STL: a
    full-width, 45-degree-ramped bracket in the z-band [0.5, 1.0]).
    Detection anchors on a VERTICAL wall whose footprint sits beyond the
    base outline; the pad extrudes along that wall's length axis, and its
    profile is the CONVEX HULL of all protruding points in the plane
    perpendicular to that axis -- exact for both rectangular lugs and
    ramped/gusseted brackets. Face-wise point gathering (clipped to the
    beyond-region) recovers the inner edge that per-point selection drops
    on coarse meshes; end-cap walls (normal parallel to the pad axis) are
    excluded so their full-height points cannot corrupt the hull; inner
    vertices are buried into the base so the union fuses. The pockets
    carving sub-levels into the flange top are emitted separately and cut
    it afterwards. Basis and anchor are stored in the PLAN frame; the
    executor composes with the frame placement via
    `lateral_pad_world_frame` (pure, tested) and builds a
    `PartDesign::Pad` on a sketch normal to the pad axis -- direction
    encoded in the placement (min-axis anchor -> default +normal
    extrusion), no Reversed/Midplane. Verified by the adversarial two-way
    world-frame match (incl. a rotated part), the world-frame composition
    identity, and a stubbed-FreeCAD wiring test; real-FreeCAD geometry is
    the `RebuiltFlange` case in scripts/freecad_smoke_test.py.
    A lateral pad's basis (plane_u, plane_v, axis) is forced RIGHT-handed
    before emission: FreeCAD's App.Placement cannot represent a reflection,
    so a left-handed frame silently drops it and mis-places the pad -- the
    flange floats OFF the body (field-observed on featuretype in FreeCAD).
    The manifold round-trip applies the full 4x4 matrix and never saw it,
    which is why the container passed at 0.11%; fixing it means emitting a
    right-handed frame (flip v and mirror the profile's v-coords: identical
    geometry, valid placement). The pad also buries its inner edge into the
    base (overlap max(0.1*depth, 12*tol)) so OCC's boolean -- which, unlike
    the manifold union, will not fuse a thin sliver -- joins it cleanly.
    Locked by a right-handed-basis test, an overlap-fraction test, and a
    single-connected-body test, since the manifold harness alone sees
    neither App.Placement nor OCC's fuse.
    Multiple protrusions sharing an outward direction are separated by
    clustering their walls along the pad axis (two lugs on one side ->
    two pads, not one hull bridging the gap; `RebuiltTwoLug` smoke case).
    A protrusion with a CURVED outer wall -- a horizontal cylinder
    reaching past the outline -- is anchored on the cylinder instead of a
    planar wall and emitted as a circular pad extruded along its axis:
    this covers both a peg boss (axis perpendicular to the wall) and a
    rounded rail (axis parallel, the inner half fusing back into the
    base), distinguished from a horizontal through-hole by whether the
    cylinder body actually reaches beyond the outline (`RebuiltBoss`,
    `RebuiltRail` smoke cases). Non-cylindrical curved outer walls (e.g.
    a swept fillet profile) remain out of scope.

37. **A frame/ring's base face has a hole; the base profile must carry
    it** (v0.15.8). The base profile was the base face's OUTER loop only,
    so the base prism filled any central opening -- over-filling frames by
    exactly the opening's volume (field-observed on idler_riser.STL:
    rebuilt 2x too big; +103.6% -> +4.5% once fixed). Round and
    rectangular openings are already recovered as hole/pocket features and
    cut, but an odd polygon (an octagonal bore) is not, so `BasePad` now
    carries the base face's THROUGH-openings as `hole_profiles` (polyline
    inner wires) and both the rebuild and the FreeCAD base sketch punch
    them. "Through" is required -- an inner loop is only carried if the
    opposite z-end face has a matching loop (same 2D centroid and area),
    so a blind-pocket mouth (solid on the far side) is NOT mistaken for a
    hole and does not wrongly open the base. An opening already recovered
    as a drilled hole/counterbore FEATURE is also excluded (it is cut as
    that feature): otherwise the base pre-holes it, the hole/counterbore
    pockets cut already-open space, and the counterbore recesses are lost
    -- field-observed on featuretype (v0.15.14), where 8 counterbored
    through-holes had pre-punched the base so the top counterbore sinks
    vanished in FreeCAD; the manifold round-trip masked it because its
    counterbore boolean still carved the recess. Verified by a synthetic
    octagonal-frame fixture (+ rotated), a blind-pocket negative, and a
    counterbored-plate negative; real-FreeCAD geometry is the
    `RebuiltFrame` and `RebuiltCBoreBottom` smoke cases.

38. **Decline parts too complex for the prismatic model** (v0.15.9). A
    faceted, terraced, or organic export can have planar facets yet no
    sensible single-axis prismatic reconstruction -- the field's box.STL
    segments into ~400 planes across ~15 levels/direction and yields 97
    phantom "steps". When the step count exceeds `MAX_STEP_LEVELS` (32; no
    ordinary machined part has that many shelves) `plan_history` declines
    with a clear message, exactly like the "no planar surfaces" /
    "degenerate base extent" checks -- so RebuildBody reports "too complex"
    instead of a nonsense stepped body, while Reconstruct still shows the
    recognized surfaces. (box.STL is genuinely ~395 sharp-edge-separated
    regions, not a fixable over-segmentation: region-growing already
    merges its coplanar faces; only 80 of 4724 cross-segment edges are
    spurious coplanar splits.)

39. **Cap the refinement-split threshold** (v0.15.10). A large flat face
    can transition smoothly (chamfer/fillet) into a faceted or curved
    perimeter; region-growing at a loose feature-edge threshold (up to
    60 deg on a coarse export) fuses them into one blob that fits no
    primitive and is dropped -- the flat plane is lost (field: counter.-
    unitsmm at 49% coverage, one 4074-face blob holding a 66,000-area flat
    top). The refinement split (`split_by_curvature`) now runs at
    `min(angle_threshold, refine_split_max)` with `refine_split_max`
    defaulting to 30 deg, so a tangent-merged blob whose internal edges
    are all below the feature threshold still breaks and its flat core is
    recognized (counter 49% -> 100% coverage, p99 5.9 -> 2.1 within
    tolerance; zero corpus regressions). Parts whose adaptive threshold is
    already below the cap are unchanged; the split only runs on segments
    that already failed a fit, so clean prismatic parts never reach it.
    The cap is emergent-per-mesh and not reproducible by a small synthetic
    fixture, so it is locked by a mechanism test (the threshold handed to
    the splitter) plus corpus validation.

40. **Canonicalize the frame axis sign** (v0.15.15). `means[best]`
    inherits its sign from the mesh's face-normal winding, so the same
    part reconstructs with frame_z=+axis from trimesh but -axis from
    FreeCAD's tessellation. The plan is self-consistent either way, but
    the executor's from-top/from-bottom placement assumes a canonical
    orientation; under the inverted sign every from-bottom feature was
    mis-placed (the common root behind a run of executor symptoms). frame_z
    is now pointed along the POSITIVE direction of its dominant axis --
    stable and mesher-independent. Verified by a monkeypatched-inverted-
    mean test (fails without the fix) and a no-op check on canonical
    fixtures; the executor half needs a FreeCAD run to confirm.

41. **Height-field terrace reconstruction** (v0.15.16). The step planner
    (from-outline shelves) and the opening-driven prismatic-pocket
    reconstruction are replaced, for horizontal faces, by a single terrace
    pass: each intermediate horizontal plane is cut to its OWN exact
    projected footprint (its mesh triangles unioned in the frame plane,
    exterior plus interior tower holes), so interlocked, nested, and
    thin-wall-edge recesses reconstruct without the over/under-cut a
    from-outline step or an opening loop produced. The mesh is carried on
    `ReconstructionReport.mesh` so planning can project exact footprints.
    Three field-derived guards keep it safe: an EXPOSURE check (skip planes
    under an overhang/lip — only cut where the mesh is air on the exposure
    side), a minimal fixed OUTLINE PUSH (~2·tol; a coincident cut face
    leaves zero-volume sliver walls, but a depth-scaled push distorts
    complex edge recesses), and a FACET DECLINE on the count of
    intermediate horizontal member segments (a tessellated export rebuilds
    into disconnected garbage). Corpus-gated to zero shape regressions:
    1002_tray_bottom 50%→0.02% vol, octagonal_pocket 39.5%→0.62%, and
    featuretype's terraced recesses 3.47%→0.17% max deviation (its 0.04%
    baseline volume was a FALSE match — compensating errors masking the
    missing recesses; judge surface + reverse deviation, never volume
    alone). Documented residuals: filleted/chamfered-floor pockets leave
    the thin transition ring uncut (xfail — reaching the opening instead
    double-cuts counterbore shoulders), and idler_riser's honest volume
    error rises while its shape is unchanged (a correct cut exposing a
    pre-existing compensating error).

42. **Terrace bores must not chimney** (v0.15.17). When the tallest level
    is a raised deck, the surrounding shelf is a terrace cut down from the
    deck; a bore through that shelf, if kept as a terrace sketch hole,
    leaves a standing column that the bore feature turns into an annular
    chimney wall rising to the deck (field-reported). A terrace interior
    loop is now kept ONLY if it bounds RAISED material (tower/boss/frame
    wall), not a bore descending through it (`_raised_loop`). The check
    probes just INSIDE the loop boundary and ray-casts toward the terrace
    (deterministic) so a thin-walled or hollow frame — air at its centre —
    still reads as raised. This is the correct form of the mean-inside
    `mesh.contains` filter that razed idler_riser's hollow frame; here the
    frame is kept while counterbore/drill openings drop. Regression test
    proven to fail without the fix.

43. **Counterbores on a terrace open below the top** (v0.15.18). When the
    tallest level is a raised deck, the base plate around it is a terrace,
    and bores on that plate open BELOW the global top. Placing their sketch
    at the global top makes PartDesign::Hole extrude the counterbore OUTWARD
    as a tower (field-reported risen walls on featuretype). Holes now carry
    `surface_z` (the frame-z of the opening face, from provenance via
    `_surface_z`), and the executor places the drill and counterbore sketch
    there so the Hole cuts inward. This is EXECUTOR-ONLY: the manifold
    container rebuild was already clean (it makes the recesses via terraces
    and has no outward-tower bug), so the gate never saw the walls -- a
    reminder that forward max-dev misses risen material; use two-way p99 or
    containment scans. The container proxy keeps its counterbore at the top
    (cutting it at surface_z double-cuts the shoulder terrace and trips a
    manifold coincident-face artifact); only FreeCAD confirms the fix.

43. **Terrace bores cut from the top through their floor** (v0.15.20). When
    the tallest level is a raised deck, the base plate around it is a terrace
    and its counterbored bores open BELOW the global top. Two earlier
    symptoms (PartDesign::Hole extruding towers; then uncut base material
    standing as columns) share a root: OCC will not reliably clear the space
    above a below-top bore via the terrace's complex face. The robust fix is
    to not depend on that face — a from-top bore that opens below the top is
    drilled from above, so the space above its opening face must be open, and
    the bore is cut from the GLOBAL TOP straight down through its floor (drill
    ThroughAll from L; counterbore length extended by `L - surface_z`, from
    the `surface_z` provenance field). This removes any standing column AND
    forms the recess in one Pocket, independent of the terrace, with zero
    over-cut (verified against the mesh). Top-face bores are unaffected.
    EXECUTOR-ONLY and invisible to the manifold gate (manifold cuts the
    terrace correctly, OCC does not); confirmed by simulating the exact cuts
    in-container (0/8 columns) and by a FreeCAD run (probe
    `scripts/probe_bore_placement.py`, which builds the body and checks each
    bore site).

## Run tests

```
pip install -e .[dev]
pytest
```
