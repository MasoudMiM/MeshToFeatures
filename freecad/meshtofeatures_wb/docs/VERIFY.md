# Verifying the FreeCAD adapter on your machine

The geometry core and the emission planner are fully covered by the pytest
suite (runs anywhere). CI additionally runs the headless smoke test of
section 3 inside a FreeCAD 1.1.x AppImage on every push and PR, so the
AppImage baseline is regression-gated automatically. Follow this
checklist for YOUR FreeCAD build (another version, a Flatpak/snap/AppImage
install, or a new feature branch) -- once per FreeCAD version.

## 1. Install Python dependencies into FreeCAD's interpreter

Find FreeCAD's Python (GUI: Python console -> `import sys; sys.executable`),
then install the runtime set (rtree is a spatial index that some trimesh
versions require):

    <freecad-python> -m pip install numpy scipy trimesh shapely rtree

To also run the pytest suite inside that interpreter, add the
test-only packages (boolean fixture construction + triangulation):

    <freecad-python> -m pip install manifold3d mapbox-earcut pytest

(AppImage/snap users: use the bundled `pip` module the same way.)

Flatpak installs are sandboxed -- run pip through the flatpak instead
(thanks @kizzard, #3):

    flatpak run --command=python3 org.freecad.FreeCAD -m pip install --user numpy scipy trimesh shapely rtree manifold3d mapbox-earcut pytest

## 2. Link the addon into FreeCAD's Mod directory

    # Linux (FreeCAD >= 1.1 uses version-scoped dirs, e.g. .../FreeCAD/v1-1/Mod;
    # check App.getUserAppDataDir() in the Python console, then adjust)
    ln -s /path/to/meshtofeatures ~/.local/share/FreeCAD/Mod/MeshToFeatures
    # Flatpak
    ln -s /path/to/meshtofeatures ~/.var/app/org.freecad.FreeCAD/data/FreeCAD/v1-1/Mod/MeshToFeatures
    # Windows (admin PowerShell)
    New-Item -ItemType SymbolicLink -Path "$env:APPDATA\FreeCAD\Mod\MeshToFeatures" -Target "C:\path\to\meshtofeatures"

## 3. Headless smoke test (the important one)

    cd /path/to/meshtofeatures
    freecadcmd freecad/meshtofeatures_wb/scripts/freecad_smoke_test.py
    # Flatpak
    flatpak run --command=FreeCADCmd org.freecad.FreeCAD freecad/meshtofeatures_wb/scripts/freecad_smoke_test.py

Expected: every line `[PASS]`, ending in `ALL CHECKS PASSED`, and a saved
`meshtofeatures_smoke.FCStd` you can open to see the reconstructed parts and
their surface-patch overlay.

Optionally, run the full pytest suite with FreeCAD's own interpreter to
verify the geometry core against the exact numpy/scipy/trimesh versions
FreeCAD ships:

    <freecad-python> -m pytest freecad/meshtofeatures_wb/tests/ -q
    # Flatpak
    flatpak run --command=python3 org.freecad.FreeCAD -m pytest freecad/meshtofeatures_wb/tests/ -q

Known things to watch for (please report which occur, with FreeCAD version):
- [ ] `Part.Cone` rejecting `Radius = 0` (adapter places the reference
      circle at the patch's lower bound, so radius 0 only occurs for
      segments reaching the apex).
- [ ] `package.xml` schema warnings in the Addon Manager log.
- [ ] Icon not shown (SVG renderer differences).

## 4. GUI walkthrough

1. Start FreeCAD -> workbench selector should list **MeshToFeatures**.
2. `File > New`, then `Mesh` workbench -> create/import any STL
   (or: Part workbench, make a cylinder, then `Mesh > Create mesh from shape`).
3. Select the mesh object, switch to **MeshToFeatures**, click
   **Reconstruct surfaces (snapped)**.
4. Expect: a `Reconstruction of <name>` group with color-coded faces
   (orange cylinders, blue planes, green cones, magenta spheres), a
   summary line in the Report View, and the snapping audit trail both in
   the Report View and in the group's `SnapActions` property.
5. Repeat with **raw fits** and compare parameters in the labels.

## 5. Task panel & threading (v0.10)

1. Select a mesh, run **MeshToFeatures panel...** -> task panel opens with
   options, a progress bar, and a results list.
2. Click **Run**: the progress bar advances with stage names and the 3D
   view stays responsive (orbit while it runs).
3. Toolbar commands now show a progress dialog instead of freezing.
4. Enable "Rebuild as PartDesign body" in the panel and re-run: a Body
   appears alongside the patch overlay.
Watch for: Qt import name (`from PySide import ...`) failing on your
build -- report the exact error if so.

### Fillets (v0.12)
The smoke test rebuilds a top-edge-filleted plate and asserts two
PartDesign::Fillet features and a 1% volume match. In the GUI, "Rebuild
as PartDesign body" on a part with horizontal rounded edges should show
Fillet features at the end of the tree; report any "no body edge
matched" warnings with the part.

## 6. Known limitations (do not report as bugs)

The rebuild emits an editable PartDesign Body (a real solid with history),
not just surface patches. The remaining limitations are documented in the
README's "Limitations" section; in short:

- The pipeline targets *prismatic* parts (planes, cylinders, cones, plus
  conical countersinks/pockets, tapered bosses, spherical caps, and tori);
  organic / sculpted / 3D-scanned shapes will not reconstruct meaningfully.
- Full spheres and standalone donuts are fitted by the core but not yet
  rebuilt as features.
- Freeform transition bands are carried by the terminal correction pass,
  not analytic primitives (the parametric body approximates them).
