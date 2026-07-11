# SPDX-License-Identifier: LGPL-2.1-or-later
"""GUI commands for the meshtofeatures workbench."""

from __future__ import annotations

import os

import FreeCAD as App  # type: ignore
import FreeCADGui as Gui  # type: ignore

_ICON_DIR = os.path.join(os.path.dirname(__file__), "resources")


def _missing_dependencies() -> list[str]:
    missing = []
    for name in ("numpy", "scipy", "trimesh"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def _selected_mesh_objects():
    return [
        o for o in Gui.Selection.getSelection()
        if getattr(o, "TypeId", "") == "Mesh::Feature"
    ]


def _run(snap: bool) -> None:
    missing = _missing_dependencies()
    if missing:
        App.Console.PrintError(
            "[meshtofeatures] missing Python packages: " + ", ".join(missing)
            + ". Install them into FreeCAD's Python, e.g.:\n"
            "    <FreeCAD python> -m pip install " + " ".join(missing) + "\n")
        return

    if App.GuiUp:
        from . import ui
        for mesh_obj in _selected_mesh_objects():
            ui.run_async(mesh_obj, snap=snap, rebuild=False)
        return

    from meshtofeatures.pipeline import reconstruct
    from meshtofeatures.snapping import snap_report
    from meshtofeatures.emission import plan_patches
    from meshtofeatures.features import detect_features
    from meshtofeatures.patterns import detect_patterns
    from . import emit

    doc = App.ActiveDocument or App.newDocument("Reconstruction")
    for mesh_obj in _selected_mesh_objects():
        App.Console.PrintMessage(f"[meshtofeatures] reconstructing '{mesh_obj.Label}'...\n")
        tm = emit.mesh_object_to_trimesh(mesh_obj)
        report = reconstruct(tm)
        actions = None
        if snap:
            result = snap_report(report)
            report, actions = result.report, result.actions
            for a in actions:
                status = "OK " if a.accepted else "REJECTED "
                App.Console.PrintMessage(f"[meshtofeatures]   [{a.kind}] {status}{a.detail}\n")
        patches = plan_patches(report)
        features = detect_features(report, patches)
        pats = detect_patterns(features)
        pattern_lines = [f"PATTERN: {p.description}" for p in pats.patterns]
        in_pattern = {id(m) for p in pats.patterns for m in p.members}
        feature_lines = [f"FEATURE: {f.description}" for f in features.features
                         if id(f) not in in_pattern]
        for line in pattern_lines + feature_lines:
            App.Console.PrintMessage(f"[meshtofeatures]   {line}\n")
        group = emit.emit_report(
            doc, patches, actions,
            group_label=f"Reconstruction of {mesh_obj.Label}",
            features=features, patterns=pats)
        App.Console.PrintMessage(
            f"[meshtofeatures] '{mesh_obj.Label}': {len(report.surfaces)} surfaces "
            f"recognized ({', '.join(report.kinds()) or 'none'}), "
            f"{len(report.unrecognized)} unrecognized, "
            f"coverage {report.coverage:.1%} -> group '{group.Label}'\n")


class _BaseCommand:
    snap = True
    text = ""
    tooltip = ""
    icon = "meshtofeatures.svg"

    def GetResources(self):  # noqa: N802 (FreeCAD API)
        return {
            "Pixmap": os.path.join(_ICON_DIR, self.icon),
            "MenuText": self.text,
            "ToolTip": self.tooltip,
        }

    def IsActive(self):  # noqa: N802
        return bool(_selected_mesh_objects())

    def Activated(self):  # noqa: N802
        try:
            _run(snap=self.snap)
        except Exception as exc:  # noqa: BLE001 - report, never crash the GUI
            import traceback
            App.Console.PrintError(
                f"[meshtofeatures] reconstruction failed: {exc}\n"
                + traceback.format_exc())


class ReconstructSnapped(_BaseCommand):
    snap = True
    icon = "mtf_reconstruct_snapped.svg"
    text = "Reconstruct surfaces (snapped)"
    tooltip = ("Segment the selected mesh, fit analytic surfaces, and snap "
               "parameters to inferred design intent (with audit trail)")


class ReconstructRaw(_BaseCommand):
    snap = False
    icon = "mtf_reconstruct_raw.svg"
    text = "Reconstruct surfaces (raw fits)"
    tooltip = ("Segment the selected mesh and fit analytic surfaces without "
               "any parameter snapping")


class RebuildBody(_BaseCommand):
    icon = "mtf_rebuild_body.svg"
    text = "Rebuild as PartDesign body"
    tooltip = ("Reconstruct, detect features, infer the build history, and "
               "create an editable PartDesign Body (prismatic parts)")

    def Activated(self):  # noqa: N802
        try:
            missing = _missing_dependencies()
            if missing:
                App.Console.PrintError(
                    "[meshtofeatures] missing: " + ", ".join(missing) + "\n")
                return
            if App.GuiUp:
                from . import ui
                for mesh_obj in _selected_mesh_objects():
                    ui.run_async(mesh_obj, snap=True, rebuild=True)
                return
            from meshtofeatures.pipeline import reconstruct
            from meshtofeatures.snapping import snap_report
            from meshtofeatures.emission import plan_patches
            from meshtofeatures.features import detect_features
            from meshtofeatures.patterns import detect_patterns
            from meshtofeatures.history import plan_history
            from . import build, emit
            doc = App.ActiveDocument or App.newDocument("Rebuilt")
            for mesh_obj in _selected_mesh_objects():
                tm = emit.mesh_object_to_trimesh(mesh_obj)
                report = snap_report(reconstruct(tm)).report
                patches = plan_patches(report)
                feats = detect_features(report, patches)
                plan = plan_history(report, feats, detect_patterns(feats),
                                    patches)
                body = build.build_body(doc, plan,
                                        name=f"Rebuilt_{mesh_obj.Name}")
                App.Console.PrintMessage(
                    f"[meshtofeatures] '{mesh_obj.Label}' -> body "
                    f"'{body.Label}': base {plan.base.length:g}, "
                    f"{len(plan.holes)} hole op(s), {len(plan.pockets)} "
                    f"pocket(s), {len(plan.pads)} pad(s); unplanned: "
                    f"{plan.unplanned or 'none'}\n")
        except Exception as exc:  # noqa: BLE001
            import traceback
            App.Console.PrintError(f"[meshtofeatures] rebuild failed: {exc}\n"
                                   + traceback.format_exc())


class OpenPanel(_BaseCommand):
    icon = "mtf_panel.svg"
    text = "MeshToFeatures panel..."
    tooltip = "Open the MeshToFeatures task panel (options, progress, results)"

    def Activated(self):  # noqa: N802
        try:
            missing = _missing_dependencies()
            if missing:
                App.Console.PrintError(
                    "[meshtofeatures] missing: " + ", ".join(missing) + "\n")
                return
            from . import ui
            Gui.Control.showDialog(
                ui.MeshToFeaturesTaskPanel(_selected_mesh_objects()))
        except Exception as exc:  # noqa: BLE001
            import traceback
            App.Console.PrintError(f"[meshtofeatures] panel failed: {exc}\n"
                                   + traceback.format_exc())


COMMANDS = {
    "MeshToFeatures_OpenPanel": OpenPanel(),
    "MeshToFeatures_ReconstructSnapped": ReconstructSnapped(),
    "MeshToFeatures_ReconstructRaw": ReconstructRaw(),
    "MeshToFeatures_RebuildBody": RebuildBody(),
}


def register() -> None:
    for name, cmd in COMMANDS.items():
        Gui.addCommand(name, cmd)
