# SPDX-License-Identifier: LGPL-2.1-or-later
"""GUI runner and task panel: pipeline off the main thread.

FreeCAD rule: document and GUI operations happen on the MAIN thread only.
The pipeline itself is pure Python, so the split is:

  main thread  -> convert Mesh object, start worker, poll a queue with a
                  QTimer, apply results (emit patches / build body)
  worker thread-> reconstruct + snap + features + patterns (+ plan),
                  pushing ('progress', stage, frac) and finally
                  ('done', payload) or ('error', text) into the queue.

Nothing in the worker touches FreeCAD objects.
"""

from __future__ import annotations

import queue
import threading
import traceback

import FreeCAD as App  # type: ignore
import FreeCADGui as Gui  # type: ignore
from PySide import QtCore, QtWidgets  # type: ignore  # FreeCAD's Qt shim


def compute(tm, snap: bool, rebuild: bool, progress=None) -> dict:
    """Pure pipeline; safe on any thread. Returns everything the
    main-thread apply step needs."""
    from meshtofeatures.pipeline import reconstruct
    from meshtofeatures.snapping import snap_report
    from meshtofeatures.emission import plan_patches
    from meshtofeatures.features import detect_features
    from meshtofeatures.patterns import detect_patterns
    from meshtofeatures.history import plan_history

    def sub(lo, hi):
        return (lambda s, f: progress(s, lo + (hi - lo) * f)) if progress else None

    report = reconstruct(tm, progress=sub(0.0, 0.62))
    actions = None
    if snap:
        result = snap_report(report, progress=sub(0.62, 0.80))
        report, actions = result.report, result.actions
    if progress:
        progress("detecting features", 0.85)
    patches = plan_patches(report)
    feats = detect_features(report, patches)
    pats = detect_patterns(feats)
    plan = None
    plan_error = None
    if rebuild:
        if progress:
            progress("planning history", 0.93)
        try:
            plan = plan_history(report, feats, pats, patches)
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            plan_error = str(exc)
    if progress:
        progress("finished", 1.0)
    return {"report": report, "actions": actions, "patches": patches,
            "features": feats, "patterns": pats, "plan": plan,
            "plan_error": plan_error}


def result_lines(res: dict) -> list[str]:
    report, feats, pats = res["report"], res["features"], res["patterns"]
    lines = [f"{len(report.surfaces)} surfaces recognized "
             f"({', '.join(sorted(set(report.kinds()))) or 'none'}), "
             f"{len(report.unrecognized)} unrecognized, "
             f"coverage {report.coverage:.1%}"]
    in_pattern = {id(m) for p in pats.patterns for m in p.members}
    lines += [f"PATTERN: {p.description}" for p in pats.patterns]
    lines += [f"FEATURE: {f.description}" for f in feats.features
              if id(f) not in in_pattern]
    plan = res.get("plan")
    if plan is not None:
        lines += [f"STEP: {lbl}" for lbl in getattr(plan, "step_labels", [])]
        for item in getattr(plan, "unplanned", []):
            lines.append(f"NOT REBUILT: {item} (outside the single-axis "
                         f"prismatic model)")
        if feats.unassigned:
            lines.append(f"{len(feats.unassigned)} surfaces belong to no "
                         f"recognized feature (e.g. chamfers) and are not "
                         f"rebuilt")
    if res.get("plan_error"):
        lines.append(f"history plan failed: {res['plan_error']}")
    if res["actions"]:
        lines += [f"[{a.kind}] {'OK ' if a.accepted else 'REJECTED '}{a.detail}"
                  for a in res["actions"]]
    return lines


def apply_results(doc, mesh_obj, res: dict) -> None:
    """Main-thread only: create document objects from computed results."""
    from . import emit
    group = emit.emit_report(
        doc, res["patches"], res["actions"],
        group_label=f"Reconstruction of {mesh_obj.Label}",
        features=res["features"], patterns=res["patterns"])
    for line in result_lines(res):
        App.Console.PrintMessage(f"[meshtofeatures]   {line}\n")
    App.Console.PrintMessage(
        f"[meshtofeatures] '{mesh_obj.Label}' -> group '{group.Label}'\n")
    if res["plan"] is not None:
        from . import build
        body = build.build_body(doc, res["plan"],
                                name=f"Rebuilt_{mesh_obj.Name}")
        App.Console.PrintMessage(
            f"[meshtofeatures] '{mesh_obj.Label}' -> body '{body.Label}'\n")


class _Runner(QtCore.QObject):
    """Worker thread + 100 ms main-thread poll timer."""

    def __init__(self, tm, snap, rebuild, on_progress, on_done, on_error):
        super().__init__()
        self._q: queue.Queue = queue.Queue()
        self._on = (on_progress, on_done, on_error)

        def work():
            try:
                res = compute(tm, snap, rebuild,
                              progress=lambda s, f: self._q.put(("progress", s, f)))
                self._q.put(("done", res))
            except Exception:  # noqa: BLE001
                self._q.put(("error", traceback.format_exc()))

        self._thread = threading.Thread(target=work, daemon=True)
        self._timer = QtCore.QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def start(self):
        self._thread.start()
        self._timer.start()

    def _poll(self):
        on_progress, on_done, on_error = self._on
        try:
            while True:
                item = self._q.get_nowait()
                if item[0] == "progress":
                    on_progress(item[1], item[2])
                else:
                    self._timer.stop()
                    (on_done if item[0] == "done" else on_error)(item[1])
                    return
        except queue.Empty:
            pass


_active_runners: list = []   # keep references alive


def run_async(mesh_obj, snap: bool, rebuild: bool) -> None:
    """Threaded execution with a progress dialog (used by the toolbar
    commands)."""
    from . import emit
    tm = emit.mesh_object_to_trimesh(mesh_obj)      # main thread, cheap
    doc = App.ActiveDocument or App.newDocument("Reconstruction")

    dlg = QtWidgets.QProgressDialog(
        f"Reconstructing '{mesh_obj.Label}'...", None, 0, 100,
        Gui.getMainWindow())
    dlg.setWindowTitle("MeshToFeatures")
    dlg.setMinimumDuration(0)
    dlg.setValue(0)

    def on_progress(stage, frac):
        dlg.setLabelText(f"'{mesh_obj.Label}': {stage}")
        dlg.setValue(int(100 * frac))

    def on_done(res):
        try:
            apply_results(doc, mesh_obj, res)
        finally:
            dlg.close()
            _active_runners.remove(runner)

    def on_error(text):
        dlg.close()
        _active_runners.remove(runner)
        App.Console.PrintError(f"[meshtofeatures] failed:\n{text}")

    runner = _Runner(tm, snap, rebuild, on_progress, on_done, on_error)
    _active_runners.append(runner)
    runner.start()


class MeshToFeaturesTaskPanel:
    """Task panel: options, progress, and a browsable results list."""

    def __init__(self, mesh_objs):
        self._meshes = list(mesh_objs)
        w = QtWidgets.QWidget()
        w.setWindowTitle("MeshToFeatures")
        lay = QtWidgets.QVBoxLayout(w)
        self._label = QtWidgets.QLabel(
            f"Selected: {', '.join(m.Label for m in self._meshes)}")
        self._label.setWordWrap(True)
        self._snap = QtWidgets.QCheckBox("Snap parameters to design intent")
        self._snap.setChecked(True)
        self._rebuild = QtWidgets.QCheckBox("Rebuild as PartDesign body")
        self._run = QtWidgets.QPushButton("Run")
        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._stage = QtWidgets.QLabel("")
        self._results = QtWidgets.QListWidget()
        for x in (self._label, self._snap, self._rebuild, self._run,
                  self._bar, self._stage, self._results):
            lay.addWidget(x)
        self._run.clicked.connect(self._start)
        self.form = w

    def _start(self):
        self._run.setEnabled(False)
        self._results.clear()
        self._pending = list(self._meshes)
        self._next()

    def _next(self):
        if not self._pending:
            self._run.setEnabled(True)
            self._stage.setText("finished")
            return
        from . import emit
        mesh_obj = self._pending.pop(0)
        tm = emit.mesh_object_to_trimesh(mesh_obj)
        doc = App.ActiveDocument or App.newDocument("Reconstruction")

        def on_progress(stage, frac):
            self._stage.setText(f"'{mesh_obj.Label}': {stage}")
            self._bar.setValue(int(100 * frac))

        def on_done(res):
            try:
                apply_results(doc, mesh_obj, res)
                self._results.addItem(f"=== {mesh_obj.Label} ===")
                for line in result_lines(res):
                    self._results.addItem(line)
            finally:
                _active_runners.remove(runner)
                self._next()

        def on_error(text):
            self._results.addItem(f"=== {mesh_obj.Label}: FAILED ===")
            App.Console.PrintError(f"[meshtofeatures] failed:\n{text}")
            _active_runners.remove(runner)
            self._next()

        runner = _Runner(tm, self._snap.isChecked(),
                         self._rebuild.isChecked(),
                         on_progress, on_done, on_error)
        _active_runners.append(runner)
        runner.start()

    def getStandardButtons(self):  # noqa: N802 (FreeCAD API)
        return QtWidgets.QDialogButtonBox.Close

    def reject(self):  # noqa: N802
        Gui.Control.closeDialog()
        return True
