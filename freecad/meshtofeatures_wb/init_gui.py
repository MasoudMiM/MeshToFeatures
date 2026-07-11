# SPDX-License-Identifier: LGPL-2.1-or-later
"""Workbench registration (imported by FreeCAD at GUI startup)."""

import os

import FreeCADGui as Gui  # type: ignore


class MeshToFeaturesWorkbench(Gui.Workbench):
    MenuText = "MeshToFeatures"
    ToolTip = "Reverse-engineer analytic surfaces from meshes (STL etc.)"
    Icon = os.path.join(os.path.dirname(__file__), "resources", "meshtofeatures.svg")

    def Initialize(self):  # noqa: N802 (FreeCAD API)
        from . import commands
        commands.register()
        names = list(commands.COMMANDS.keys())
        self.appendToolbar("MeshToFeatures", names)
        self.appendMenu("&MeshToFeatures", names)

    def GetClassName(self):  # noqa: N802
        return "Gui::PythonWorkbench"


Gui.addWorkbench(MeshToFeaturesWorkbench())
