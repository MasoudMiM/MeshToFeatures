# SPDX-License-Identifier: LGPL-2.1-or-later
"""Regression test for the `freecad` namespace-shadowing bug.

FreeCAD 1.x has its own `freecad` package containing modules like
``module_io`` (the file-import dispatcher). Our addon contributes
``freecad.meshtofeatures_wb`` to that same namespace. If our
``freecad/__init__.py`` does not carry the pkgutil ``extend_path``
boilerplate, our directory *shadows* FreeCAD's package and
``from freecad import module_io`` fails inside FreeCAD, breaking file
import application-wide (observed in the field on FreeCAD 1.1.1/snap).

This test simulates the collision: a fake "FreeCAD-side" ``freecad``
package with a ``module_io`` module is placed on ``sys.path`` *after*
the addon root, exactly the losing position that triggered the bug.
Both halves of the namespace must remain importable.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ADDON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_init_carries_extend_path_boilerplate():
    src = open(os.path.join(ADDON_ROOT, "freecad", "__init__.py")).read()
    assert "extend_path" in src, (
        "freecad/__init__.py lost the pkgutil extend_path boilerplate; "
        "this shadows FreeCAD's own `freecad` package and breaks file "
        "import in the application")


def test_namespace_merges_with_freecad_side_package():
    with tempfile.TemporaryDirectory() as tmp:
        # fake FreeCAD-internal package, mirroring FreeCAD's own layout
        fc_side = os.path.join(tmp, "freecad_side")
        os.makedirs(os.path.join(fc_side, "freecad"))
        shutil.copy(
            os.path.join(ADDON_ROOT, "freecad", "__init__.py"),
            os.path.join(fc_side, "freecad", "__init__.py"))
        with open(os.path.join(fc_side, "freecad", "module_io.py"), "w") as f:
            f.write("MARKER = 'freecad-side'\n")

        # a subprocess gives a pristine import state; the addon root comes
        # FIRST on sys.path (the position in which shadowing occurred)
        code = (
            "import sys;"
            f"sys.path.insert(0, {fc_side!r});"
            f"sys.path.insert(0, {ADDON_ROOT!r});"
            "from freecad import module_io;"
            "import freecad.meshtofeatures_wb;"
            "assert module_io.MARKER == 'freecad-side';"
            "print('MERGED')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert "MERGED" in proc.stdout
