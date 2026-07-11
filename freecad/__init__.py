# SPDX-License-Identifier: LGPL-2.1-or-later
# This file MUST contain the pkgutil boilerplate below and nothing else
# executable. FreeCAD 1.x ships its own `freecad` package (module_io,
# utils, ...); our addon contributes `freecad.meshtofeatures_wb` to the same
# namespace. Without extend_path, whichever `freecad/` directory Python
# finds first SHADOWS all others -- in practice our Mod directory wins and
# FreeCAD's own `from freecad import module_io` breaks, killing file
# import for the whole application. See tests/test_namespace.py.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
