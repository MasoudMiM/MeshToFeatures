# SPDX-License-Identifier: LGPL-2.1-or-later
"""Standard hole-size identification: diameters carry manufacturing intent.

A 6.6 mm hole is an M6 clearance hole (ISO 273 medium fit); a 5.0 mm hole
is an M6 tap drill (coarse thread). Recovering the standard turns a
number into a decision. Only unambiguous table matches are reported --
non-standard sizes return None rather than the nearest guess.
"""

from __future__ import annotations

__all__ = ["identify_metric"]

#: ISO 273 clearance holes, medium fit (mm)
_CLEARANCE = {2.4: "M2", 2.9: "M2.5", 3.4: "M3", 4.5: "M4", 5.5: "M5",
              6.6: "M6", 9.0: "M8", 11.0: "M10", 13.5: "M12",
              17.5: "M16", 22.0: "M20", 26.0: "M24"}

#: tap drills for ISO coarse threads (mm)
_TAP = {1.6: "M2", 2.05: "M2.5", 2.5: "M3", 3.3: "M4", 4.2: "M5",
        5.0: "M6", 6.8: "M8", 8.5: "M10", 10.2: "M12",
        14.0: "M16", 17.5: "M20", 21.0: "M24"}


def identify_metric(diameter: float, rtol: float = 0.015) -> str | None:
    """Name the metric standard a hole diameter implements, or None.

    ``rtol`` is the relative window (default 1.5%: tight enough that M3
    clearance at 3.4 and M4 tap at 3.3 never blur, wide enough for
    snapped real-scan diameters). 17.5 appears in both tables (M16
    clearance / M20 tap): the clearance reading wins as the more common
    intent, deliberately.
    """
    best = None
    best_err = rtol
    for table, kind in ((_CLEARANCE, "clearance"), (_TAP, "tap drill")):
        for ref, name in table.items():
            err = abs(diameter - ref) / ref
            if err < best_err or (err == best_err and best is None):
                best_err = err
                best = f"{name} {kind}"
    return best
