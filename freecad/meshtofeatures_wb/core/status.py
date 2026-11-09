# SPDX-License-Identifier: LGPL-2.1-or-later
"""Per-feature deviation report (phase 3): the trust data as a table.

The pipeline already MEASURES its own trust: :func:`.solidify.plan_corrections`
localizes the deviation between the parametric rebuild and the source mesh
as OVER/UNDER patch components; :func:`.build.build_body` records (via the
executor's loud logging, now via an :class:`ExecutionLedger`) which features
were rebuilt, deferred, recovered, dressed geometrically, or skipped;
``plan.unplanned`` and ``plan.undressable`` name what recognition saw but the
model could not rebuild. None of it was ever shown as ONE view. This module
is the plumbing that composes those existing numbers into per-feature rows
-- no new pipeline math: attribution only, matching each already-computed
correction component to the planned feature it lies nearest to.

Pure Python (numpy only), FreeCAD-free, so the whole report is testable
without the executor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["EXECUTION_STATUSES", "ExecutionLedger", "FeatureRow",
           "attribute_corrections", "feature_report",
           "format_feature_report"]

#: Executor outcome per planned op. ``rebuilt`` is the default (a feature
#: that computes and stays in the chain); ``deferred`` means it failed in
#: place and was rebuilt at the END of the chain (same solid, later step);
#: ``recovered`` means an epsilon retry fixed an in-place failure;
#: ``geometric`` means a blend dressed at the mesh-fit position because the
#: parametric body had no edge there; ``fallback`` means the semantic
#: PartDesign feature was unavailable and a pocket/primitive cut was used;
#: ``skipped`` means the feature is NOT in the rebuilt part.
EXECUTION_STATUSES = frozenset(
    {"rebuilt", "deferred", "recovered", "geometric", "fallback", "skipped"})

#: merge priority when several events hit one planned op: failures outrank
#: successes so a skip is never hidden by a sibling sub-op's success
_STATUS_RANK = {"rebuilt": 0, "recovered": 1, "fallback": 1,
                "geometric": 1, "deferred": 2, "skipped": 3}

#: the table prints human status words
_STATUS_LABEL = {
    "rebuilt": "rebuilt",
    "deferred": "deferred -> rebuilt at chain end",
    "recovered": "rebuilt (after retry)",
    "geometric": "rebuilt (geometric dress)",
    "fallback": "rebuilt (pocket fallback)",
    "skipped": "SKIPPED - not in the part",
}

#: ledger categories that are SUB-cuts of another row's feature: recorded
#: under their own key so a sub-op failure can never mask (or be masked by)
#: the parent feature's own outcome, and rolled up into the parent row's
#: detail (a drop is never silent, but never a lie either).
_SUB_KINDS = {"hole": ("counterbore", "countersink")}


class ExecutionLedger:
    """What the executor did with each planned op: ``record(kind, k, ...)``
    from the build loops; ``get(kind, k)`` for the report. Ops with no
    recorded event default to ``rebuilt``."""

    def __init__(self):
        self._events: dict[tuple[str, int], tuple[str, str]] = {}

    def record(self, kind: str, k: int, status: str, detail: str = "") -> None:
        if status not in EXECUTION_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        key = (kind, k)
        cur = self._events.get(key)
        if cur is not None:
            rank_cur, rank_new = _STATUS_RANK[cur[0]], _STATUS_RANK[status]
            if rank_cur > rank_new:
                return            # never hide the worse outcome
            if rank_cur == rank_new and cur[0] == status and cur[1] \
                    and detail and cur[1] != detail:
                self._events[key] = (status, f"{cur[1]}; {detail}")
                return
        self._events[key] = (status, detail)

    def get(self, kind: str, k: int) -> tuple[str, str]:
        return self._events.get((kind, k), ("rebuilt", ""))


@dataclass
class FeatureRow:
    """One table line: a planned feature, how the executor treated it, and
    the deviation volume attributed to it (mm3; ``+`` = the rebuild lacks
    material there (UNDER), ``-`` = the rebuild overshoots (OVER)); ``0``
    when no correction component belongs to this feature."""

    kind: str                  # plan category: 'hole', 'pocket', ...
    label: str
    status: str                # EXECUTION_STATUSES member ('unplanned'
                               # and 'undressable' for never-planned)
    detail: str = ""
    deviation_mm3: float = 0.0


def _frame_of(plan) -> np.ndarray:
    t = np.eye(4)
    t[:3, 0] = plan.frame_x
    t[:3, 1] = plan.frame_y
    t[:3, 2] = plan.frame_z
    t[:3, 3] = plan.frame_origin
    return t


def _to_world(plan, pts) -> list[np.ndarray]:
    m = _frame_of(plan)
    return [(m @ np.array([p[0], p[1], p[2], 1.0]))[:3] for p in pts]


def _loop_centroid(profile):
    """Rough in-plane centroid of a sketch profile (its point primitives),
    or ``None`` when the profile carries no points at all (a malformed op:
    no site, so no deviation can be attributed to it -- never a bogus
    point at the frame origin)."""
    pts = []
    for p in profile or []:
        for attr in ("start", "end"):
            q = getattr(p, attr, None)
            if q is not None:
                pts.append(np.asarray([float(q[0]), float(q[1])]))
        c = getattr(p, "center", None)
        if c is not None:
            pts.append(np.asarray([float(c[0]), float(c[1])]))
    if not pts:
        return None
    return sum(pts) / len(pts)


def op_sites(plan) -> list[list[np.ndarray]]:
    """Representative WORLD points of every planned op, in :func:`plan_ops`
    order: where the feature lives, as a point set the deviation components
    can be matched against."""
    L = float(plan.base.length)
    sites: list[list[np.ndarray]] = []

    base_c = _loop_centroid(plan.base.profile)
    sites.append([] if base_c is None
                 else _to_world(plan, [(base_c[0], base_c[1], L / 2.0)]))

    for p in getattr(plan, "pads", []):
        if getattr(p, "axis", None) is not None:
            po = getattr(p, "plane_origin", None)
            # one site per pad row (alignment with plan_ops order is part of
            # the contract); a lateral pad without a recorded plane origin
            # falls back to the part centre rather than dropping the row
            if po is not None:
                sites.append([np.asarray(po, dtype=float)])
            else:
                sites.append(_to_world(plan, [(0.0, 0.0, L / 2.0)]))
            continue
        c = _loop_centroid(p.profile)
        if c is None:
            sites.append([])
            continue
        face = L if getattr(p, "from_top", True) else 0.0
        out = face + p.length if getattr(p, "from_top", True) else face - p.length
        sites.append(_to_world(plan, [(c[0], c[1], face), (c[0], c[1], out)]))

    for pk in getattr(plan, "pockets", []):
        c = _loop_centroid(pk.profile)
        if c is None:
            sites.append([])
            continue
        top = getattr(pk, "from_top", True)
        if top:
            face, far = L, (0.0 if getattr(pk, "through", False)
                            else L - float(pk.depth))
        else:
            face, far = 0.0, (L if getattr(pk, "through", False)
                              else float(pk.depth))
        sites.append(_to_world(plan, [(c[0], c[1], face), (c[0], c[1], far)]))

    for h in getattr(plan, "holes", []):
        top = getattr(h, "from_top", True)
        open_z = (float(h.surface_z) if getattr(h, "surface_z", None)
                  is not None else (L if top else 0.0))
        if getattr(h, "through", False):
            far_z = 0.0 if top else L
        else:
            far_z = open_z - float(h.depth) if top else open_z + float(h.depth)
        pts = []
        for (x, y) in h.positions:
            pts.append((float(x), float(y), open_z))
            pts.append((float(x), float(y), far_z))
            cb = getattr(h, "counterbore_depth", None)
            if getattr(h, "counterbore_diameter", None) and cb:
                pts.append((float(x), float(y),
                            open_z - float(cb) if top else open_z + float(cb)))
        sites.append(_to_world(plan, pts))

    for ch in getattr(plan, "cross_holes", []):
        sites.append([np.asarray(p, dtype=float) for p in ch.positions3d])

    for op in list(getattr(plan, "cones", [])) \
            + list(getattr(plan, "spheres", [])) \
            + list(getattr(plan, "tori", [])):
        face = (float(op.surface_z) if getattr(op, "surface_z", None)
                is not None else (L if op.from_top else 0.0))
        pts = [(float(x), float(y), face) for (x, y) in op.positions]
        pts += [(float(x) + 0.5 * float(getattr(op, "rim_radius", 0.0)),
                 float(y), face) for (x, y) in op.positions]
        sites.append(_to_world(plan, pts))

    for op in list(getattr(plan, "fillets", [])) \
            + list(getattr(plan, "chamfers", [])):
        mid = 0.5 * (np.asarray(op.edge_start, dtype=float)
                     + np.asarray(op.edge_end, dtype=float))
        sites.append([mid])
    return sites


def plan_ops(plan):
    """(category, index, op) for every planned feature, in report order
    (matches :func:`op_sites` order)."""
    yield ("base", 0, plan.base)
    for k, op in enumerate(getattr(plan, "pads", [])):
        yield ("pad", k, op)
    for k, op in enumerate(getattr(plan, "pockets", [])):
        yield ("pocket", k, op)
    for k, op in enumerate(getattr(plan, "holes", [])):
        yield ("hole", k, op)
    for k, op in enumerate(getattr(plan, "cross_holes", [])):
        yield ("cross-hole", k, op)
    for k, op in enumerate(getattr(plan, "cones", [])):
        yield ("cone", k, op)
    for k, op in enumerate(getattr(plan, "spheres", [])):
        yield ("sphere", k, op)
    for k, op in enumerate(getattr(plan, "tori", [])):
        yield ("torus", k, op)
    for k, op in enumerate(getattr(plan, "fillets", [])):
        yield ("fillet", k, op)
    for k, op in enumerate(getattr(plan, "chamfers", [])):
        yield ("chamfer", k, op)


def _diag(plan) -> float:
    xs, ys = [0.0], [0.0]
    for p in plan.base.profile or []:
        for attr in ("start", "end"):
            q = getattr(p, attr, None)
            if q is not None:
                xs.append(float(q[0]))
                ys.append(float(q[1]))
    diag = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    return max(float(np.hypot(diag, float(plan.base.length))), 1e-9)


def attribute_corrections(plan, tol: float | None = None):
    """Assign each plan_corrections component to the planned feature whose
    site set lies closest to the component's centroid.

    Returns a list sized like :func:`plan_ops`: signed mm3 per feature
    (``+`` UNDER -- the rebuild lacks this material there, ``-`` OVER --
    the rebuild overshoots) plus the total unattributed mm3 (a patch too
    far from every feature -- a freeform transition no feature owns).
    ``tol`` defaults to 10% of the part diagonal: a component beyond every
    feature's extent is freeform, not the feature's fault.
    """
    ops = list(plan_ops(plan))
    sites = op_sites(plan)
    out = [0.0] * len(ops)
    unattr_under = unattr_over = 0.0
    if not sites:
        return out, (0.0, 0.0)
    if tol is None:
        tol = 0.10 * _diag(plan)
    for corr in getattr(plan, "corrections", None) or []:
        comp = getattr(corr, "mesh", None)
        if comp is None or len(comp.faces) == 0:
            continue
        c = np.asarray(comp.centroid, dtype=float)
        # an op with no usable site points (malformed plan) can never
        # attract a component: +inf so the nearest-feature search skips it
        dists = [(min(float(np.linalg.norm(c - p)) for p in s) if s
                  else float("inf")) for s in sites]
        j = int(np.argmin(dists))
        # OVER is 'cut' (remove from rebuild), UNDER is 'add'
        signed = float(corr.volume) if corr.kind == "add" \
            else -float(corr.volume)
        if dists[j] > tol:
            if signed >= 0:
                unattr_under += signed
            else:
                unattr_over += -signed
            continue
        out[j] += signed
    return out, (unattr_under, unattr_over)


def _op_label(kind: str, k: int, op) -> str:
    lab = getattr(op, "label", "") or ""
    return lab or f"{kind.capitalize()} {k}"


def feature_report(plan, ledger: ExecutionLedger | None = None):
    """One :class:`FeatureRow` per planned feature (executor status +
    attributed deviation), then the unplanned/undressable detections, then
    the correction-pass summary. Numbers are the ones other pipeline
    stages already computed; nothing here re-fits anything."""
    ledger = ledger or ExecutionLedger()
    dev, (un_under, un_over) = attribute_corrections(plan)
    rows: list[FeatureRow] = []
    for i, (kind, k, op) in enumerate(plan_ops(plan)):
        status, detail = ledger.get(kind, k)
        # sub-cut failures (counterbore/countersink of a hole) never change
        # the parent's status -- they could lie about a recovered bore or
        # hide behind one -- they are rolled into the detail instead, loud
        # enough that grepping the report for a failure word finds it
        subs = []
        for sub in _SUB_KINDS.get(kind, ()):
            sst, sdt = ledger.get(sub, k)
            if sst != "rebuilt":
                subs.append(_STATUS_LABEL.get(sst, sst)
                            + (f" ({sdt})" if sdt else "") + f" [{sub}]")
        if subs:
            detail = "; ".join([detail] + subs) if detail else "; ".join(subs)
        rows.append(FeatureRow(kind=kind, label=_op_label(kind, k, op),
                               status=status, detail=detail,
                               deviation_mm3=dev[i] if i < len(dev) else 0.0))
    for item in getattr(plan, "unplanned", []):
        rows.append(FeatureRow(kind="unplanned", label=str(item),
                               status="unplanned",
                               detail="outside the single-axis prismatic model"))
    for item in getattr(plan, "undressable", []):
        rows.append(FeatureRow(kind="blend", label=str(item),
                               status="undressable",
                               detail="detected, undressable: degenerate frame"))
    return rows, (un_under, un_over)


def format_feature_report(rows, *, src_volume=None, body_volume=None,
                          corrected_volume=None, unattributed=(0.0, 0.0),
                          name="") -> list[str]:
    """The table as plain report-view lines. Overall volume ratios come
    from the existing pipeline numbers (source mesh, parameteric body,
    corrected solid), never from new measurements."""
    lines = []
    head = "FEATURE REPORT" + (f" for {name}" if name else "")
    ratios = []
    if src_volume and body_volume is not None:
        ratios.append(f"body {100.0 * body_volume / src_volume:.2f}% "
                      f"of source volume")
    if src_volume and corrected_volume is not None:
        ratios.append(f"corrected "
                      f"{100.0 * corrected_volume / src_volume:.2f}%")
    if ratios:
        head += "  (" + ", ".join(ratios) + ")"
    lines.append(head)
    un_under, un_over = unattributed
    for r in rows:
        status = _STATUS_LABEL.get(r.status, r.status)
        dev = ""
        if abs(r.deviation_mm3) > 0.0:
            side = "under" if r.deviation_mm3 > 0 else "over"
            dev = f"   {abs(r.deviation_mm3):.2f} mm3 {side}"
        detail = f"   ({r.detail})" if r.detail else ""
        lines.append(f"  {status:36} {r.label:34}{detail}{dev}")
    tail = []
    if any(abs(r.deviation_mm3) > 0.0 for r in rows):
        tail.append("mm3: deviation of the parametric model from the "
                    "source mesh -- exactly the patches the terminal "
                    "correction pass absorbs; not the body's own error")
    if un_under or un_over:
        tail.append("freeform deviation carried by the correction pass: "
                    f"{un_under:.2f} mm3 under, {un_over:.2f} mm3 over")
    for t in tail:
        lines.append("  " + t)
    if not rows:
        lines.append("  (nothing planned to rebuild)")
    return lines
