# SPDX-License-Identifier: LGPL-2.1-or-later
"""meshtofeatures: reverse-engineer analytic surfaces from triangle meshes.

FreeCAD-independent geometry core of a (working title) FreeCAD
reverse-engineering addon. Pipeline: mesh -> smooth-region segmentation ->
per-segment primitive fitting with model selection.
"""

from .primitives import Plane, Sphere, Cylinder, Cone, Primitive
from .fitting import FitResult, fit_plane, fit_sphere, fit_cylinder, fit_cone, fit_best
from .segmentation import Segment, segment_mesh
from .pipeline import RecognizedSurface, ReconstructionReport, reconstruct
from .snapping import SnapConfig, SnapAction, SnapResult, snap_report
from .emission import PatchSpec, plan_patches
from .conditioning import ConditioningReport, condition_mesh
from .features import Feature, FeatureReport, detect_features
from .patterns import Pattern, PatternReport, detect_patterns
from .history import (BuildPlan, SketchArc, SketchCircle, SketchLine,
                      fillet_edge_matches, hole_op_properties,
                      loop_to_sketch, plan_history)
from .standards import identify_metric
from .segmentation import adaptive_angle_threshold, face_curvature, split_by_curvature

__version__ = "0.16.0"
__all__ = [
    "Plane", "Sphere", "Cylinder", "Cone", "Primitive",
    "FitResult", "fit_plane", "fit_sphere", "fit_cylinder", "fit_cone", "fit_best",
    "Segment", "segment_mesh",
    "RecognizedSurface", "ReconstructionReport", "reconstruct",
    "SnapConfig", "SnapAction", "SnapResult", "snap_report",
    "PatchSpec", "plan_patches",
    "ConditioningReport", "condition_mesh",
    "Feature", "FeatureReport", "detect_features",
    "Pattern", "PatternReport", "detect_patterns",
    "BuildPlan", "SketchLine", "SketchArc", "SketchCircle",
    "loop_to_sketch", "plan_history", "hole_op_properties",
    "fillet_edge_matches",
    "identify_metric",
    "adaptive_angle_threshold", "face_curvature", "split_by_curvature",
]
