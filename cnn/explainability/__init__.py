"""Explainability helpers for Grad-CAM and SHAP."""

from .gradcam import (
	HeadwiseGradCAMPlusPlus,
	HeadwiseGradCAMResult,
	build_head_overlay_frames,
	build_headwise_grid_frames,
	prepare_clip_for_headwise_gradcam,
	save_headwise_grid_video,
)
from .head_analysis import summarize_head_outputs
from .shap_utils import (
	SemanticShapTarget,
	aggregate_shap_values,
	collect_background_tensor,
	create_gradient_explainer,
	explain_clip_with_shap,
	load_semantic_model,
	load_video_clip_for_shap,
	resolve_concept_index,
)

