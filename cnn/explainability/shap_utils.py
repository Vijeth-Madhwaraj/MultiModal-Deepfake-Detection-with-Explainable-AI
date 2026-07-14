"""SHAP helpers for the semantic 3D CNN video model.

SHAP over a full 5D video tensor is expensive, so this module intentionally
uses a gradient-based explainer with a small background set and clip-level
aggregation. That keeps the runtime practical while still giving per-frame
attributions that can be visualized like heatmaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from configs.config import Config
from model import DEFAULT_CONCEPT_VOCABULARY, SemanticDeepfakeDetector, load_compatible_checkpoint
from video_dataset import (
    FaceAlignmentProcessor,
    VideoDataset,
    prepare_display_and_model_frame as prepare_aligned_display_and_model_frame,
)

from .visualization import save_overlay_video

try:  # pragma: no cover - import availability is validated at runtime
    import shap
except Exception:  # pragma: no cover - SHAP is optional at import time
    shap = None


TargetFamily = Literal["class", "concept"]
OutputKind = Literal["logit", "score"]


@dataclass(frozen=True)
class ShapTargetSpec:
    """Describe the model output being explained."""

    family: TargetFamily
    index: int
    output_kind: OutputKind = "logit"

    @property
    def display_name(self) -> str:
        if self.family == "class":
            return {0: "Real", 1: "Fake"}.get(self.index, f"class_{self.index}")
        concept_names = get_concept_vocabulary()
        if 0 <= self.index < len(concept_names):
            return concept_names[self.index]
        return f"concept_{self.index}"


def _require_shap() -> None:
    if shap is None:
        raise RuntimeError(
            "The 'shap' package is required for SHAP explanations. Install it with `pip install shap`."
        )


def get_concept_vocabulary(concept_vocabulary: Sequence[str] | None = None) -> Tuple[str, ...]:
    """Resolve the active concept vocabulary with a branch-safe fallback."""

    if concept_vocabulary is not None:
        return tuple(concept_vocabulary)

    config_concepts = getattr(Config, "CONCEPT_NAMES", None)
    if config_concepts:
        return tuple(config_concepts)
    return tuple(DEFAULT_CONCEPT_VOCABULARY)


def read_video_frames(video_path: str | Path) -> List[np.ndarray]:
    """Decode a video into RGB frames."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")
    return frames


def sample_center_clip(frames: Sequence[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Select a deterministic center clip and pad short videos by repetition."""

    if len(frames) >= clip_length:
        start_index = max((len(frames) - clip_length) // 2, 0)
        clip = list(frames[start_index : start_index + clip_length])
    else:
        clip = list(frames)
        last_frame = clip[-1]
        while len(clip) < clip_length:
            clip.append(last_frame)
    return clip[:clip_length]


def prepare_display_and_model_frame(
    frame_rgb: np.ndarray,
    frame_size: int,
    align_faces: bool = Config.ALIGN_FACES,
    face_aligner: FaceAlignmentProcessor | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the same face crop for display and model input."""

    display_frame, model_frame, _ = prepare_aligned_display_and_model_frame(
        frame_rgb,
        frame_size=frame_size,
        align_faces=align_faces,
        face_aligner=face_aligner,
    )
    return display_frame, model_frame


def frames_to_tensor(frames: Sequence[np.ndarray]) -> Tensor:
    """Convert preprocessed RGB frames to a float tensor with shape (1, 3, T, H, W)."""

    clip_array = np.asarray(frames, dtype=np.float32)
    clip_array = np.transpose(clip_array, (3, 0, 1, 2))
    return torch.from_numpy(clip_array).unsqueeze(0)


def load_video_clip_for_shap(
    video_path: str | Path,
    frame_size: int = Config.IMAGE_SIZE,
    clip_length: int = Config.CLIP_LENGTH,
    align_faces: bool = Config.ALIGN_FACES,
) -> Tuple[List[np.ndarray], Tensor]:
    """Return all source frames plus one fixed-length model input clip."""

    frames = read_video_frames(video_path)
    clip_frames = sample_center_clip(frames, clip_length=clip_length)

    model_frames: List[np.ndarray] = []
    face_aligner = FaceAlignmentProcessor() if align_faces else None
    try:
        for frame_rgb in clip_frames:
            _, model_frame = prepare_display_and_model_frame(
                frame_rgb,
                frame_size,
                align_faces=align_faces,
                face_aligner=face_aligner,
            )
            model_frames.append(model_frame)
    finally:
        if face_aligner is not None:
            face_aligner.close()

    # SHAP is calculated on the fixed-length center clip. The visualization
    # layer interpolates that temporal attribution volume over all source frames.
    return frames, frames_to_tensor(model_frames)


def _select_background_indices(total_size: int, desired_size: int, seed: int = 42) -> List[int]:
    """Pick a small, representative background subset."""

    if total_size <= 0:
        raise ValueError("Cannot select background samples from an empty dataset.")

    desired_size = max(1, min(desired_size, total_size))
    if desired_size == total_size:
        return list(range(total_size))

    # Even spacing keeps the background representative without loading too many clips.
    indices = np.linspace(0, total_size - 1, num=desired_size, dtype=np.int64)
    return sorted(set(int(index) for index in indices))


def collect_background_tensor(
    background_root: str | Path,
    num_background: int = 8,
    frame_size: int = Config.IMAGE_SIZE,
    clip_length: int = Config.CLIP_LENGTH,
    is_train: bool = False,
    align_faces: bool = Config.ALIGN_FACES,
) -> Tuple[Tensor, List[Path]]:
    """Collect a small batch of background clips for GradientExplainer.

    SHAP on a full video tensor is computationally heavy, so a tiny background
    set is the practical fallback here.
    """

    dataset = VideoDataset(
        root_dir=background_root,
        clip_length=clip_length,
        frame_size=frame_size,
        is_train=is_train,
        align_faces=align_faces,
    )

    selected_indices = _select_background_indices(len(dataset), num_background)
    background_clips: List[Tensor] = []
    background_paths: List[Path] = []

    try:
        for index in selected_indices:
            clip_tensor, _ = dataset[index]
            background_clips.append(clip_tensor)
            background_paths.append(dataset.get_video_path(index))
    finally:
        dataset.close()

    return torch.stack(background_clips, dim=0), background_paths


class SemanticShapTarget(nn.Module):
    """Wrap the semantic model so SHAP sees a single scalar output per sample."""

    def __init__(self, model: SemanticDeepfakeDetector, target_spec: ShapTargetSpec) -> None:
        super().__init__()
        self.model = model
        self.target_spec = target_spec

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.model(x, return_dict=True)

        if self.target_spec.family == "class":
            target = outputs["logits"][:, self.target_spec.index]
        elif self.target_spec.family == "concept":
            concept_logits = outputs["concept_logits"][:, self.target_spec.index]
            target = concept_logits if self.target_spec.output_kind == "logit" else torch.sigmoid(concept_logits)
        else:  # pragma: no cover - Literal protects this path
            raise ValueError(f"Unsupported target family: {self.target_spec.family}")

        return target.unsqueeze(1)


def load_semantic_model(
    checkpoint_path: str | Path,
    device: torch.device,
    concept_vocabulary: Sequence[str] | None = None,
) -> SemanticDeepfakeDetector:
    """Instantiate the semantic model and load a checkpoint compatibly."""

    active_concepts = get_concept_vocabulary(concept_vocabulary)
    model = SemanticDeepfakeDetector(
        num_classes=Config.NUM_CLASSES,
        concept_vocabulary=active_concepts,
        extra_unsupervised_concepts=tuple(getattr(Config, "EXTRA_UNSUPERVISED_CONCEPTS", ())),
    ).to(device)
    load_compatible_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device, strict=False)
    model.eval()
    return model


def create_gradient_explainer(
    model: SemanticShapTarget,
    background_tensor: Tensor,
) -> object:
    """Create a GradientExplainer for the provided background tensor."""

    _require_shap()
    model.eval()
    background_tensor = background_tensor.detach()
    return shap.GradientExplainer(model, background_tensor)


def explain_clip_with_shap(
    explainer: object,
    clip_tensor: Tensor,
) -> np.ndarray:
    """Run SHAP for a single clip and normalize the output format."""

    _require_shap()
    shap_values = explainer.shap_values(clip_tensor)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values)


def aggregate_shap_values(shap_values: np.ndarray) -> np.ndarray:
    """Aggregate SHAP values from ``(B, C, T, H, W)`` to ``(T, H, W)``."""

    values = np.asarray(shap_values, dtype=np.float32)
    values = np.squeeze(values)

    if values.ndim == 5 and values.shape[0] == 1:
        values = values[0]

    if values.ndim != 4:
        raise ValueError(f"Expected SHAP values with 4 dimensions after squeezing, got {values.shape}")

    return np.mean(np.abs(values), axis=0)


def aggregate_signed_shap_values(shap_values: np.ndarray) -> np.ndarray:
    """Sum signed RGB attributions into a ``(T, H, W)`` contribution volume."""
    values = np.squeeze(np.asarray(shap_values, dtype=np.float32))
    if values.ndim == 5 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 4:
        raise ValueError(f"Expected SHAP values with 4 dimensions after squeezing, got {values.shape}")
    return np.sum(values, axis=0)


def explain_video_clip(
    checkpoint_path: str | Path,
    video_path: str | Path,
    background_root: str | Path,
    device: torch.device,
    target_spec: ShapTargetSpec,
    num_background: int = 8,
    frame_size: int = Config.IMAGE_SIZE,
    clip_length: int = Config.CLIP_LENGTH,
    align_faces: bool = Config.ALIGN_FACES,
) -> Tuple[List[np.ndarray], Tensor, np.ndarray, SemanticShapTarget, float]:
    """Convenience wrapper that prepares the clip, background, and explainer inputs."""

    semantic_model = load_semantic_model(checkpoint_path=checkpoint_path, device=device)
    wrapped_model = SemanticShapTarget(semantic_model, target_spec=target_spec).to(device)

    display_frames, clip_tensor = load_video_clip_for_shap(
        video_path=video_path,
        frame_size=frame_size,
        clip_length=clip_length,
        align_faces=align_faces,
    )

    background_tensor, _ = collect_background_tensor(
        background_root=background_root,
        num_background=num_background,
        frame_size=frame_size,
        clip_length=clip_length,
        align_faces=align_faces,
    )

    background_on_device = background_tensor.to(device)
    with torch.no_grad():
        baseline_value = float(wrapped_model(background_on_device).mean().item())
    explainer = create_gradient_explainer(wrapped_model, background_on_device)
    shap_values = explain_clip_with_shap(explainer, clip_tensor.to(device))
    return display_frames, clip_tensor, shap_values, wrapped_model, baseline_value


def explain_and_save_video(
    checkpoint_path: str | Path,
    video_path: str | Path,
    background_root: str | Path,
    output_path: str | Path,
    device: torch.device,
    target_spec: ShapTargetSpec,
    num_background: int = 8,
    frame_size: int = Config.IMAGE_SIZE,
    clip_length: int = Config.CLIP_LENGTH,
    alpha: float = 0.4,
    fps: float = 24.0,
    align_faces: bool = Config.ALIGN_FACES,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run SHAP, build frame-level overlays, and write a visualization video."""

    display_frames, clip_tensor, shap_values, wrapped_model, _ = explain_video_clip(
        checkpoint_path=checkpoint_path,
        video_path=video_path,
        background_root=background_root,
        device=device,
        target_spec=target_spec,
        num_background=num_background,
        frame_size=frame_size,
        clip_length=clip_length,
        align_faces=align_faces,
    )

    attribution_volume = aggregate_shap_values(shap_values)
    save_overlay_video(display_frames, attribution_volume, output_path=output_path, alpha=alpha, fps=fps)
    return attribution_volume, display_frames


def resolve_concept_index(concept_name: str) -> int:
    """Resolve a concept name to its fixed index."""

    concept_names = list(get_concept_vocabulary())
    if concept_name not in concept_names:
        raise KeyError(f"Unknown concept '{concept_name}'. Expected one of: {concept_names}")
    return concept_names.index(concept_name)
