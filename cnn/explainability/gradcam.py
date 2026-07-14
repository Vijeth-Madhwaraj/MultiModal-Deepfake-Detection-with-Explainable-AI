"""Head-wise Grad-CAM++ utilities for the semantic attention model.

The top-level ``gradcam.py`` script keeps the original final-layer Grad-CAM
workflow. This module focuses on the semantic model's four attention heads and
computes one Grad-CAM++ volume per head from the head feature maps captured
before learned fusion and before the concept projection layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from configs.config import Config
from inference import clip_to_tensor, read_video_frames, sample_clip
from video_dataset import FaceAlignmentProcessor, prepare_display_and_model_frame

from .head_analysis import summarize_head_outputs
from .visualization import normalize_attribution, overlay_heatmap, write_video


@dataclass(frozen=True)
class HeadwiseGradCAMResult:
    """Container for per-head attribution outputs."""

    heatmaps: Dict[int, np.ndarray]
    predicted_class: int
    confidence: float
    head_statistics: List[Mapping[str, float | int]]

    @property
    def predicted_label(self) -> str:
        return "Real" if self.predicted_class == 0 else "Fake"


def _gradcam_pp_for_head(activations: Tensor, gradients: Tensor) -> Tensor:
    """Compute a single Grad-CAM++ volume from ``(C, T, H, W)`` tensors."""

    gradients_2 = gradients.pow(2)
    gradients_3 = gradients.pow(3)
    denominator = 2.0 * gradients_2 + (activations * gradients_3).sum(dim=(1, 2, 3), keepdim=True)
    denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
    alphas = gradients_2 / (denominator + 1e-8)
    weights = (alphas * torch.relu(gradients)).sum(dim=(1, 2, 3), keepdim=True)
    return torch.relu((weights * activations).sum(dim=0))


def _resize_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    """Resize a ``(T, H, W)`` volume along time using linear interpolation."""

    if volume.shape[0] == target_depth:
        return volume

    source_depth, height, width = volume.shape
    source_positions = np.arange(source_depth, dtype=np.float32)
    target_positions = np.linspace(0, source_depth - 1, num=target_depth, dtype=np.float32)
    flattened = volume.reshape(source_depth, -1)
    resized = np.empty((target_depth, flattened.shape[1]), dtype=np.float32)
    for spatial_index in range(flattened.shape[1]):
        resized[:, spatial_index] = np.interp(target_positions, source_positions, flattened[:, spatial_index])
    return resized.reshape(target_depth, height, width)


class HeadwiseGradCAMPlusPlus:
    """Compute Grad-CAM++ independently for each semantic attention head."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def generate(self, input_tensor: Tensor, target_class: int | None = None) -> HeadwiseGradCAMResult:
        """Return one normalized heat volume per attention head."""

        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_tensor, return_dict=True)
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class_tensor = torch.max(probabilities, dim=1)
        predicted_class = int(predicted_class_tensor.item())

        if target_class is None:
            target_class = predicted_class

        head_feature_maps = outputs["head_feature_maps"]
        score = logits[:, target_class].sum()
        gradients = torch.autograd.grad(score, head_feature_maps, create_graph=True, retain_graph=True)[0]

        heatmaps: Dict[int, np.ndarray] = {}
        for head_index in range(head_feature_maps.shape[1]):
            cam = _gradcam_pp_for_head(
                activations=head_feature_maps[0, head_index],
                gradients=gradients[0, head_index],
            )
            heatmaps[head_index] = normalize_attribution(cam.detach().cpu().numpy())

        return HeadwiseGradCAMResult(
            heatmaps=heatmaps,
            predicted_class=predicted_class,
            confidence=float(confidence.item()),
            head_statistics=summarize_head_outputs(outputs),
        )


def prepare_clip_for_headwise_gradcam(
    video_path: str | Path,
    frame_size: int = Config.IMAGE_SIZE,
    clip_length: int = Config.CLIP_LENGTH,
    align_faces: bool = Config.ALIGN_FACES,
) -> Tuple[List[np.ndarray], Tensor]:
    """Return all source frames plus one fixed-length model input clip."""

    frames = read_video_frames(video_path, frame_size=None)
    clip_frames = sample_clip(frames, clip_length=clip_length)
    face_aligner = FaceAlignmentProcessor() if align_faces else None
    model_frames: List[np.ndarray] = []

    try:
        for frame_rgb in clip_frames:
            _, model_frame, _ = prepare_display_and_model_frame(
                frame_rgb,
                frame_size=frame_size,
                align_faces=align_faces,
                face_aligner=face_aligner,
            )
            model_frames.append(model_frame)
    finally:
        if face_aligner is not None:
            face_aligner.close()

    # Models based on Conv3d expect (N, C, T, H, W). ``clip_to_tensor``
    # returns one unbatched clip as (C, T, H, W), so add N=1 here.
    # The model sees the sampled center clip, while visualization interpolates
    # its temporal heat volume over every decoded source frame.
    return frames, clip_to_tensor(model_frames).unsqueeze(0)


def build_head_overlay_frames(
    frames: Sequence[np.ndarray],
    heatmaps: Mapping[int, np.ndarray],
    alpha: float = Config.HEATMAP_ALPHA,
) -> Dict[int, List[np.ndarray]]:
    """Build separate overlay frame sequences for every head."""

    overlays: Dict[int, List[np.ndarray]] = {}
    for head_index, volume in heatmaps.items():
        resized_volume = _resize_depth(np.asarray(volume, dtype=np.float32), target_depth=len(frames))
        head_frames: List[np.ndarray] = []
        for frame_rgb, heat_slice in zip(frames, resized_volume):
            height, width = frame_rgb.shape[:2]
            heat_slice = cv2.resize(heat_slice, (width, height), interpolation=cv2.INTER_LINEAR)
            head_frames.append(overlay_heatmap(frame_rgb, normalize_attribution(heat_slice), alpha=alpha))
        overlays[head_index] = head_frames
    return overlays


def _put_label(frame_rgb: np.ndarray, label: str) -> np.ndarray:
    frame = frame_rgb.copy()
    cv2.putText(
        frame,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def build_headwise_grid_frames(
    frames: Sequence[np.ndarray],
    heatmaps: Mapping[int, np.ndarray],
    predicted_label: str,
    confidence: float,
    alpha: float = Config.HEATMAP_ALPHA,
    head_names: Sequence[str] | None = None,
) -> List[np.ndarray]:
    """Create a grid with aligned frame plus four per-head overlays."""

    overlays = build_head_overlay_frames(frames, heatmaps, alpha=alpha)
    names = tuple(head_names or getattr(Config, "HEAD_NAMES", ()))
    grid_frames: List[np.ndarray] = []

    for frame_index, frame_rgb in enumerate(frames):
        cells = [_put_label(frame_rgb, f"Aligned | {predicted_label} {confidence:.2f}")]
        for head_index in sorted(overlays):
            name = names[head_index] if head_index < len(names) else f"head_{head_index + 1}"
            cells.append(_put_label(overlays[head_index][frame_index], name))

        height, width = cells[0].shape[:2]
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        while len(cells) < 6:
            cells.append(blank)

        top = np.concatenate(cells[:3], axis=1)
        bottom = np.concatenate(cells[3:6], axis=1)
        grid_frames.append(np.concatenate([top, bottom], axis=0))

    return grid_frames


def save_headwise_grid_video(
    frames: Sequence[np.ndarray],
    result: HeadwiseGradCAMResult,
    output_path: str | Path,
    alpha: float = Config.HEATMAP_ALPHA,
    fps: float = 24.0,
    head_names: Sequence[str] | None = None,
) -> None:
    """Write a video grid containing aligned frames and per-head overlays."""

    grid_frames = build_headwise_grid_frames(
        frames=frames,
        heatmaps=result.heatmaps,
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        alpha=alpha,
        head_names=head_names,
    )
    write_video(grid_frames, output_path=output_path, fps=fps)
