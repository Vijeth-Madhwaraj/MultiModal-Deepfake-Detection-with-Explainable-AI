"""Manual Grad-CAM for the custom 3D CNN deepfake classifier.

This module implements Grad-CAM from scratch for a 3D CNN operating on video
clips. It extracts activations and gradients from the final convolutional
layer, computes class-specific importance weights, builds a 3D heat volume,
and expands the sampled clip's heat volume over every frame of the input video.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from configs.config import Config
from model import SemanticDeepfakeDetector, load_compatible_checkpoint
from video_dataset import (
    FACE_DETECTOR,
    FaceAlignmentProcessor,
    detect_face_bbox,
    prepare_display_and_model_frame as prepare_aligned_display_and_model_frame,
)


def read_video_frames(video_path: str | Path, frame_size: Optional[int] = None) -> List[np.ndarray]:
    """Read a video file into a list of RGB frames, optionally resizing them."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_size is not None:
                frame_rgb = cv2.resize(frame_rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")

    return frames


def read_video_fps(video_path: str | Path, fallback: float = 24.0) -> float:
    """Return the source frame rate, falling back when metadata is unavailable."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
    finally:
        capture.release()
    return fps if np.isfinite(fps) and fps > 0.0 else fallback


def sample_clip(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Select a deterministic center clip from a video."""

    if len(frames) >= clip_length:
        start_index = max((len(frames) - clip_length) // 2, 0)
        clip = frames[start_index : start_index + clip_length]
    else:
        clip = list(frames)
        last_frame = clip[-1]
        while len(clip) < clip_length:
            clip.append(last_frame)

    return clip[:clip_length]


def frames_to_tensor(frames: List[np.ndarray]) -> Tensor:
    """Convert preprocessed RGB frames to a float tensor with shape (1, 3, T, H, W)."""

    clip_array = np.asarray(frames, dtype=np.float32)
    clip_array = np.transpose(clip_array, (3, 0, 1, 2))
    return torch.from_numpy(clip_array).unsqueeze(0)


def prepare_display_and_model_frame(
    frame_rgb: np.ndarray,
    frame_size: int,
    align_faces: bool = Config.ALIGN_FACES,
    face_aligner: FaceAlignmentProcessor | None = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Prepare one frame for display overlay and model input using the same crop."""

    return prepare_aligned_display_and_model_frame(
        frame_rgb,
        frame_size=frame_size,
        align_faces=align_faces,
        face_aligner=face_aligner,
    )


def resize_frames(frames: List[np.ndarray], frame_size: int) -> List[np.ndarray]:
    """Resize a list of RGB frames to a square size."""

    return [cv2.resize(frame_rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR) for frame_rgb in frames]


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize a heatmap to the [0, 1] range."""

    heatmap = np.maximum(heatmap, 0).astype(np.float32, copy=False)
    min_value = float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    if max_value > min_value:
        heatmap = (heatmap - min_value) / (max_value - min_value)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    return np.clip(heatmap, 0.0, 1.0)


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Convert a normalized heatmap to a BGR color heatmap."""

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay a heatmap on top of an RGB frame and return an RGB frame."""

    heatmap_bgr = apply_colormap(heatmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(frame_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def build_face_mask(frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int], margin: float = 0.18) -> np.ndarray:
    """Create a soft elliptical mask around the detected face region."""

    height, width = frame_shape[:2]
    x, y, box_width, box_height = bbox

    x_margin = int(box_width * margin)
    y_margin = int(box_height * margin)

    left = max(x - x_margin, 0)
    top = max(y - y_margin, 0)
    right = min(x + box_width + x_margin, width)
    bottom = min(y + box_height + y_margin, height)

    mask = np.zeros((height, width), dtype=np.float32)
    axis_x = max((right - left) // 2, 1)
    axis_y = max((bottom - top) // 2, 1)
    center_x = left + axis_x
    center_y = top + axis_y

    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=axis_x * 0.12 + 1.0, sigmaY=axis_y * 0.12 + 1.0)
    return np.clip(mask, 0.0, 1.0)


def apply_face_focus(frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Restrict the heatmap to the face region when a face can be detected."""

    bbox = detect_face_bbox(frame_rgb)
    if bbox is None:
        return heatmap

    face_mask = build_face_mask(frame_rgb.shape, bbox)
    focused_heatmap = heatmap * face_mask
    return normalize_heatmap(focused_heatmap)


def write_video(frames_rgb: List[np.ndarray], output_path: str | Path, fps: float = 24.0) -> None:
    """Write a list of RGB frames to an MP4 file."""

    if not frames_rgb:
        raise ValueError("No frames were provided for video writing.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for frame_rgb in frames_rgb:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    """Load either a full training checkpoint or a plain state dict."""

    load_compatible_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device, strict=False)


class GradCAM3D:
    """Manual Grad-CAM implementation for a single 3D CNN target layer."""

    def __init__(self, model: nn.Module, target_layer: str) -> None:
        self.model = model
        self.target_layer = self._resolve_target_layer(target_layer)
        self.activations: Optional[Tensor] = None
        self.last_activation_shape: Optional[Tuple[int, ...]] = None
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _resolve_target_layer(self, target_layer: str) -> nn.Module:
        """Resolve a dotted module path such as 'conv3' on the model."""

        module: nn.Module = self.model
        for name in target_layer.split("."):
            if not hasattr(module, name):
                raise AttributeError(f"Target layer '{target_layer}' not found on model.")
            module = getattr(module, name)
            if not isinstance(module, nn.Module):
                raise TypeError(f"Attribute '{name}' is not a module.")
        return module

    def _forward_hook(self, module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        """Store the target layer activations during the forward pass."""

        self.activations = output

    def remove_hooks(self) -> None:
        """Remove registered hooks to avoid leaking references."""

        self._forward_handle.remove()

    def generate(
        self,
        input_tensor: Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate a Grad-CAM heat volume for the given input tensor.

        Returns
        -------
        heat_volume:
            A numpy array with shape (T, H, W) normalized to [0, 1].
        predicted_class:
            The model's predicted class index.
        confidence:
            The softmax confidence for the predicted class.
        """

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class_tensor = torch.max(probabilities, dim=1)
        predicted_class = int(predicted_class_tensor.item())

        if target_class is None:
            target_class = predicted_class

        score = logits[:, target_class].sum()

        if self.activations is None:
            raise RuntimeError("Failed to capture activations from the target layer.")

        activations = self.activations[0]
        self.last_activation_shape = tuple(self.activations.shape)
        gradients = torch.autograd.grad(score, self.activations, create_graph=True, retain_graph=True)[0][0]

        gradients_2 = gradients.pow(2)
        gradients_3 = gradients.pow(3)

        denominator = 2.0 * gradients_2 + (activations * gradients_3).sum(dim=(1, 2, 3), keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))

        alphas = gradients_2 / (denominator + 1e-8)
        positive_gradients = torch.relu(gradients)
        weights = (alphas * positive_gradients).sum(dim=(1, 2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)

        heat_volume = cam.detach().cpu().numpy()
        heat_volume = normalize_heatmap(heat_volume)
        return heat_volume, predicted_class, float(confidence.item())


def upsample_heatmap_slice(heat_slice: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upsample a single heatmap slice to the frame size."""

    target_height, target_width = target_shape
    return cv2.resize(heat_slice, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def resize_heat_volume(heat_volume: np.ndarray, target_depth: int) -> np.ndarray:
    """Resize a 3D heat volume so its temporal depth matches the clip length."""

    if heat_volume.shape[0] == target_depth:
        return heat_volume

    if heat_volume.ndim != 3:
        raise ValueError(f"Expected heat volume with shape (T, H, W), got {heat_volume.shape}")

    source_depth, height, width = heat_volume.shape
    source_positions = np.arange(source_depth, dtype=np.float32)
    target_positions = np.linspace(0, source_depth - 1, num=target_depth, dtype=np.float32)

    flattened = heat_volume.reshape(source_depth, -1).astype(np.float32)
    resized_flat = np.empty((target_depth, flattened.shape[1]), dtype=np.float32)
    for spatial_idx in range(flattened.shape[1]):
        resized_flat[:, spatial_idx] = np.interp(target_positions, source_positions, flattened[:, spatial_idx])

    resized = resized_flat.reshape(target_depth, height, width)
    return normalize_heatmap(resized)


def build_overlay_frames(
    frames: List[np.ndarray],
    heat_volume: np.ndarray,
    alpha: float,
) -> List[np.ndarray]:
    """Overlay the Grad-CAM heat volume on every frame in the clip."""

    heat_volume = resize_heat_volume(heat_volume, target_depth=len(frames))

    output_frames: List[np.ndarray] = []

    for frame_rgb, heat_slice in zip(frames, heat_volume):
        heat_slice = upsample_heatmap_slice(heat_slice, target_shape=frame_rgb.shape[:2])
        heat_slice = normalize_heatmap(heat_slice)
        heat_slice = apply_face_focus(frame_rgb, heat_slice)
        output_frames.append(overlay_heatmap(frame_rgb, heat_slice, alpha=alpha))

    return output_frames


def run_gradcam(
    model: nn.Module,
    video_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    target_layer: str = Config.TARGET_LAYER,
    alpha: float = Config.HEATMAP_ALPHA,
    clip_length: int = Config.CLIP_LENGTH,
    frame_size: int = Config.IMAGE_SIZE,
    align_faces: bool = Config.ALIGN_FACES,
) -> Tuple[str, float]:
    """Generate a Grad-CAM overlay video and return the prediction summary."""

    frames = read_video_frames(video_path, frame_size=None)
    clip_frames = sample_clip(frames, clip_length=clip_length)
    print(f"Face detector backend: {FACE_DETECTOR.backend_name}")
    display_frames: List[np.ndarray] = []
    model_frames: List[np.ndarray] = []
    face_crops_used = 0
    face_aligner = FaceAlignmentProcessor() if align_faces else None

    try:
        for frame_rgb in clip_frames:
            display_frame, model_frame, used_face_crop = prepare_display_and_model_frame(
                frame_rgb,
                frame_size,
                align_faces=align_faces,
                face_aligner=face_aligner,
            )
            display_frames.append(display_frame)
            model_frames.append(model_frame)
            face_crops_used += int(used_face_crop)
    finally:
        if face_aligner is not None:
            face_aligner.close()

    fallback_count = len(clip_frames) - face_crops_used
    print(
        f"Model-preprocess face detection for sampled clip: "
        f"detected={face_crops_used}/{len(clip_frames)}, fallback_full_frame={fallback_count}/{len(clip_frames)}"
    )

    input_tensor = frames_to_tensor(model_frames).to(device)

    gradcam = GradCAM3D(model=model, target_layer=target_layer)
    try:
        heat_volume, predicted_class, confidence = gradcam.generate(input_tensor)
        if gradcam.last_activation_shape is not None:
            print(
                f"Grad-CAM hook layer '{target_layer}' activation shape: "
                f"{gradcam.last_activation_shape}"
            )
    finally:
        gradcam.remove_hooks()

    # The model operates on a fixed-length center clip, but the visualization
    # should retain the complete input video. Interpolate the temporal CAM over
    # all decoded frames and overlay it at the original video resolution.
    overlay_frames = build_overlay_frames(frames, heat_volume, alpha=alpha)
    write_video(overlay_frames, output_path=output_path, fps=read_video_fps(video_path))

    label = "Real" if predicted_class == 0 else "Fake"
    return label, confidence


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for Grad-CAM generation."""

    parser = argparse.ArgumentParser(description="Generate a Grad-CAM overlay video for one input clip.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(Path(Config.CHECKPOINT_DIR) / Config.CHECKPOINT_NAME),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(Path(Config.OUTPUT_DIR) / "gradcam_video.mp4"),
        help="Path where the Grad-CAM video will be saved.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=Config.TARGET_LAYER,
        help="Model layer to use for Grad-CAM, for example conv3.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=Config.DEVICE,
        help="Device to run inference on, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--align-faces",
        action="store_true",
        default=Config.ALIGN_FACES,
        help="Use landmark-based eye alignment before face crop/resize when landmarks are available.",
    )
    return parser


def main() -> None:
    """Run Grad-CAM on a single video and save the overlay video."""

    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SemanticDeepfakeDetector(
        num_classes=Config.NUM_CLASSES,
        concept_vocabulary=Config.CONCEPT_NAMES,
        extra_unsupervised_concepts=Config.EXTRA_UNSUPERVISED_CONCEPTS,
    ).to(device)
    load_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    label, confidence = run_gradcam(
        model=model,
        video_path=args.video_path,
        output_path=args.output_path,
        device=device,
        target_layer=args.target_layer,
        align_faces=args.align_faces,
    )

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Saved Grad-CAM video to: {args.output_path}")


if __name__ == "__main__":
    main()
