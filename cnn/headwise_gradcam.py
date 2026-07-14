"""CLI for per-head Grad-CAM++ on the semantic attention model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from configs.config import Config
from explainability.gradcam import (
    HeadwiseGradCAMPlusPlus,
    prepare_clip_for_headwise_gradcam,
    save_headwise_grid_video,
)
from model import SemanticDeepfakeDetector, load_compatible_checkpoint


def read_video_fps(video_path: str | Path, fallback: float = 24.0) -> float:
    """Read the source FPS so the explanation keeps the original duration."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
    finally:
        capture.release()
    return fps if fps > 0.0 else fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-head Grad-CAM++ overlays for one input clip.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(Path(Config.CHECKPOINT_DIR) / Config.CHECKPOINT_NAME),
        help="Path to the trained semantic checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(Path(Config.OUTPUT_DIR) / "headwise_gradcam_grid.mp4"),
        help="Path where the head-wise Grad-CAM++ grid video will be saved.",
    )
    parser.add_argument("--target-class", type=int, default=None, help="Class index to explain. Defaults to prediction.")
    parser.add_argument("--frame-size", type=int, default=Config.IMAGE_SIZE)
    parser.add_argument("--clip-length", type=int, default=Config.CLIP_LENGTH)
    parser.add_argument("--alpha", type=float, default=Config.HEATMAP_ALPHA)
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    parser.add_argument(
        "--align-faces",
        action="store_true",
        default=Config.ALIGN_FACES,
        help="Use landmark-based eye alignment before face crop/resize when landmarks are available.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    model = SemanticDeepfakeDetector(
        num_classes=Config.NUM_CLASSES,
        concept_vocabulary=Config.CONCEPT_NAMES,
        extra_unsupervised_concepts=Config.EXTRA_UNSUPERVISED_CONCEPTS,
    ).to(device)
    load_compatible_checkpoint(model=model, checkpoint_path=args.checkpoint_path, device=device, strict=False)
    model.eval()

    display_frames, clip_tensor = prepare_clip_for_headwise_gradcam(
        video_path=args.video_path,
        frame_size=args.frame_size,
        clip_length=args.clip_length,
        align_faces=args.align_faces,
    )
    result = HeadwiseGradCAMPlusPlus(model).generate(clip_tensor.to(device), target_class=args.target_class)
    save_headwise_grid_video(
        frames=display_frames,
        result=result,
        output_path=args.output_path,
        alpha=args.alpha,
        fps=read_video_fps(args.video_path),
        head_names=Config.HEAD_NAMES,
    )

    print(f"Prediction: {result.predicted_label}")
    print(f"Confidence: {result.confidence:.4f}")
    print("Head statistics:")
    for summary in result.head_statistics:
        print(
            f"  head_{int(summary['head_index']) + 1}: "
            f"weight={summary['fusion_weight']:.4f}, "
            f"energy={summary['feature_mean_abs']:.4f}, "
            f"concentration={summary['attention_concentration']:.4f}"
        )
    print(f"Saved head-wise Grad-CAM++ grid video to: {args.output_path}")


if __name__ == "__main__":
    main()
