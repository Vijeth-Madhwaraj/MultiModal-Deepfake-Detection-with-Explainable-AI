"""Command-line entrypoint for SHAP explanations on the semantic deepfake model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from configs.config import Config
from explainability.shap_utils import (
    ShapTargetSpec,
    aggregate_signed_shap_values,
    aggregate_shap_values,
    explain_video_clip,
    resolve_concept_index,
)
from explainability.visualization import save_signed_overlay_video


def read_video_fps(video_path: str | Path, fallback: float = 24.0) -> float:
    """Read the source FPS so the SHAP output keeps the original duration."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
    finally:
        capture.release()
    return fps if np.isfinite(fps) and fps > 0.0 else fallback


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for SHAP video explanations."""

    parser = argparse.ArgumentParser(description="Generate SHAP explanations for one video clip.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(Path(Config.CHECKPOINT_DIR) / Config.CHECKPOINT_NAME),
        help="Path to the trained semantic checkpoint.",
    )
    parser.add_argument(
        "--background-root",
        type=str,
        default=Config.TRAIN_DIR,
        help="Root directory used to sample background clips, usually data/train.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(Path(Config.OUTPUT_DIR) / "shap_video.mp4"),
        help="Path where the SHAP overlay video will be saved.",
    )
    parser.add_argument(
        "--npy-path",
        type=str,
        default=str(Path(Config.OUTPUT_DIR) / "shap_attribution.npy"),
        help="Path where the aggregated attribution volume will be saved as .npy.",
    )
    parser.add_argument("--raw-npy-path", type=str, default=str(Path(Config.OUTPUT_DIR) / "shap_raw_signed.npy"))
    parser.add_argument("--report-path", type=str, default=str(Path(Config.OUTPUT_DIR) / "shap_report.json"))
    parser.add_argument(
        "--target-family",
        type=str,
        choices=("class", "concept"),
        default="class",
        help="Explain the final class logits or the concept outputs.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=1,
        help="Class index for class explanations. Ignored for concept-name mode.",
    )
    parser.add_argument(
        "--concept-name",
        type=str,
        default=None,
        help="Concept name to explain. Use this instead of --target-index for concept explanations.",
    )
    parser.add_argument(
        "--output-kind",
        type=str,
        choices=("logit", "score"),
        default="logit",
        help="Explain raw logits or sigmoid concept scores.",
    )
    parser.add_argument("--num-background", type=int, default=8, help="Number of background clips to use.")
    parser.add_argument("--frame-size", type=int, default=Config.IMAGE_SIZE, help="Frame size used by the model.")
    parser.add_argument("--clip-length", type=int, default=Config.CLIP_LENGTH, help="Clip length used by the model.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha for visualization.")
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="Device to run SHAP on.")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Deterministic seed for background sampling.")
    parser.add_argument(
        "--align-faces",
        action="store_true",
        default=Config.ALIGN_FACES,
        help="Use landmark-based eye alignment before face crop/resize when landmarks are available.",
    )
    return parser


def main() -> None:
    """Run SHAP on one video clip and save the attribution visualization."""

    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if args.target_family == "concept":
        target_index = resolve_concept_index(args.concept_name) if args.concept_name is not None else args.target_index
    else:
        target_index = args.target_index

    target_spec = ShapTargetSpec(family=args.target_family, index=target_index, output_kind=args.output_kind)

    Path(args.npy_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.raw_npy_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    # ``explain_video_clip`` returns both the prepared clip and the SHAP values.
    display_frames, clip_tensor, shap_values, wrapped_model, baseline_value = explain_video_clip(
        checkpoint_path=args.checkpoint_path,
        video_path=args.video_path,
        background_root=args.background_root,
        device=device,
        target_spec=target_spec,
        num_background=args.num_background,
        frame_size=args.frame_size,
        clip_length=args.clip_length,
        align_faces=args.align_faces,
    )

    attribution_volume = aggregate_shap_values(shap_values)
    signed_volume = aggregate_signed_shap_values(shap_values)
    save_signed_overlay_video(
        display_frames,
        signed_volume,
        output_path=args.output_path,
        target_name=target_spec.display_name,
        alpha=args.alpha,
        fps=read_video_fps(args.video_path),
    )
    np.save(args.npy_path, attribution_volume)
    np.save(args.raw_npy_path, np.asarray(shap_values, dtype=np.float32))

    with torch.no_grad():
        target_value = wrapped_model(clip_tensor.to(device)).item()

    positive = float(signed_volume[signed_volume > 0].sum())
    negative = float(signed_volume[signed_volume < 0].sum())
    residual = float(target_value - (baseline_value + positive + negative))
    report = {
        "target_family": target_spec.family,
        "target_name": target_spec.display_name,
        "reference_value": baseline_value,
        "positive_support": positive,
        "negative_opposition": negative,
        "approximation_residual": residual,
        "final_target_value": float(target_value),
        "frame_contributions": signed_volume.sum(axis=(1, 2)).astype(float).tolist(),
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Target family: {target_spec.family}")
    print(f"Target name: {target_spec.display_name}")
    print(f"Target value: {target_value:.6f}")
    print(f"Reference/background value: {baseline_value:+.6f}")
    print(f"Positive SHAP support:       {positive:+.6f}")
    print(f"Negative SHAP opposition:    {negative:+.6f}")
    print(f"Approximation residual:      {residual:+.6f}")
    print("---------------------------------------------")
    print(f"Final {target_spec.display_name} value:       {target_value:+.6f}")
    print(f"Saved attribution volume to: {args.npy_path}")
    print(f"Saved raw signed SHAP values to: {args.raw_npy_path}")
    print(f"Saved contribution report to: {args.report_path}")
    print(f"Saved SHAP visualization video to: {args.output_path}")


if __name__ == "__main__":
    main()
