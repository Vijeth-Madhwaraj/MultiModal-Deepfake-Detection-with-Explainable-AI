"""Run every deepfake signal on one video and write a unified JSON report.

The report combines:

* semantic 3D CNN classification (Real=0, Fake=1) and confidence;
* boundary-inconsistency and eye-blink-irregularity heuristic scores;
* audio-versus-video transcript metrics (WER, word match, character and
  semantic similarity); and
* Random Forest audio classification and probabilities.

Heavy dependencies are imported lazily so ``--help`` remains usable even when
the machine's ML environment has not been activated.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CNN_DIR = PROJECT_ROOT / "cnn"
RANDOM_FOREST_DIR = PROJECT_ROOT / "random_forest"
TRANSCRIPT_DIR = PROJECT_ROOT / "transcript"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
LABEL_IDS = {"real": 0, "fake": 1}


def _prepend_import_path(path: Path) -> None:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _label_payload(label: str, confidence: float) -> Dict[str, Any]:
    normalized = label.strip().lower()
    if normalized not in LABEL_IDS:
        raise ValueError(f"Unexpected classification label: {label!r}")
    return {
        "label": normalized.upper(),
        "label_id": LABEL_IDS[normalized],
        "confidence": round(float(confidence), 6),
    }


def run_cnn_3d(
    video_path: Path,
    checkpoint_path: Path,
    device_name: str,
    align_faces: bool,
) -> Dict[str, Any]:
    """Run the semantic 3D CNN and return its predicted-class confidence."""

    _prepend_import_path(CNN_DIR)
    import torch
    from configs.config import Config
    from inference import load_checkpoint, predict_video
    from model import SemanticDeepfakeDetector

    resolved_device = device_name
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the 3D CNN but is unavailable.")

    device = torch.device(resolved_device)
    model = SemanticDeepfakeDetector(
        num_classes=Config.NUM_CLASSES,
        concept_vocabulary=Config.CONCEPT_NAMES,
        extra_unsupervised_concepts=Config.EXTRA_UNSUPERVISED_CONCEPTS,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    label, confidence = predict_video(
        model=model,
        video_path=video_path,
        device=device,
        align_faces=align_faces,
    )
    result = _label_payload(label, confidence)
    result.update({"device": str(device), "checkpoint": str(checkpoint_path)})

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_weak_label_heuristics(video_path: Path, max_frames: int) -> Dict[str, Any]:
    """Compute boundary and blink irregularity scores in the [0, 1] range."""

    _prepend_import_path(CNN_DIR)
    from generate_weak_labels import (
        BlinkScorer,
        BoundaryScorer,
        FaceDetector,
        VideoItem,
        process_video,
    )

    face_detector = FaceDetector(CNN_DIR)
    blink_scorer = BlinkScorer()
    item = VideoItem(path=video_path, class_name="unknown", video_id=video_path.stem)
    try:
        scores = process_video(
            item=item,
            face_detector=face_detector,
            blink_scorer=blink_scorer,
            boundary_scorer=BoundaryScorer(),
            max_frames=max_frames,
        )
        return {
            "boundary_inconsistency": round(float(scores["boundary_inconsistency"]), 6),
            "eye_blink_irregularity": round(float(scores["eye_blink_irregularity"]), 6),
            "score_range": [0.0, 1.0],
            "eye_backend": blink_scorer.backend_name,
            "frames_limit": max_frames,
        }
    finally:
        face_mesh = getattr(blink_scorer, "face_mesh", None)
        if face_mesh is not None and hasattr(face_mesh, "close"):
            face_mesh.close()


def run_random_forest(video_path: Path) -> Dict[str, Any]:
    """Run the audio-feature Random Forest classifier."""

    _prepend_import_path(RANDOM_FOREST_DIR)
    from predict_video import predict_video

    label, fake_probability = predict_video(video_path)
    fake_probability = float(fake_probability)
    real_probability = 1.0 - fake_probability
    confidence = fake_probability if label.strip().lower() == "fake" else real_probability
    result = _label_payload(label, confidence)
    result.update(
        {
            "fake_probability": round(fake_probability, 6),
            "real_probability": round(real_probability, 6),
        }
    )
    return result


def run_transcript_comparison(
    video_path: Path,
    gpu_index: int,
    detector: str,
    semantic_similarity: bool,
) -> Dict[str, Any]:
    """Generate audio/video transcripts and their comparison metrics.

    Transcript inference is isolated in a child process because the audio and
    visual speech-recognition checkpoints are large. The child exits after the
    metrics are produced, immediately releasing that memory.
    """

    with tempfile.TemporaryDirectory(prefix="gradcam3d_transcript_") as temp_dir:
        result_path = Path(temp_dir) / "transcript_result.json"
        command = [
            sys.executable,
            "infer.py",
            "compare_transcripts=true",
            f"data_filename={video_path.as_posix()}",
            f"gpu_idx={gpu_index}",
            f"detector={detector}",
            f"semantic_similarity={str(semantic_similarity).lower()}",
            f"dst_filename={result_path.as_posix()}",
        ]
        child_environment = os.environ.copy()
        child_environment["HYDRA_FULL_ERROR"] = "1"
        completed = subprocess.run(
            command,
            cwd=TRANSCRIPT_DIR,
            text=True,
            capture_output=True,
            check=False,
            env=child_environment,
        )
        if completed.returncode != 0:
            output_parts = []
            if completed.stdout.strip():
                output_parts.append(f"stdout:\n{completed.stdout.strip()}")
            if completed.stderr.strip():
                output_parts.append(f"stderr:\n{completed.stderr.strip()}")
            details = "\n\n".join(output_parts) or "The child process produced no diagnostic output."
            if len(details) > 8000:
                details = details[-8000:]
            raise RuntimeError(
                f"Transcript inference exited with code {completed.returncode}.\n{details}"
            )
        if not result_path.exists():
            raise RuntimeError("Transcript inference completed without producing its JSON output.")
        raw_result = json.loads(result_path.read_text(encoding="utf-8"))

    metrics = raw_result.get("metrics", {})
    result: Dict[str, Any] = {
        "audio_transcript": raw_result.get("audio_transcript", ""),
        "video_transcript": raw_result.get("video_transcript", ""),
        "wer": metrics.get("wer"),
        "word_match_rate": metrics.get("word_match"),
        "character_similarity": metrics.get("character_similarity"),
        "semantic_similarity": metrics.get("semantic_similarity"),
        "percentage_metric_range": [0.0, 100.0],
    }
    return result


def _run_stage(
    report: Dict[str, Any],
    stage_name: str,
    operation: Callable[[], Dict[str, Any]],
) -> None:
    print(f"Running {stage_name}...", flush=True)
    try:
        report["results"][stage_name] = operation()
        print(f"Completed {stage_name}.", flush=True)
    except Exception as error:  # Keep the other independent stages running.
        report["errors"][stage_name] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        print(f"Failed {stage_name}: {error}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CNN, heuristic, transcript, and Random Forest analysis on one video."
    )
    parser.add_argument("video", type=Path, help="Video file to analyze.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Default: outputs/full_analysis_<video>.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="CNN device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--align-faces", action="store_true")
    parser.add_argument("--max-heuristic-frames", type=int, default=96)
    parser.add_argument("--transcript-gpu-index", type=int, default=0)
    parser.add_argument("--transcript-detector", choices=("mediapipe", "retinaface"), default="mediapipe")
    parser.add_argument(
        "--no-semantic-similarity",
        action="store_true",
        help="Skip sentence-transformer semantic similarity.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any component fails (JSON is still written).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video_path = args.video.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if args.max_heuristic_frames <= 0:
        raise ValueError("--max-heuristic-frames must be greater than zero.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else PROJECT_ROOT / "outputs" / f"full_analysis_{video_path.stem}.json"
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_video": str(video_path),
        "label_encoding": {"REAL": 0, "FAKE": 1},
        "results": {},
        "errors": {},
    }

    _run_stage(
        report,
        "cnn_3d",
        lambda: run_cnn_3d(video_path, checkpoint_path, args.device, args.align_faces),
    )
    _run_stage(
        report,
        "weak_label_heuristics",
        lambda: run_weak_label_heuristics(video_path, args.max_heuristic_frames),
    )
    _run_stage(report, "random_forest", lambda: run_random_forest(video_path))
    _run_stage(
        report,
        "transcript_comparison",
        lambda: run_transcript_comparison(
            video_path=video_path,
            gpu_index=args.transcript_gpu_index,
            detector=args.transcript_detector,
            semantic_similarity=not args.no_semantic_similarity,
        ),
    )

    report["status"] = "complete" if not report["errors"] else "partial"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON report: {output_path}")
    if args.strict and report["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
