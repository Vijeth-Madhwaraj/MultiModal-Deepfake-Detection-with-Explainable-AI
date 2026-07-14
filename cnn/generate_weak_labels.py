"""Generate per-video weak labels for semantic deepfake concepts.

The script walks an input split such as ``data/train`` and writes:

    weak_label_dir/fake/sample_001_weak_labels.json
    weak_label_dir/fake/sample_001_weak_labels.npy
    weak_label_dir/fake/sample_001_ff_label.json

The weak-label vector order is:

    ["boundary_inconsistency", "eye_blink_irregularity"]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


CONCEPT_NAMES: Tuple[str, str] = ("boundary_inconsistency", "eye_blink_irregularity")
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
CLASS_NAMES: Tuple[str, str] = ("real", "fake")
MEDIAPIPE_LEFT_EYE = (33, 160, 158, 133, 153, 144)
MEDIAPIPE_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
FF_METHOD_NAMES: Tuple[str, ...] = (
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection",
)


@dataclass(frozen=True)
class VideoItem:
    path: Path
    class_name: str
    video_id: str


def clamp01(value: float) -> float:
    """Clamp a numeric score to [0, 1]."""

    if not math.isfinite(value):
        return 0.5
    return float(min(max(value, 0.0), 1.0))


def discover_videos(input_root: Path) -> List[VideoItem]:
    """Find videos below real/fake class folders."""

    items: List[VideoItem] = []
    for class_name in CLASS_NAMES:
        class_root = input_root / class_name
        if not class_root.exists():
            continue

        for video_path in sorted(class_root.rglob("*")):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                items.append(
                    VideoItem(
                        path=video_path,
                        class_name=class_name,
                        video_id=video_path.stem,
                    )
                )
    return items


def sampled_frames(video_path: Path, max_frames: int) -> Tuple[List[np.ndarray], float]:
    """Decode a contiguous center window so blink timing remains meaningful."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or not math.isfinite(fps):
            fps = 24.0

        if frame_count > max_frames:
            start_index = max((frame_count - max_frames) // 2, 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)

        frames: List[np.ndarray] = []
        while len(frames) < max_frames:
            success, frame_bgr = capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, fps


class FaceDetector:
    """Face detector using the project OpenCV DNN weights, with Haar fallback."""

    def __init__(self, project_root: Path) -> None:
        self.dnn_net: cv2.dnn_Net | None = None
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        prototxt = project_root / "weights" / "deploy.prototxt"
        model = project_root / "weights" / "res10_300x300_ssd_iter_140000.caffemodel"
        if prototxt.exists() and model.exists():
            # OpenCV wheels expose the Caffe reader under different names.
            # Some minimal builds omit it entirely; in that case the Haar
            # detector below remains a valid fallback.
            caffe_reader = getattr(cv2.dnn, "readNetFromCaffe", None)
            if caffe_reader is None:
                caffe_reader = getattr(cv2, "dnn_readNetFromCaffe", None)
            if caffe_reader is not None:
                try:
                    self.dnn_net = caffe_reader(str(prototxt), str(model))
                except cv2.error:
                    self.dnn_net = None

    def detect(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int] | None:
        if self.dnn_net is not None:
            bbox = self._detect_dnn(frame_rgb)
            if bbox is not None:
                return bbox
        return self._detect_haar(frame_rgb)

    def _detect_dnn(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int] | None:
        assert self.dnn_net is not None
        height, width = frame_rgb.shape[:2]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(frame_bgr, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        best_score = 0.5
        best_bbox: Tuple[int, int, int, int] | None = None
        for index in range(detections.shape[2]):
            score = float(detections[0, 0, index, 2])
            if score < best_score:
                continue

            x1 = int(detections[0, 0, index, 3] * width)
            y1 = int(detections[0, 0, index, 4] * height)
            x2 = int(detections[0, 0, index, 5] * width)
            y2 = int(detections[0, 0, index, 6] * height)
            left = max(min(x1, width - 1), 0)
            top = max(min(y1, height - 1), 0)
            right = max(min(x2, width), left + 1)
            bottom = max(min(y2, height), top + 1)
            best_score = score
            best_bbox = (left, top, right - left, bottom - top)
        return best_bbox

    def _detect_haar(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(width), int(height)


class BlinkScorer:
    """Eye-blink irregularity scorer using MediaPipe EAR when available."""

    def __init__(self) -> None:
        self.face_mesh = None
        self.backend_error: str | None = None
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        try:
            import mediapipe as mp  # type: ignore

            solutions = getattr(mp, "solutions", None)
            if solutions is None:
                from mediapipe.python.solutions import face_mesh  # type: ignore
            else:
                face_mesh = solutions.face_mesh
            self.face_mesh = face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            self.face_mesh = None

    @property
    def backend_name(self) -> str:
        return "mediapipe_face_mesh" if self.face_mesh is not None else "opencv_eye_fallback"

    def score(self, frames_rgb: Sequence[np.ndarray], fps: float, face_detector: FaceDetector) -> float:
        if self.face_mesh is not None:
            ear_values = self._ear_series_mediapipe(frames_rgb)
            if len(ear_values) >= max(8, len(frames_rgb) // 4):
                return self._score_ear_series(np.asarray(ear_values, dtype=np.float32), fps)

        return self._score_eye_detection_fallback(frames_rgb, fps, face_detector)

    def _ear_series_mediapipe(self, frames_rgb: Sequence[np.ndarray]) -> List[float]:
        assert self.face_mesh is not None
        ears: List[float] = []
        for frame_rgb in frames_rgb:
            result = self.face_mesh.process(frame_rgb)
            if not result.multi_face_landmarks:
                continue

            height, width = frame_rgb.shape[:2]
            landmarks = result.multi_face_landmarks[0].landmark
            left = self._eye_points(landmarks, MEDIAPIPE_LEFT_EYE, width, height)
            right = self._eye_points(landmarks, MEDIAPIPE_RIGHT_EYE, width, height)
            ears.append((self._ear(left) + self._ear(right)) / 2.0)
        return ears

    @staticmethod
    def _eye_points(landmarks: Sequence[object], indices: Sequence[int], width: int, height: int) -> np.ndarray:
        points = []
        for index in indices:
            landmark = landmarks[index]
            points.append((float(landmark.x) * width, float(landmark.y) * height))
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _ear(points: np.ndarray) -> float:
        horizontal = np.linalg.norm(points[0] - points[3])
        if horizontal <= 1e-6:
            return 0.0
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def _score_ear_series(self, ears: np.ndarray, fps: float) -> float:
        median_ear = float(np.median(ears))
        threshold = min(max(median_ear * 0.75, 0.15), 0.24)
        closed = ears < threshold

        blink_events = 0
        segment_lengths: List[int] = []
        index = 0
        while index < len(closed):
            if not closed[index]:
                index += 1
                continue
            start = index
            while index < len(closed) and closed[index]:
                index += 1
            length = index - start
            segment_lengths.append(length)
            if 1 <= length <= max(int(0.6 * fps), 1):
                blink_events += 1

        duration_minutes = max(len(ears) / fps / 60.0, 1e-6)
        blink_rate = blink_events / duration_minutes
        rate_penalty = min(abs(blink_rate - 18.0) / 30.0, 1.0)
        variability_penalty = min(float(np.std(ears)) / max(median_ear, 1e-6), 1.0)
        long_closure_penalty = min(sum(length > max(int(0.6 * fps), 1) for length in segment_lengths) / 3.0, 1.0)
        no_blink_penalty = 0.6 if blink_events == 0 and len(ears) / fps >= 4.0 else 0.0

        score = 0.45 * rate_penalty + 0.25 * variability_penalty + 0.20 * long_closure_penalty + 0.10 * no_blink_penalty
        return clamp01(score)

    def _score_eye_detection_fallback(
        self,
        frames_rgb: Sequence[np.ndarray],
        fps: float,
        face_detector: FaceDetector,
    ) -> float:
        detections: List[int] = []
        geometry: List[float] = []
        for frame_rgb in frames_rgb:
            bbox = face_detector.detect(frame_rgb)
            if bbox is None:
                continue
            x, y, width, height = bbox
            face = frame_rgb[y : y + height, x : x + width]
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))
            detections.append(1 if len(eyes) >= 1 else 0)
            if len(eyes) >= 1:
                # Eye-box height/width is a noisy but continuous openness proxy,
                # and avoids returning the same constant whenever eyes are seen.
                ratios = [float(box[3]) / max(float(box[2]), 1.0) for box in eyes[:2]]
                geometry.append(float(np.mean(ratios)))

        if len(detections) < max(8, len(frames_rgb) // 4):
            return 0.5

        eye_seen = np.asarray(detections, dtype=np.float32)
        missing = 1.0 - eye_seen
        transitions = float(np.mean(np.abs(np.diff(missing)))) if len(missing) > 1 else 0.0
        missing_ratio = float(np.mean(missing))
        geometry_variation = 0.0
        geometry_jitter = 0.0
        if len(geometry) >= 4:
            values = np.asarray(geometry, dtype=np.float32)
            geometry_variation = min(float(np.std(values)) / max(float(np.mean(values)), 1e-6) * 3.0, 1.0)
            geometry_jitter = min(float(np.mean(np.abs(np.diff(values)))) * 5.0, 1.0)
        score = (
            0.35 * min(transitions * 4.0, 1.0)
            + 0.20 * min(abs(missing_ratio - 0.1) / 0.5, 1.0)
            + 0.25 * geometry_variation
            + 0.20 * geometry_jitter
        )
        return clamp01(score)


class BoundaryScorer:
    """Boundary inconsistency scorer using a ring around the detected face."""

    def score(self, frames_rgb: Sequence[np.ndarray], face_detector: FaceDetector) -> float:
        frame_scores: List[float] = []
        for frame_rgb in frames_rgb:
            bbox = face_detector.detect(frame_rgb)
            if bbox is None:
                continue
            frame_scores.append(self._score_frame(frame_rgb, bbox))

        if not frame_scores:
            return 0.5
        return clamp01(float(np.median(frame_scores)))

    def _score_frame(self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        inner_mask, ring_mask, outer_ring_mask = self._masks(frame_rgb.shape[:2], bbox)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        inner_energy = float(np.mean(lap[inner_mask])) if np.any(inner_mask) else 0.0
        ring_energy = float(np.mean(lap[ring_mask])) if np.any(ring_mask) else 0.0
        edge_ratio_score = clamp01((ring_energy / (inner_energy + 1e-6) - 1.0) / 2.5)

        inner_color = np.mean(lab[inner_mask], axis=0) if np.any(inner_mask) else np.zeros(3, dtype=np.float32)
        outer_color = np.mean(lab[outer_ring_mask], axis=0) if np.any(outer_ring_mask) else inner_color
        color_delta = float(np.linalg.norm(inner_color - outer_color))
        color_score = clamp01(color_delta / 45.0)

        return clamp01(0.7 * edge_ratio_score + 0.3 * color_score)

    @staticmethod
    def _masks(
        frame_shape: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = frame_shape
        x, y, box_width, box_height = bbox
        center = (int(x + box_width / 2), int(y + box_height / 2))

        inner_axes = (max(int(box_width * 0.48), 1), max(int(box_height * 0.55), 1))
        mid_axes = (max(int(box_width * 0.58), 1), max(int(box_height * 0.67), 1))
        outer_axes = (max(int(box_width * 0.72), 1), max(int(box_height * 0.82), 1))

        inner = np.zeros((height, width), dtype=np.uint8)
        mid = np.zeros((height, width), dtype=np.uint8)
        outer = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(inner, center, inner_axes, 0, 0, 360, 1, -1)
        cv2.ellipse(mid, center, mid_axes, 0, 0, 360, 1, -1)
        cv2.ellipse(outer, center, outer_axes, 0, 0, 360, 1, -1)

        inner_mask = inner.astype(bool)
        ring_mask = (mid.astype(bool)) & (~inner_mask)
        outer_ring_mask = (outer.astype(bool)) & (~mid.astype(bool))
        return inner_mask, ring_mask, outer_ring_mask


def infer_ff_method_label(video_path: Path) -> str | None:
    """Infer a coarse FF++ method label from path parts when present."""

    lower_parts = {part.lower(): part for part in video_path.parts}
    for method in FF_METHOD_NAMES:
        if method.lower() in lower_parts:
            return method
    return None


def write_outputs(
    output_root: Path,
    item: VideoItem,
    labels: Dict[str, float],
    ff_method_label: str | None,
    overwrite: bool,
) -> None:
    """Write JSON, NPY, and optional FF label sidecar."""

    output_dir = output_root / item.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{item.video_id}_weak_labels.json"
    npy_path = output_dir / f"{item.video_id}_weak_labels.npy"
    ff_path = output_dir / f"{item.video_id}_ff_label.json"

    if not overwrite and (json_path.exists() or npy_path.exists()):
        return

    ordered = np.asarray([labels[name] for name in CONCEPT_NAMES], dtype=np.float32)
    json_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    np.save(npy_path, ordered)
    ff_path.write_text(json.dumps({"ff_method_label": ff_method_label}, indent=2), encoding="utf-8")


def process_video(
    item: VideoItem,
    face_detector: FaceDetector,
    blink_scorer: BlinkScorer,
    boundary_scorer: BoundaryScorer,
    max_frames: int,
) -> Dict[str, float]:
    """Compute both weak-label scores for one video."""

    frames, fps = sampled_frames(item.path, max_frames=max_frames)
    boundary_score = boundary_scorer.score(frames, face_detector)
    blink_score = blink_scorer.score(frames, fps, face_detector)
    return {
        "boundary_inconsistency": clamp01(boundary_score),
        "eye_blink_irregularity": clamp01(blink_score),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate weak semantic labels for GradCAM3D training videos.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/train"),
        help="Input root containing real/ and fake/ folders. Default: data/train",
    )
    parser.add_argument(
        "--weak-label-dir",
        type=Path,
        default=Path("weak_labels/train"),
        help="Output root for weak labels. Default: weak_labels/train",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=96,
        help="Maximum sampled frames per video. Larger is slower but more stable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing weak-label outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    input_root = args.input_root
    output_root = args.weak_label_dir

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    items = discover_videos(input_root)
    if not items:
        raise RuntimeError(f"No videos found under {input_root}/real or {input_root}/fake")

    face_detector = FaceDetector(project_root=project_root)
    blink_scorer = BlinkScorer()
    boundary_scorer = BoundaryScorer()

    print(f"Input root: {input_root}")
    print(f"Weak-label output root: {output_root}")
    print(f"Videos found: {len(items)}")
    print(f"Blink backend: {blink_scorer.backend_name}")
    if blink_scorer.backend_error is not None:
        print(f"MediaPipe unavailable reason: {blink_scorer.backend_error}")
    print(f"Concept order: {list(CONCEPT_NAMES)}")

    failures: List[Tuple[Path, str]] = []
    for item in tqdm(items, desc="Weak labels"):
        try:
            labels = process_video(
                item=item,
                face_detector=face_detector,
                blink_scorer=blink_scorer,
                boundary_scorer=boundary_scorer,
                max_frames=args.max_frames,
            )
            ff_method_label = infer_ff_method_label(item.path)
            write_outputs(
                output_root=output_root,
                item=item,
                labels=labels,
                ff_method_label=ff_method_label,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failures.append((item.path, str(exc)))

    print(f"Completed: {len(items) - len(failures)}")
    print(f"Failed: {len(failures)}")
    if failures:
        failure_log = output_root / "weak_label_failures.json"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"video_path": str(path), "error": error} for path, error in failures]
        failure_log.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Failure log: {failure_log}")


if __name__ == "__main__":
    main()
