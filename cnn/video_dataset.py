"""Video dataset utilities for binary deepfake classification.

This module provides a lightweight, self-contained PyTorch Dataset that reads
videos from a folder structure like:

    root_dir/
        real/
        fake/

The dataset:
- reads videos with OpenCV,
- samples random clips during training,
- returns RGB clips as float32 tensors in (C, T, H, W) format,
- normalizes pixel values to the [0, 1] range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_CLASS_NAMES: Tuple[str, str] = ("real", "fake")
DEFAULT_CONCEPT_NAMES: Tuple[str, str] = (
    "boundary_inconsistency",
    "eye_blink_irregularity",
)
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DNN_CONFIDENCE_THRESHOLD = 0.5
LEFT_EYE_LANDMARKS: Tuple[int, ...] = (33, 133, 160, 159, 157, 154, 145, 144)
RIGHT_EYE_LANDMARKS: Tuple[int, ...] = (362, 263, 387, 386, 385, 380, 374, 373)


def _resolve_open_cv_dnn_model_paths() -> Tuple[Path, Path] | None:
    """Return available OpenCV DNN face model paths if present in common locations."""

    base_data_dir = Path(cv2.data.haarcascades).parent
    project_weights_dir = Path(__file__).parent / "weights"
    prototxt_candidates = (
        project_weights_dir / "deploy.prototxt",
        base_data_dir / "dnn" / "deploy.prototxt",
        base_data_dir / "deploy.prototxt",
    )
    model_candidates = (
        project_weights_dir / "res10_300x300_ssd_iter_140000.caffemodel",
        base_data_dir / "dnn" / "res10_300x300_ssd_iter_140000_fp16.caffemodel",
        base_data_dir / "res10_300x300_ssd_iter_140000.caffemodel",
        base_data_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel",
    )

    prototxt_path = next((path for path in prototxt_candidates if path.exists()), None)
    model_path = next((path for path in model_candidates if path.exists()), None)

    if prototxt_path is None or model_path is None:
        return None
    return prototxt_path, model_path


class FaceDetector:
    """Face detector preferring RetinaFace, with OpenCV DNN fallback."""

    def __init__(self) -> None:
        self._retinaface = None
        self._dnn_net: cv2.dnn_Net | None = None

        try:
            from retinaface import RetinaFace  # type: ignore

            self._retinaface = RetinaFace
        except Exception:
            self._retinaface = None

        if self._retinaface is None:
            model_paths = _resolve_open_cv_dnn_model_paths()
            if model_paths is not None:
                prototxt_path, model_path = model_paths
                try:
                    self._dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
                except AttributeError:
                    self._dnn_net = None
                except cv2.error:
                    self._dnn_net = None

    @property
    def backend_name(self) -> str:
        if self._retinaface is not None:
            return "retinaface"
        if self._dnn_net is not None:
            return "opencv_dnn_res10"
        return "none"

    def detect_largest_face(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int] | None:
        """Detect the largest face in an RGB frame and return (x, y, w, h)."""

        bbox = self._detect_with_retinaface(frame_rgb)
        if bbox is not None:
            return bbox
        return self._detect_with_dnn(frame_rgb)

    def _detect_with_retinaface(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int] | None:
        if self._retinaface is None:
            return None

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detections = self._retinaface.detect_faces(frame_bgr)
        if not isinstance(detections, dict) or len(detections) == 0:
            return None

        bboxes: List[Tuple[int, int, int, int]] = []
        for detection in detections.values():
            facial_area = detection.get("facial_area") if isinstance(detection, dict) else None
            if facial_area is None or len(facial_area) != 4:
                continue

            x1, y1, x2, y2 = [int(value) for value in facial_area]
            width = max(x2 - x1, 1)
            height = max(y2 - y1, 1)
            bboxes.append((x1, y1, width, height))

        if not bboxes:
            return None
        return max(bboxes, key=lambda candidate: candidate[2] * candidate[3])

    def _detect_with_dnn(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int] | None:
        if self._dnn_net is None:
            return None

        height, width = frame_rgb.shape[:2]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(frame_bgr, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        best_bbox: Tuple[int, int, int, int] | None = None
        best_score = DNN_CONFIDENCE_THRESHOLD

        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < best_score:
                continue

            x1 = int(detections[0, 0, index, 3] * width)
            y1 = int(detections[0, 0, index, 4] * height)
            x2 = int(detections[0, 0, index, 5] * width)
            y2 = int(detections[0, 0, index, 6] * height)

            left = max(min(x1, width - 1), 0)
            top = max(min(y1, height - 1), 0)
            right = max(min(x2, width), left + 1)
            bottom = max(min(y2, height), top + 1)

            best_score = confidence
            best_bbox = (left, top, right - left, bottom - top)

        return best_bbox


FACE_DETECTOR = FaceDetector()


@dataclass(frozen=True)
class VideoSample:
    """A single labeled video file."""

    path: Path
    label: int
    class_name: str
    video_id: str


def _is_video_file(path: Path, extensions: Sequence[str]) -> bool:
    """Return True when the path points to a supported video file."""

    return path.is_file() and path.suffix.lower() in extensions


def _resize_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
    """Resize a single RGB frame to a square target size."""

    return cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)


def detect_face_bbox(frame_rgb: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Detect the largest face in an RGB frame and return ``(x, y, w, h)``."""

    return FACE_DETECTOR.detect_largest_face(frame_rgb)


class FaceAlignmentProcessor:
    """Optional landmark-based face alignment using eye landmarks.

    The alignment step rotates the frame so the eye line becomes horizontal.
    If MediaPipe is unavailable or landmarks cannot be detected, alignment is
    skipped and the original frame is returned unchanged.
    """

    def __init__(self) -> None:
        self._face_mesh = None

        try:
            import mediapipe as mp  # type: ignore

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception:
            self._face_mesh = None

    @property
    def backend_name(self) -> str:
        return "mediapipe_face_mesh" if self._face_mesh is not None else "disabled"

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()

    def _landmark_points(self, landmarks: Sequence[object], indices: Sequence[int], width: int, height: int) -> np.ndarray:
        points = []
        for index in indices:
            landmark = landmarks[index]
            points.append((float(landmark.x) * width, float(landmark.y) * height))
        return np.asarray(points, dtype=np.float32)

    def align(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Rotate a frame so the eyes are level, or return the input unchanged."""

        if self._face_mesh is None:
            return frame_rgb, False

        result = self._face_mesh.process(frame_rgb)
        if not getattr(result, "multi_face_landmarks", None):
            return frame_rgb, False

        landmarks = result.multi_face_landmarks[0].landmark
        height, width = frame_rgb.shape[:2]

        left_eye = self._landmark_points(landmarks, LEFT_EYE_LANDMARKS, width, height)
        right_eye = self._landmark_points(landmarks, RIGHT_EYE_LANDMARKS, width, height)
        if left_eye.size == 0 or right_eye.size == 0:
            return frame_rgb, False

        left_center = left_eye.mean(axis=0)
        right_center = right_eye.mean(axis=0)
        eye_delta = right_center - left_center
        angle = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))

        if not np.isfinite(angle):
            return frame_rgb, False

        eye_center = tuple(((left_center + right_center) / 2.0).tolist())
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(
            frame_rgb,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned, True


def maybe_align_frame(
    frame_rgb: np.ndarray,
    align_faces: bool = False,
    face_aligner: FaceAlignmentProcessor | None = None,
) -> Tuple[np.ndarray, bool]:
    """Optionally align a face frame and report whether alignment occurred."""

    if not align_faces or face_aligner is None:
        return frame_rgb, False
    return face_aligner.align(frame_rgb)


def get_video_id(video_path: str | Path) -> str:
    """Resolve a stable video identifier from a filename."""

    return Path(video_path).stem


def apply_train_augmentations(
    frame_rgb_float: np.ndarray,
    do_flip: bool | None = None,
    contrast_scale: float | None = None,
    brightness_shift: float | None = None,
    do_noise: bool | None = None,
) -> np.ndarray:
    """Apply lightweight train-time augmentations on a float RGB frame in [0, 1]."""

    frame = frame_rgb_float

    if do_flip is None:
        do_flip = random.random() < 0.5
    if contrast_scale is None:
        if random.random() < 0.8:
            contrast_scale = random.uniform(0.9, 1.1)
            brightness_shift = random.uniform(-0.12, 0.12)
        else:
            contrast_scale = 1.0
            brightness_shift = 0.0
    if do_noise is None:
        do_noise = random.random() < 0.3

    if do_flip:
        frame = np.ascontiguousarray(np.fliplr(frame))

    if contrast_scale != 1.0 or brightness_shift != 0.0:
        frame = frame * contrast_scale + brightness_shift

    if do_noise:
        noise = np.random.normal(loc=0.0, scale=0.015, size=frame.shape).astype(np.float32)
        frame = frame + noise

    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def crop_face_region(frame_rgb: np.ndarray, margin: float = 0.18) -> np.ndarray:
    """Crop the detected face region with a small margin; fall back to the full frame."""

    bbox = detect_face_bbox(frame_rgb)
    if bbox is None:
        return frame_rgb

    height, width = frame_rgb.shape[:2]
    x, y, box_width, box_height = bbox

    x_margin = int(box_width * margin)
    y_margin = int(box_height * margin)

    left = max(x - x_margin, 0)
    top = max(y - y_margin, 0)
    right = min(x + box_width + x_margin, width)
    bottom = min(y + box_height + y_margin, height)

    return frame_rgb[top:bottom, left:right]


def prepare_frame_for_model(
    frame_rgb: np.ndarray,
    frame_size: int,
    apply_augmentation: bool = False,
    do_flip: bool | None = None,
    contrast_scale: float | None = None,
    brightness_shift: float | None = None,
    do_noise: bool | None = None,
    align_faces: bool = False,
    face_aligner: FaceAlignmentProcessor | None = None,
) -> np.ndarray:
    """Crop and normalize one frame for model input, with optional train-time augmentation."""

    working_frame, _ = maybe_align_frame(frame_rgb, align_faces=align_faces, face_aligner=face_aligner)
    face_region = crop_face_region(working_frame)
    resized_frame = _resize_frame(face_region, frame_size).astype(np.float32) / 255.0
    if apply_augmentation:
        resized_frame = apply_train_augmentations(
            resized_frame,
            do_flip=do_flip,
            contrast_scale=contrast_scale,
            brightness_shift=brightness_shift,
            do_noise=do_noise,
        )
    normalized_frame = (resized_frame - IMAGENET_MEAN) / IMAGENET_STD
    return normalized_frame


def prepare_display_and_model_frame(
    frame_rgb: np.ndarray,
    frame_size: int,
    align_faces: bool = False,
    face_aligner: FaceAlignmentProcessor | None = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Prepare a display crop and model crop from the same optionally aligned frame."""

    working_frame, aligned = maybe_align_frame(frame_rgb, align_faces=align_faces, face_aligner=face_aligner)
    face_region = crop_face_region(working_frame)
    used_face_crop = face_region.shape[:2] != working_frame.shape[:2]
    display_frame = cv2.resize(face_region, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
    resized_frame = display_frame.astype(np.float32) / 255.0
    model_frame = (resized_frame - IMAGENET_MEAN) / IMAGENET_STD
    return display_frame, model_frame, used_face_crop or aligned


def read_video_frames(video_path: Path) -> List[np.ndarray]:
    """Read all frames from a video file as RGB numpy arrays.

    Parameters
    ----------
    video_path:
        Path to the video file.

    Returns
    -------
    list of numpy.ndarray
        Frames in RGB order with shape (H, W, 3).

    Raises
    ------
    RuntimeError
        If the video cannot be opened or no frames can be decoded.
    """

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
            frames.append(frame_rgb)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames were decoded from video: {video_path}")

    return frames


def _pad_frames(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Pad a short frame list by repeating the last frame."""

    if len(frames) >= clip_length:
        return frames[:clip_length]

    padded = list(frames)
    last_frame = padded[-1]
    while len(padded) < clip_length:
        padded.append(last_frame)
    return padded


def _sample_training_clip(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Sample a random contiguous clip for training.

    When the video is shorter than the requested length, the last frame is
    repeated so that the returned clip always has exactly ``clip_length``
    frames.
    """

    if len(frames) <= clip_length:
        return _pad_frames(frames, clip_length)

    max_start = len(frames) - clip_length
    start_index = random.randint(0, max_start)
    clip = frames[start_index : start_index + clip_length]
    return clip


def _sample_evaluation_clip(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Sample a deterministic center clip for validation/testing."""

    if len(frames) <= clip_length:
        return _pad_frames(frames, clip_length)

    start_index = max((len(frames) - clip_length) // 2, 0)
    clip = frames[start_index : start_index + clip_length]
    return clip


def _frames_to_tensor(frames: List[np.ndarray]) -> Tensor:
    """Convert preprocessed RGB frames to a float tensor in (C, T, H, W) format."""

    clip_array = np.asarray(frames, dtype=np.float32)
    clip_array = np.transpose(clip_array, (3, 0, 1, 2))
    return torch.from_numpy(clip_array)


def _load_weak_label_values(
    weak_label_dir: Path,
    class_name: str,
    video_id: str,
    concept_names: Sequence[str],
) -> np.ndarray:
    """Load weak labels from JSON first, then NPY as a fallback."""

    class_dir = weak_label_dir / class_name
    json_path = class_dir / f"{video_id}_weak_labels.json"
    npy_path = class_dir / f"{video_id}_weak_labels.npy"

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        missing_keys = [name for name in concept_names if name not in payload]
        if missing_keys:
            raise KeyError(
                f"Weak-label JSON is missing keys {missing_keys} for video '{video_id}' at {json_path}"
            )

        values = np.asarray([float(payload[name]) for name in concept_names], dtype=np.float32)
        return np.clip(values, 0.0, 1.0)

    if npy_path.exists():
        values = np.load(npy_path)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != len(concept_names):
            raise ValueError(
                f"Weak-label NPY has size {values.size}, expected {len(concept_names)} for video '{video_id}' at {npy_path}"
            )
        return np.clip(values, 0.0, 1.0)

    raise FileNotFoundError(
        f"Missing weak-label files for video '{video_id}'. Expected one of: {json_path} or {npy_path}"
    )


class VideoDataset(Dataset[Tuple[Tensor, Tensor]]):
    """PyTorch dataset for binary video classification.

    Parameters
    ----------
    root_dir:
        Directory containing class subfolders such as ``real`` and ``fake``.
    clip_length:
        Number of frames per clip.
    frame_size:
        Final spatial resolution of each frame.
    is_train:
        If True, clips are sampled randomly. Otherwise, center clips are used.
    class_names:
        Ordered class names. The default maps ``real -> 0`` and ``fake -> 1``.
    video_extensions:
        Supported video file extensions.
    """

    def __init__(
        self,
        root_dir: str | Path,
        clip_length: int = 16,
        frame_size: int = 112,
        is_train: bool = True,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        video_extensions: Sequence[str] = VIDEO_EXTENSIONS,
        weak_label_dir: str | Path | None = None,
        return_weak_labels: bool = False,
        concept_names: Sequence[str] = DEFAULT_CONCEPT_NAMES,
        align_faces: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.is_train = is_train
        self.class_names = tuple(class_names)
        self.video_extensions = tuple(ext.lower() for ext in video_extensions)
        self.return_weak_labels = return_weak_labels
        self.concept_names = tuple(concept_names)
        self.weak_label_dir = Path(weak_label_dir) if weak_label_dir is not None else None
        self.align_faces = align_faces
        self.face_aligner: FaceAlignmentProcessor | None = None

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        if self.return_weak_labels and self.weak_label_dir is None:
            raise ValueError("return_weak_labels=True requires weak_label_dir to be provided.")
        if self.return_weak_labels and not self.concept_names:
            raise ValueError("concept_names cannot be empty when return_weak_labels=True.")

        self.class_to_idx: Dict[str, int] = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        self.samples: List[VideoSample] = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                f"No video files were found under {self.root_dir}. "
                f"Expected class folders: {', '.join(self.class_names)}"
            )

    def _discover_samples(self) -> List[VideoSample]:
        """Scan the directory tree and collect labeled video samples."""

        samples: List[VideoSample] = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue

            for video_path in sorted(class_dir.iterdir()):
                if _is_video_file(video_path, self.video_extensions):
                    samples.append(
                        VideoSample(
                            path=video_path,
                            label=label,
                            class_name=class_name,
                            video_id=get_video_id(video_path),
                        )
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_face_aligner(self) -> FaceAlignmentProcessor | None:
        """Create the optional aligner lazily so DataLoader workers can pickle the dataset."""

        if not self.align_faces:
            return None
        if self.face_aligner is None:
            self.face_aligner = FaceAlignmentProcessor()
        return self.face_aligner

    def __getitem__(self, index: int):
        """Load a video clip and its label.

        The returned clip has shape ``(3, T, H, W)`` and the label is a scalar
        ``torch.long`` tensor.
        """

        sample = self.samples[index]
        frames = read_video_frames(sample.path)
        face_aligner = self._get_face_aligner()

        if self.is_train:
            clip_frames = _sample_training_clip(frames, self.clip_length)
            # Sample parameters consistently for this clip
            do_flip = random.random() < 0.5
            if random.random() < 0.8:
                contrast_scale = random.uniform(0.9, 1.1)
                brightness_shift = random.uniform(-0.12, 0.12)
            else:
                contrast_scale = 1.0
                brightness_shift = 0.0
            do_noise = random.random() < 0.3

            clip_frames = [
                prepare_frame_for_model(
                    frame_rgb,
                    self.frame_size,
                    apply_augmentation=True,
                    do_flip=do_flip,
                    contrast_scale=contrast_scale,
                    brightness_shift=brightness_shift,
                    do_noise=do_noise,
                    align_faces=self.align_faces,
                    face_aligner=face_aligner,
                )
                for frame_rgb in clip_frames
            ]
        else:
            clip_frames = _sample_evaluation_clip(frames, self.clip_length)
            clip_frames = [
                prepare_frame_for_model(
                    frame_rgb,
                    self.frame_size,
                    apply_augmentation=False,
                    align_faces=self.align_faces,
                    face_aligner=face_aligner,
                )
                for frame_rgb in clip_frames
            ]

        clip_tensor = _frames_to_tensor(clip_frames)
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        if not self.return_weak_labels:
            return clip_tensor, label_tensor

        weak_label_values = _load_weak_label_values(
            weak_label_dir=self.weak_label_dir,
            class_name=sample.class_name,
            video_id=sample.video_id,
            concept_names=self.concept_names,
        )
        weak_label_tensor = torch.from_numpy(weak_label_values.astype(np.float32, copy=False))
        return clip_tensor, label_tensor, weak_label_tensor

    def get_video_path(self, index: int) -> Path:
        """Return the source path for a dataset item."""

        return self.samples[index].path

    def get_class_to_idx(self) -> Dict[str, int]:
        """Return the class mapping used by the dataset."""

        return dict(self.class_to_idx)

    def close(self) -> None:
        """Release optional resources held by the dataset."""

        if self.face_aligner is not None:
            self.face_aligner.close()
