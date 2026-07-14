"""MP4-first orchestration for upstream inference and rule-based fusion."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FusionConfig
from .engine import FusionEngine
from .io import read_json, write_json
from .models import InputReport, ValidationError


class PipelineError(RuntimeError):
    """Raised when a video cannot reach the fusion stage."""


def validate_mp4(path: str | Path) -> Path:
    video = Path(path).resolve()
    if not video.is_file():
        raise PipelineError(f"video does not exist: {video}")
    if video.suffix.lower() != ".mp4":
        raise PipelineError(f"expected an .mp4 file: {video}")
    if video.stat().st_size < 12:
        raise PipelineError(f"MP4 file is empty or truncated: {video}")
    with video.open("rb") as handle:
        header = handle.read(64)
    if b"ftyp" not in header:
        raise PipelineError(f"file does not contain an MP4 ftyp header: {video}")
    return video


@dataclass(frozen=True)
class InferenceCommand:
    command: list[str]
    timeout_seconds: int = 1800

    @classmethod
    def from_file(cls, path: str | Path) -> "InferenceCommand":
        data = read_json(path)
        command = data.get("command") if isinstance(data, dict) else None
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise PipelineError("inference config 'command' must be a non-empty string array")
        timeout = data.get("timeout_seconds", 1800)
        if not isinstance(timeout, int) or timeout <= 0:
            raise PipelineError("inference config 'timeout_seconds' must be a positive integer")
        return cls(command=command, timeout_seconds=timeout)

    def run(self, video: Path, output_json: Path) -> None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        replacements = {
            "{input_video}": str(video),
            "{output_json}": str(output_json.resolve()),
        }
        command = [replacements.get(part, part) for part in self.command]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise PipelineError(f"inference executable was not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PipelineError(
                f"upstream inference timed out after {self.timeout_seconds} seconds"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise PipelineError(
                f"upstream inference failed with exit code {completed.returncode}"
                + (f": {detail}" if detail else "")
            )
        if not output_json.is_file():
            raise PipelineError(f"upstream inference did not create: {output_json}")


class VideoPipeline:
    def __init__(
        self,
        *,
        fusion_config: FusionConfig | None = None,
        inference_command: InferenceCommand | None = None,
        work_dir: str | Path = "work/inference",
    ) -> None:
        self.engine = FusionEngine(fusion_config)
        self.inference_command = inference_command
        self.work_dir = Path(work_dir)

    def process(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        detector_results: str | Path | None = None,
        reuse_results: bool = False,
    ) -> dict[str, Any]:
        video = validate_mp4(video_path)
        results_path = (
            Path(detector_results)
            if detector_results
            else self.work_dir / f"{video.stem}.json"
        )

        should_run = self.inference_command is not None and not (
            reuse_results and results_path.is_file()
        )
        if should_run:
            self.inference_command.run(video, results_path)
        elif not results_path.is_file():
            raise PipelineError(
                "no detector result JSON is available for this MP4. "
                "Configure --inference-config, or supply --detector-results. "
                "The repository does not contain the upstream CNN/RF/transcript models."
            )

        try:
            report = InputReport.from_dict(read_json(results_path))
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise PipelineError(f"invalid detector result JSON '{results_path}': {exc}") from exc

        reported_video = Path(str(report.raw.get("input_video", ""))).resolve()
        if reported_video != video:
            raise PipelineError(
                "detector JSON belongs to a different video: "
                f"reported '{reported_video}', selected '{video}'"
            )

        fused = self.engine.fuse(report)
        write_json(fused, output_path)
        return fused


def discover_videos(path: str | Path, recursive: bool = False) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [validate_mp4(source)]
    if not source.is_dir():
        raise PipelineError(f"input path does not exist: {source.resolve()}")
    pattern = "**/*.mp4" if recursive else "*.mp4"
    videos = sorted(item.resolve() for item in source.glob(pattern) if item.is_file())
    if not videos:
        raise PipelineError(f"no MP4 files found in: {source.resolve()}")
    return videos
