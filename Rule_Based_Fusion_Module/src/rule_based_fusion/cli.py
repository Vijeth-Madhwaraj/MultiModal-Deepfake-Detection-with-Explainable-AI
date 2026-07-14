"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .engine import FusionEngine
from .io import read_json, write_json
from .models import InputReport, ValidationError
from .pipeline import InferenceCommand, PipelineError, VideoPipeline, discover_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rule-based fusion from an MP4, input folder, or detector JSON"
    )
    parser.add_argument("input", help="MP4 file, folder containing MP4 files, or detector JSON")
    parser.add_argument("-o", "--output", help="output JSON path, or output directory for batch mode")
    parser.add_argument("--config", help="optional rule configuration JSON")
    parser.add_argument("--inference-config", help="JSON config for the upstream inference command")
    parser.add_argument("--detector-results", help="precomputed detector JSON for one MP4")
    parser.add_argument("--work-dir", default="work/inference", help="upstream JSON working directory")
    parser.add_argument("--reuse-results", action="store_true", help="reuse working detector JSON")
    parser.add_argument("--recursive", action="store_true", help="scan input folders recursively")
    parser.add_argument("--validate-only", action="store_true", help="validate JSON or MP4 and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = Path(args.input)
        if source.suffix.lower() == ".json" and source.is_file():
            report = InputReport.from_dict(read_json(source))
            if args.validate_only:
                print("Detector JSON is valid.")
                return 0
            result = FusionEngine(load_config(args.config)).fuse(report)
            if args.output:
                write_json(result, args.output)
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        videos = discover_videos(source, recursive=args.recursive)
        if args.validate_only:
            for video in videos:
                print(f"Valid MP4: {video}")
            return 0
        if args.detector_results and len(videos) != 1:
            raise PipelineError("--detector-results can only be used with one MP4")

        inference = (
            InferenceCommand.from_file(args.inference_config)
            if args.inference_config
            else None
        )
        pipeline = VideoPipeline(
            fusion_config=load_config(args.config),
            inference_command=inference,
            work_dir=args.work_dir,
        )
        batch_mode = source.is_dir() or len(videos) > 1
        output_base = Path(args.output or "output")
        for video in videos:
            output_path = (
                output_base / f"{video.stem}_fused.json"
                if batch_mode or not output_base.suffix
                else output_base
            )
            result = pipeline.process(
                video,
                output_path,
                detector_results=args.detector_results,
                reuse_results=args.reuse_results,
            )
            fusion = result["results"]["rule_based_fusion"]
            print(
                f"{video.name}: {fusion['label']} "
                f"(confidence={fusion['confidence']}) -> {output_path}"
            )
        return 0
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        PipelineError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
