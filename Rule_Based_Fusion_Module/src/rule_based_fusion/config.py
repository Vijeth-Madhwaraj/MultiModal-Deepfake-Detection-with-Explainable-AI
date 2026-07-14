"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "thresholds": {
        "video_fake_score",
        "audio_fake_score",
        "transcript_consistency_max_anomaly",
    },
    "source_weights": {
        "cnn_3d",
        "weak_label_heuristics",
    },
    "weak_label_weights": {
        "boundary_inconsistency",
        "eye_blink_irregularity",
    },
    "transcript_weights": {
        "wer",
        "word_mismatch",
        "character_dissimilarity",
        "semantic_dissimilarity",
    },
}

@dataclass(frozen=True)
class FusionConfig:
    thresholds: dict[str, float]
    source_weights: dict[str, float]
    weak_label_weights: dict[str, float]
    transcript_weights: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FusionConfig":
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a JSON object")

        unknown_groups = set(data) - set(REQUIRED_KEYS)
        if unknown_groups:
            raise ValueError(
                f"unknown configuration group(s): {', '.join(sorted(unknown_groups))}"
            )

        parsed: dict[str, dict[str, float]] = {}
        for group, required_keys in REQUIRED_KEYS.items():
            values = data.get(group)
            if not isinstance(values, dict):
                raise ValueError(f"configuration group '{group}' is required")

            missing = required_keys - set(values)
            unknown = set(values) - required_keys
            if missing:
                raise ValueError(
                    f"{group} is missing required key(s): {', '.join(sorted(missing))}"
                )
            if unknown:
                raise ValueError(
                    f"{group} contains unknown key(s): {', '.join(sorted(unknown))}"
                )

            parsed[group] = {}
            for key, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{group}.{key} must be a number")
                parsed[group][key] = float(value)

        config = cls(**parsed)

        for key, value in config.thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"thresholds.{key} must be between 0 and 1")

        for name in ("source_weights", "weak_label_weights", "transcript_weights"):
            weights = getattr(config, name)
            if any(value < 0 for value in weights.values()):
                raise ValueError(f"{name} cannot contain negative weights")
            if abs(sum(weights.values()) - 1.0) > 1e-9:
                raise ValueError(f"{name} must sum to 1")
        return config


def default_config_path() -> Path:
    repository_config = Path(__file__).resolve().parents[2] / "config" / "default_rules.json"
    if repository_config.exists():
        return repository_config
    return Path(__file__).with_name("default_rules.json")


def load_config(path: str | Path | None = None) -> FusionConfig:
    config_path = Path(path) if path else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        return FusionConfig.from_dict(json.load(handle))
