"""Typed parsing and validation for the upstream JSON report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ValidationError(ValueError):
    """Raised when an input report does not match the expected schema."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be a JSON object")
    return value


def _required(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise ValidationError(f"missing required field: {path}.{key}")
    return data[key]


def _number(data: Mapping[str, Any], key: str, path: str) -> float:
    value = _required(data, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path}.{key} must be a number")
    return float(value)


def _bounded(value: float, low: float, high: float, path: str) -> float:
    if not low <= value <= high:
        raise ValidationError(f"{path} must be between {low} and {high}")
    return value


def _label(data: Mapping[str, Any], path: str) -> str:
    value = _required(data, "label", path)
    if value not in {"REAL", "FAKE"}:
        raise ValidationError(f"{path}.label must be REAL or FAKE")
    return value


@dataclass(frozen=True)
class CnnResult:
    label: str
    confidence: float


@dataclass(frozen=True)
class WeakLabelResult:
    boundary_inconsistency: float
    eye_blink_irregularity: float


@dataclass(frozen=True)
class RandomForestResult:
    label: str
    confidence: float
    fake_probability: float
    real_probability: float


@dataclass(frozen=True)
class TranscriptResult:
    wer: float
    word_match_rate: float
    character_similarity: float
    semantic_similarity: float


@dataclass(frozen=True)
class InputReport:
    raw: dict[str, Any]
    cnn: CnnResult
    weak: WeakLabelResult
    random_forest: RandomForestResult
    transcript: TranscriptResult

    @classmethod
    def from_dict(cls, value: Any) -> "InputReport":
        root = _mapping(value, "root")
        if _required(root, "schema_version", "root") != 1:
            raise ValidationError("root.schema_version must be 1")
        if _required(root, "status", "root") != "complete":
            raise ValidationError("root.status must be complete before fusion")

        encoding = _mapping(_required(root, "label_encoding", "root"), "label_encoding")
        if encoding.get("REAL") != 0 or encoding.get("FAKE") != 1:
            raise ValidationError("label_encoding must be exactly REAL=0 and FAKE=1")

        results = _mapping(_required(root, "results", "root"), "results")
        cnn_data = _mapping(_required(results, "cnn_3d", "results"), "results.cnn_3d")
        weak_data = _mapping(
            _required(results, "weak_label_heuristics", "results"),
            "results.weak_label_heuristics",
        )
        rf_data = _mapping(
            _required(results, "random_forest", "results"), "results.random_forest"
        )
        transcript_data = _mapping(
            _required(results, "transcript_comparison", "results"),
            "results.transcript_comparison",
        )

        cnn_label = _label(cnn_data, "results.cnn_3d")
        cnn_confidence = _bounded(
            _number(cnn_data, "confidence", "results.cnn_3d"),
            0.0,
            1.0,
            "results.cnn_3d.confidence",
        )
        cnn_label_id = _required(cnn_data, "label_id", "results.cnn_3d")
        if cnn_label_id != encoding[cnn_label]:
            raise ValidationError("results.cnn_3d.label_id does not match label_encoding")

        boundary = _bounded(
            _number(weak_data, "boundary_inconsistency", "results.weak_label_heuristics"),
            0.0,
            1.0,
            "results.weak_label_heuristics.boundary_inconsistency",
        )
        eye = _bounded(
            _number(weak_data, "eye_blink_irregularity", "results.weak_label_heuristics"),
            0.0,
            1.0,
            "results.weak_label_heuristics.eye_blink_irregularity",
        )

        rf_label = _label(rf_data, "results.random_forest")
        rf_confidence = _bounded(
            _number(rf_data, "confidence", "results.random_forest"),
            0.0,
            1.0,
            "results.random_forest.confidence",
        )
        fake_probability = _bounded(
            _number(rf_data, "fake_probability", "results.random_forest"),
            0.0,
            1.0,
            "results.random_forest.fake_probability",
        )
        real_probability = _bounded(
            _number(rf_data, "real_probability", "results.random_forest"),
            0.0,
            1.0,
            "results.random_forest.real_probability",
        )
        if abs(fake_probability + real_probability - 1.0) > 1e-6:
            raise ValidationError("random-forest probabilities must sum to 1")
        if _required(rf_data, "label_id", "results.random_forest") != encoding[rf_label]:
            raise ValidationError("results.random_forest.label_id does not match label_encoding")
        expected_confidence = fake_probability if rf_label == "FAKE" else real_probability
        if abs(rf_confidence - expected_confidence) > 1e-6:
            raise ValidationError("random-forest confidence must equal its predicted probability")

        transcript = TranscriptResult(
            wer=_bounded(
                _number(transcript_data, "wer", "results.transcript_comparison"),
                0.0,
                1.0,
                "results.transcript_comparison.wer",
            ),
            word_match_rate=_bounded(
                _number(transcript_data, "word_match_rate", "results.transcript_comparison"),
                0.0,
                100.0,
                "results.transcript_comparison.word_match_rate",
            ),
            character_similarity=_bounded(
                _number(transcript_data, "character_similarity", "results.transcript_comparison"),
                0.0,
                100.0,
                "results.transcript_comparison.character_similarity",
            ),
            semantic_similarity=_bounded(
                _number(transcript_data, "semantic_similarity", "results.transcript_comparison"),
                0.0,
                100.0,
                "results.transcript_comparison.semantic_similarity",
            ),
        )

        return cls(
            raw=dict(root),
            cnn=CnnResult(cnn_label, cnn_confidence),
            weak=WeakLabelResult(boundary, eye),
            random_forest=RandomForestResult(
                rf_label, rf_confidence, fake_probability, real_probability
            ),
            transcript=transcript,
        )
