"""Deterministic and explainable fusion engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import FusionConfig, load_config
from .models import InputReport


def _round(value: float) -> float:
    return round(value, 6)


class FusionEngine:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or load_config()

    def fuse(self, report: InputReport) -> dict[str, Any]:
        c = self.config
        t = c.thresholds

        cnn_fake = report.cnn.confidence if report.cnn.label == "FAKE" else 1.0 - report.cnn.confidence
        rf_fake = report.random_forest.fake_probability
        weak_fake = (
            report.weak.boundary_inconsistency
            * c.weak_label_weights["boundary_inconsistency"]
            + report.weak.eye_blink_irregularity
            * c.weak_label_weights["eye_blink_irregularity"]
        )
        transcript_parts = {
            "wer": report.transcript.wer,
            "word_mismatch": 1.0 - report.transcript.word_match_rate / 100.0,
            "character_dissimilarity": 1.0 - report.transcript.character_similarity / 100.0,
            "semantic_dissimilarity": 1.0 - report.transcript.semantic_similarity / 100.0,
        }
        transcript_fake = sum(
            transcript_parts[name] * weight
            for name, weight in c.transcript_weights.items()
        )

        evidence = {
            "cnn_3d_fake_score": cnn_fake,
            "random_forest_fake_score": rf_fake,
            "weak_label_fake_score": weak_fake,
            "transcript_anomaly_score": transcript_fake,
        }
        # Visual authenticity is decided only from visual evidence. Re-normalize
        # the existing CNN and weak-label source weights so they sum to one.
        cnn_weight = c.source_weights["cnn_3d"]
        weak_weight = c.source_weights["weak_label_heuristics"]
        visual_weight_total = cnn_weight + weak_weight
        if visual_weight_total <= 0.0:
            raise ValueError("CNN and weak-label source weights cannot both be zero")
        video_fake_score = (
            cnn_fake * cnn_weight + weak_fake * weak_weight
        ) / visual_weight_total
        video_label = "FAKE" if video_fake_score >= t["video_fake_score"] else "REAL"
        video_confidence = (
            video_fake_score if video_label == "FAKE" else 1.0 - video_fake_score
        )

        # The Random Forest is trained on audio features, so it supplies the
        # audio-side authenticity decision without mixing in visual evidence.
        audio_fake_score = rf_fake
        audio_label = "FAKE" if audio_fake_score >= t["audio_fake_score"] else "REAL"
        audio_confidence = (
            audio_fake_score if audio_label == "FAKE" else 1.0 - audio_fake_score
        )

        transcript_consistent = transcript_fake <= t["transcript_consistency_max_anomaly"]
        transcript_match = 1.0 - transcript_fake
        transcript_confidence = (
            transcript_match if transcript_consistent else transcript_fake
        )

        joint_labels = {
            ("REAL", "REAL"): ("REAL_VIDEO_REAL_AUDIO", 0, "Real video / Real audio"),
            ("REAL", "FAKE"): ("REAL_VIDEO_FAKE_AUDIO", 1, "Real video / Fake audio"),
            ("FAKE", "REAL"): ("FAKE_VIDEO_REAL_AUDIO", 2, "Fake video / Real audio"),
            ("FAKE", "FAKE"): ("FAKE_VIDEO_FAKE_AUDIO", 3, "Fake video / Fake audio"),
        }
        label, label_id, display_label = joint_labels[(video_label, audio_label)]
        # A joint four-way decision is only as reliable as its weaker modality.
        confidence = min(video_confidence, audio_confidence)

        rules = [
            {
                "rule": "visual_authenticity",
                "matched": video_label == "FAKE",
                "details": (
                    f"visual fake score {_round(video_fake_score)} compared with "
                    f"threshold {t['video_fake_score']}"
                ),
            },
            {
                "rule": "audio_authenticity",
                "matched": audio_label == "FAKE",
                "details": (
                    f"audio fake score {_round(audio_fake_score)} compared with "
                    f"threshold {t['audio_fake_score']}"
                ),
            },
            {
                "rule": "transcript_consistency",
                "matched": transcript_consistent,
                "details": (
                    f"transcript anomaly score {_round(transcript_fake)} compared with maximum "
                    f"{t['transcript_consistency_max_anomaly']}"
                ),
            },
            {
                "rule": "four_way_modality_combination",
                "matched": True,
                "details": f"{video_label} video combined with {audio_label} audio",
            },
        ]

        fusion = {
            "label": label,
            "label_id": label_id,
            "display_label": display_label,
            "confidence": _round(confidence),
            "video": {
                "label": video_label,
                "label_id": 1 if video_label == "FAKE" else 0,
                "confidence": _round(video_confidence),
                "fake_score": _round(video_fake_score),
                "real_score": _round(1.0 - video_fake_score),
                "decision_basis": "weighted_cnn_and_visual_heuristics",
            },
            "audio": {
                "label": audio_label,
                "label_id": 1 if audio_label == "FAKE" else 0,
                "confidence": _round(audio_confidence),
                "fake_score": _round(audio_fake_score),
                "real_score": _round(1.0 - audio_fake_score),
                "decision_basis": "random_forest_audio_classifier",
            },
            "cross_modal_consistency": {
                "label": "CONSISTENT" if transcript_consistent else "INCONSISTENT",
                "consistent": transcript_consistent,
                "confidence": _round(transcript_confidence),
                "transcript_anomaly_score": _round(transcript_fake),
                "transcript_match_score": _round(transcript_match),
            },
            "evidence": {key: _round(value) for key, value in evidence.items()},
            "rule_trace": rules,
            "method": "deterministic_modality_fusion_v2",
        }

        output = deepcopy(report.raw)
        output["results"] = dict(output["results"])
        output["results"]["rule_based_fusion"] = fusion
        return output
