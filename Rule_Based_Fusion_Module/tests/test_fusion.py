import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from rule_based_fusion import (
    FusionEngine,
    InputReport,
    PipelineError,
    ValidationError,
    VideoPipeline,
)
from rule_based_fusion.config import FusionConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = json.loads((ROOT / "examples" / "00181_input.json").read_text(encoding="utf-8"))

    def test_supplied_report_is_fake_video_real_audio(self):
        output = FusionEngine().fuse(InputReport.from_dict(self.sample))
        fusion = output["results"]["rule_based_fusion"]
        self.assertEqual(fusion["label"], "FAKE_VIDEO_REAL_AUDIO")
        self.assertEqual(fusion["label_id"], 2)
        self.assertEqual(fusion["video"]["label"], "FAKE")
        self.assertEqual(fusion["audio"]["label"], "REAL")
        self.assertAlmostEqual(fusion["evidence"]["weak_label_fake_score"], 0.400362, places=6)

    def test_supplied_report_becomes_fake_video_fake_audio(self):
        data = deepcopy(self.sample)
        data["results"]["random_forest"].update({
            "label": "FAKE", "label_id": 1, "confidence": 0.9,
            "fake_probability": 0.9, "real_probability": 0.1,
        })
        fusion = FusionEngine().fuse(InputReport.from_dict(data))["results"]["rule_based_fusion"]
        self.assertEqual(fusion["label"], "FAKE_VIDEO_FAKE_AUDIO")
        self.assertEqual(fusion["label_id"], 3)
        self.assertEqual(fusion["video"]["label"], "FAKE")
        self.assertEqual(fusion["audio"]["label"], "FAKE")

    def _four_way_report(self, video_label, audio_label):
        data = deepcopy(self.sample)
        video_fake = video_label == "FAKE"
        data["results"]["cnn_3d"].update({
            "label": video_label,
            "label_id": 1 if video_fake else 0,
            "confidence": 0.9,
        })
        visual_score = 0.9 if video_fake else 0.1
        data["results"]["weak_label_heuristics"].update({
            "boundary_inconsistency": visual_score,
            "eye_blink_irregularity": visual_score,
        })
        audio_fake = audio_label == "FAKE"
        fake_probability = 0.9 if audio_fake else 0.1
        data["results"]["random_forest"].update({
            "label": audio_label,
            "label_id": 1 if audio_fake else 0,
            "confidence": 0.9,
            "fake_probability": fake_probability,
            "real_probability": 1.0 - fake_probability,
        })
        return data

    def test_all_four_modality_labels(self):
        cases = {
            ("REAL", "REAL"): ("REAL_VIDEO_REAL_AUDIO", 0),
            ("REAL", "FAKE"): ("REAL_VIDEO_FAKE_AUDIO", 1),
            ("FAKE", "REAL"): ("FAKE_VIDEO_REAL_AUDIO", 2),
            ("FAKE", "FAKE"): ("FAKE_VIDEO_FAKE_AUDIO", 3),
        }
        for (video_label, audio_label), expected in cases.items():
            with self.subTest(video=video_label, audio=audio_label):
                data = self._four_way_report(video_label, audio_label)
                fusion = FusionEngine().fuse(InputReport.from_dict(data))["results"]["rule_based_fusion"]
                self.assertEqual((fusion["label"], fusion["label_id"]), expected)

    def test_rejects_inconsistent_label_id(self):
        data = deepcopy(self.sample)
        data["results"]["cnn_3d"]["label_id"] = 0
        with self.assertRaises(ValidationError):
            InputReport.from_dict(data)

    def test_preserves_upstream_fields(self):
        output = FusionEngine().fuse(InputReport.from_dict(self.sample))
        self.assertEqual(output["input_video"], self.sample["input_video"])
        self.assertEqual(output["results"]["cnn_3d"], self.sample["results"]["cnn_3d"])

    def test_mp4_pipeline_accepts_precomputed_detector_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
            detector_json = root / "sample.json"
            detector_data = deepcopy(self.sample)
            detector_data["input_video"] = str(video.resolve())
            detector_json.write_text(json.dumps(detector_data), encoding="utf-8")
            output = root / "sample_fused.json"

            result = VideoPipeline(work_dir=root / "work").process(
                video, output, detector_results=detector_json
            )

            self.assertTrue(output.is_file())
            self.assertEqual(
                result["results"]["rule_based_fusion"]["label"],
                "FAKE_VIDEO_REAL_AUDIO",
            )

    def test_mp4_pipeline_refuses_to_invent_missing_detector_scores(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
            with self.assertRaisesRegex(PipelineError, "no detector result JSON"):
                VideoPipeline(work_dir=root / "work").process(video, root / "out.json")

    def test_video_threshold_treats_exact_threshold_as_fake(self):
        data = deepcopy(self.sample)
        data["results"]["cnn_3d"].update(
            {"label": "FAKE", "label_id": 1, "confidence": 0.5}
        )
        data["results"]["random_forest"].update(
            {
                "label": "REAL",
                "label_id": 0,
                "confidence": 0.6,
                "fake_probability": 0.4,
                "real_probability": 0.6,
            }
        )
        data["results"]["weak_label_heuristics"].update(
            {"boundary_inconsistency": 0.5, "eye_blink_irregularity": 0.5}
        )
        data["results"]["transcript_comparison"].update(
            {
                "wer": 0.2,
                "word_match_rate": 80.0,
                "character_similarity": 80.0,
                "semantic_similarity": 80.0,
            }
        )

        fusion = FusionEngine().fuse(InputReport.from_dict(data))["results"][
            "rule_based_fusion"
        ]
        self.assertEqual(fusion["video"]["fake_score"], 0.5)
        self.assertEqual(fusion["video"]["label"], "FAKE")
        self.assertEqual(fusion["label"], "FAKE_VIDEO_REAL_AUDIO")

    def test_rejects_missing_result_section(self):
        data = deepcopy(self.sample)
        del data["results"]["transcript_comparison"]
        with self.assertRaisesRegex(ValidationError, "transcript_comparison"):
            InputReport.from_dict(data)

    def test_default_configuration_is_valid(self):
        config = load_config()
        self.assertEqual(sum(config.source_weights.values()), 1.0)

    def test_configuration_rejects_missing_key(self):
        data = json.loads((ROOT / "config" / "default_rules.json").read_text())
        del data["thresholds"]["video_fake_score"]
        with self.assertRaisesRegex(ValueError, "video_fake_score"):
            FusionConfig.from_dict(data)

    def test_configuration_rejects_boolean_weight(self):
        data = json.loads((ROOT / "config" / "default_rules.json").read_text())
        data["source_weights"]["cnn_3d"] = True
        with self.assertRaisesRegex(ValueError, "must be a number"):
            FusionConfig.from_dict(data)

    def test_configuration_rejects_out_of_range_threshold(self):
        data = json.loads((ROOT / "config" / "default_rules.json").read_text())
        data["thresholds"]["video_fake_score"] = 1.1
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            FusionConfig.from_dict(data)


if __name__ == "__main__":
    unittest.main()
