#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import hydra
import json
import os
import gc
import tempfile
from pipelines.pipeline import InferencePipeline


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


def extract_audio_to_wav(data_filename):
    if os.path.splitext(data_filename)[1].lower() in AUDIO_EXTENSIONS:
        return data_filename, None

    import av
    import numpy as np
    import soundfile as sf

    tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_audio.close()

    chunks = []
    with av.open(data_filename) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError(f"No audio stream found in: {data_filename}")

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(audio_stream):
            for resampled_frame in resampler.resample(frame):
                samples = resampled_frame.to_ndarray()
                chunks.append(samples.reshape(-1))

    if not chunks:
        raise RuntimeError(f"Could not extract audio from: {data_filename}")

    audio = np.concatenate(chunks).astype(np.float32) / np.iinfo(np.int16).max
    sf.write(tmp_audio.name, audio, 16000)
    return tmp_audio.name, tmp_audio.name


def compare_audio_video_transcripts(cfg, device):
    audio_pipeline = InferencePipeline(
        cfg.audio_config_filename,
        device=device,
        detector=cfg.detector,
        face_track=False,
    )
    audio_filename, tmp_audio_filename = extract_audio_to_wav(cfg.data_filename)
    try:
        audio_transcript = audio_pipeline(audio_filename)
    finally:
        if tmp_audio_filename:
            os.remove(tmp_audio_filename)

    # The ASR and VSR checkpoints are large. Release the audio model before
    # loading the video model so the combined analysis does not hold both in
    # RAM/VRAM at the same time.
    del audio_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    video_pipeline = InferencePipeline(
        cfg.video_config_filename,
        device=device,
        detector=cfg.detector,
        face_track=True,
    )

    video_transcript = video_pipeline(cfg.data_filename, cfg.landmarks_filename)
    del video_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from metrics.comparator import compute_transcript_metrics

    metrics = compute_transcript_metrics(
        reference=audio_transcript,
        hypothesis=video_transcript,
        include_semantic=cfg.semantic_similarity,
    )

    return {
        "audio_transcript": audio_transcript,
        "video_transcript": video_transcript,
        "metrics": metrics,
    }


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")

    if cfg.compare_transcripts:
        output = compare_audio_video_transcripts(cfg, device)
        output_text = json.dumps(output, indent=2)
        print(output_text)
    else:
        output = InferencePipeline(cfg.config_filename, device=device, detector=cfg.detector, face_track=True)(cfg.data_filename, cfg.landmarks_filename)
        output_text = f"hyp: {output}"
        print(output_text)

    if cfg.dst_filename:
        with open(cfg.dst_filename, "w", encoding="utf-8") as f:
            f.write(output_text)
            f.write("\n")


if __name__ == '__main__':
    main()
