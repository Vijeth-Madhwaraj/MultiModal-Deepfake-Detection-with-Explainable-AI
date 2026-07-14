from metrics.character import compute_character_similarity
from metrics.wer import compute_wer


def compute_transcript_metrics(reference: str, hypothesis: str, include_semantic: bool = True) -> dict:
    """
    Compare audio and video transcripts using all configured metrics.

    Args:
        reference (str): Audio transcript, treated as the reference.
        hypothesis (str): Video transcript, treated as the prediction.
        include_semantic (bool): Whether to compute sentence-transformer similarity.

    Returns:
        dict containing WER, word match, character similarity, and optionally
        semantic similarity.
    """

    metrics = {}
    metrics.update(compute_wer(reference, hypothesis))
    metrics.update(compute_character_similarity(reference, hypothesis))

    if include_semantic:
        from metrics.semantic import compute_semantic_similarity

        metrics.update(compute_semantic_similarity(reference, hypothesis))

    return metrics
