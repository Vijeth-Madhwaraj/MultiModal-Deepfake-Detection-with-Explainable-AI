from jiwer import wer


def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    Compute Word Error Rate (WER) and Word Match percentage.

    Args:
        reference (str): Ground truth transcript (Audio)
        hypothesis (str): Predicted transcript (Video)

    Returns:
        dict:
        {
            "wer": float,
            "word_match": float
        }
    """

    reference = reference.strip()
    hypothesis = hypothesis.strip()

    # Handle empty inputs
    if not reference and not hypothesis:
        return {
            "wer": 0.0,
            "word_match": 100.0
        }

    if not reference:
        return {
            "wer": 1.0,
            "word_match": 0.0
        }

    error = wer(reference, hypothesis)

    # WER can exceed 1.0 in some cases
    error = min(error, 1.0)

    word_match = (1 - error) * 100

    return {
        "wer": round(error, 4),
        "word_match": round(word_match, 2)
    }