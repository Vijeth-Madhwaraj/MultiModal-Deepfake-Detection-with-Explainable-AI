from difflib import SequenceMatcher


def compute_character_similarity(reference: str, hypothesis: str) -> dict:
    """
    Compute character-level similarity between two transcripts.

    Args:
        reference (str): Ground truth transcript (Audio)
        hypothesis (str): Predicted transcript (Video)

    Returns:
        dict:
        {
            "character_similarity": float
        }
    """

    reference = reference.strip()
    hypothesis = hypothesis.strip()

    # Handle empty inputs
    if not reference and not hypothesis:
        return {
            "character_similarity": 100.0
        }

    if not reference or not hypothesis:
        return {
            "character_similarity": 0.0
        }

    similarity = SequenceMatcher(
        None,
        reference.lower(),
        hypothesis.lower()
    ).ratio() * 100

    return {
        "character_similarity": round(similarity, 2)
    }