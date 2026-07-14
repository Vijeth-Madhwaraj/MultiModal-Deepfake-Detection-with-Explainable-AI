from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model only once when the module is imported
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_semantic_similarity(reference: str, hypothesis: str) -> dict:
    """
    Compute semantic similarity using Sentence Transformers.

    Args:
        reference (str): Ground truth transcript (Audio)
        hypothesis (str): Predicted transcript (Video)

    Returns:
        dict:
        {
            "semantic_similarity": float
        }
    """

    reference = reference.strip()
    hypothesis = hypothesis.strip()

    # Handle empty inputs
    if not reference and not hypothesis:
        return {
            "semantic_similarity": 100.0
        }

    if not reference or not hypothesis:
        return {
            "semantic_similarity": 0.0
        }

    embeddings = model.encode(
        [reference, hypothesis],
        convert_to_numpy=True
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    similarity = similarity * 100

    return {
        "semantic_similarity": round(float(similarity), 2)
    }