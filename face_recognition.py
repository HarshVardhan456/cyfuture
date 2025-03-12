import numpy as np
from scipy.spatial.distance import cosine  
from face_embedding import get_embedding  

def recognize_face_against_dataset(image_path, saved_embeddings):
    """Compare a given image with all stored embeddings and find the closest match."""
    test_embedding = get_embedding(image_path)

    # Ensure embedding is valid
    if test_embedding is None or not isinstance(test_embedding, np.ndarray) or test_embedding.ndim != 1:
        return "Error: Invalid embedding for given image", 0.0

    best_match = None
    best_score = 0.0

    for person, embeddings in saved_embeddings.items():
        for emb in embeddings:
            # Ensure stored embedding is also valid
            if emb is None or not isinstance(emb, np.ndarray) or emb.ndim != 1:
                print(f"Warning: Skipping invalid embedding for {person}")
                continue

            similarity = 1 - cosine(test_embedding, emb)

            if similarity > best_score:
                best_score = similarity
                best_match = person

    if best_match and best_score > 0.5:
        return f"Match Found: {best_match}", best_score
    else:
        return "No match found in dataset", best_score
