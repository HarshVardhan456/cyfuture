import pickle
import numpy as np
import cv2
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

embedder = FaceNet()


with open("face_embeddings.pkl", "rb") as f:
    stored_embeddings = pickle.load(f)

THRESHOLD = 0.5  


def recognize_face(test_image_path):
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    test_embedding = embedder.embeddings(img)[0]

    best_match = None
    best_score = float("-inf")

    for person_name, stored_embedding in stored_embeddings.items():
        similarity = 1 - cosine(test_embedding, stored_embedding)
        if similarity > THRESHOLD and similarity > best_score:
            best_match = person_name
            best_score = similarity

    if best_match:
        return f"✅ Recognized as {best_match} (Score: {round(best_score, 2)})"
    else:
        return "❌ Unknown face"

if __name__ == "__main__":
    result = recognize_face("test_face.jpg")
    print(result)
