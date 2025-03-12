import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

embedder = FaceNet()
EMBEDDINGS_FILE = "embeddings.pkl"
THRESHOLD = 0.5  

with open(EMBEDDINGS_FILE, "rb") as f:
    stored_embeddings = pickle.load(f)

def get_embedding_from_frame(frame):
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

def recognize_face(frame):
    emb = get_embedding_from_frame(frame)
    if emb is None:
        return "No Face Detected", 0.0

    best_match = "Unknown"
    best_score = 0.0

    for person_name, stored_emb in stored_embeddings.items():
        similarity = 1 - cosine(emb, stored_emb)  
        if similarity > best_score and similarity > THRESHOLD:
            best_match = person_name
            best_score = similarity

    return best_match, best_score


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_name, score = recognize_face(frame)
    text = f"{person_name} ({round(score, 2)})"
    
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Recognition - Press 'q' to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
