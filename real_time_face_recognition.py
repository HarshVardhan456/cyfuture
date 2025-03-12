import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

embedder = FaceNet()
EMBEDDINGS_FILE = "embeddings.pkl"
THRESHOLD = 0.5

def load_embeddings():
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Embeddings file not found!")
        return {}

stored_embeddings = load_embeddings()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_embedding_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return None  # No face detected
    
    x, y, w, h = faces[0]  # Use the first detected face
    face = frame[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

def recognize_face(frame):
    emb = get_embedding_from_frame(frame)
    if emb is None:
        return "No Face Detected", 0.0

    best_match = "Unknown"
    best_score = 0.0

    for person_name, stored_emb_list in stored_embeddings.items():
        for stored_emb in stored_emb_list:
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
