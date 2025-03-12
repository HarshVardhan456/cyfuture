import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet

# Load FaceNet model
embedder = FaceNet()

# Load stored face embeddings
EMBEDDINGS_FILE = "embeddings.pkl"

try:
    with open(EMBEDDINGS_FILE, "rb") as f:
        saved_embeddings = pickle.load(f)
    print("✅ Embeddings loaded successfully!")
except FileNotFoundError:
    print("❌ Error: embeddings.pkl not found! Train and save embeddings first.")
    saved_embeddings = {}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_embedding_from_frame(frame):
    """Detect face, crop it, and extract its embedding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None  # No face detected

    x, y, w, h = faces[0]  # Take the first detected face
    face = frame[y:y+h, x:x+w]  # Crop face
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))  # FaceNet requires 160x160 images
    face = np.expand_dims(face, axis=0)

    return embedder.embeddings(face)[0]  # Extract the 128-dimensional embedding

def recognize_face(frame):
    """Compare a captured face with stored embeddings and find the closest match."""
    test_embedding = get_embedding_from_frame(frame)
    if test_embedding is None:
        return "No Face Detected", 0.0
    
    best_match = "Unknown"
    best_score = float("-inf")

    for person, embeddings in saved_embeddings.items():
        for emb in embeddings:
            if emb is None or emb.ndim != 1:
                continue
            
            similarity = 1 - cosine(test_embedding, emb)
            if similarity > best_score:
                best_score = similarity
                best_match = person

    return (best_match, best_score) if best_score > 0.5 else ("Unknown", best_score)

# Start real-time face recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Recognize face in the frame
    name, score = recognize_face(frame)
    
    # Display results on the video feed
    cv2.putText(frame, f"{name} ({score:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Face Recognition", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
