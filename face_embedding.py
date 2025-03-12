import os
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet

# Load FaceNet model
embedder = FaceNet()

DATASET_PATH = "dataset/"
EMBEDDINGS_FILE = "embeddings.pkl"

def get_embedding(image_path):
    """Extract face embedding for an image."""
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load {image_path} (corrupt or unsupported format).")
        return None

    face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))  # FaceNet requires 160x160 images
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]  # Extract the 128-dimensional embedding

# Dictionary to store embeddings for all people
embeddings_dict = {}

# Iterate through all people in the dataset folder
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)
    
    if os.path.isdir(person_folder):  # Ensure it's a folder
        embeddings_dict[person_name] = []

        # Process each image in the person's folder
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                embedding = get_embedding(image_path)

                if embedding is not None:
                    embeddings_dict[person_name].append(embedding)
                    print(f"✅ Processed {image_name} for {person_name}")
                else:
                    print(f"❌ Skipped {image_name} for {person_name}")

# Save embeddings to a file
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(embeddings_dict, f)

print("\n✅ All embeddings saved successfully!")
