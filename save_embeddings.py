from keras_facenet import FaceNet
import cv2
import numpy as np
import os
import pickle

embedder = FaceNet()
DATASET_PATH = "dataset/"
EMBEDDINGS_FILE = "embeddings.pkl"

def get_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))  
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]  

def process_dataset():
    embeddings = {}
    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if os.path.isdir(person_folder):
            person_embeddings = []
            for file in os.listdir(person_folder):
                if file.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_folder, file)
                    emb = get_embedding(image_path)
                    if emb is not None:
                        person_embeddings.append(emb)

            if person_embeddings:
                embeddings[person_name] = np.mean(person_embeddings, axis=0)  

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    print("Embeddings saved successfully.")

if __name__ == "__main__":
    process_dataset()
