import os
import cv2
import numpy as np
from keras_facenet import FaceNet  

# Load FaceNet model  
embedder = FaceNet()  

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

# Path to dataset
dataset_path = "dataset/harsh/"  # Change folder name to process different people

# Iterate through all images in the folder
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)

    # Ensure the file is an image before processing
    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        embedding = get_embedding(image_path)

        if embedding is not None:
            print(f"✅ Processed {image_name}")
        else:
            print(f"❌ Skipped {image_name}")
