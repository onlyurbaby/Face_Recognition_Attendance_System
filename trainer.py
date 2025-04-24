import cv2
import os
import numpy as np
from PIL import Image

# Path setup
dataset_path = "dataset"
trainer_path = "trained_model"
os.makedirs(trainer_path, exist_ok=True)

# Haarcascade (for verification)
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to get images & labels
def get_images_and_labels(path):
    face_samples = []
    ids = []
    label_map = {}  # name -> numeric ID
    current_id = 0

    for user_folder in os.listdir(path):
        user_path = os.path.join(path, user_folder)
        if not os.path.isdir(user_path):
            continue

        # Map name to numeric ID
        if user_folder not in label_map:
            label_map[user_folder] = current_id
            current_id += 1

        label_id = label_map[user_folder]

        for image_file in os.listdir(user_path):
            img_path = os.path.join(user_path, image_file)
            pil_image = Image.open(img_path).convert("L")  # Grayscale
            img_np = np.array(pil_image, 'uint8')

            faces = face_cascade.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(label_id)

    return face_samples, ids, label_map

print("[INFO] Training faces. This may take a while...")
faces, ids, label_map = get_images_and_labels(dataset_path)

recognizer.train(faces, np.array(ids))

# Save trained model
recognizer.save(f"{trainer_path}/trainer.yml")

# Save label map (name-ID)
with open(f"{trainer_path}/labels.txt", "w") as f:
    for name, id_ in label_map.items():
        f.write(f"{id_}:{name}\n")

print("[INFO] Training complete. Model and labels saved.")
