import cv2
import numpy as np
import os
from datetime import datetime

# Load face recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model/trainer.yml")
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Load label map
label_map = {}
with open("trained_model/labels.txt") as f:
    for line in f:
        id_, name = line.strip().split(":")
        label_map[int(id_)] = name

# Attendance folder setup
attendance_folder = "Attendance"
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

date_str = datetime.now().strftime("%Y-%m-%d")
attendance_file = os.path.join(attendance_folder, f"{date_str}.csv")

# Create CSV file if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time,Confidence\n")

# Avoid duplicate entries
marked_names = set()
with open(attendance_file, "r") as f:
    lines = f.readlines()[1:]  # Skip header
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 1:
            marked_names.add(parts[0])

# Start camera
cap = cv2.VideoCapture(0)
print("[INFO] Webcam started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face)

        if conf < 60:
            name = label_map.get(id_, "Unknown")
            color = (0, 255, 0)

            if name not in marked_names:
                time_now = datetime.now().strftime("%H:%M:%S")
                with open(attendance_file, "a") as f:
                    f.write(f"{name},{date_str},{time_now},{round(conf, 2)}\n")
                marked_names.add(name)
                print(f"[ATTENDANCE] {name} marked at {time_now}")
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition - Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Mark absentees before closing
for id_, name in label_map.items():
    if name not in marked_names:
        time_now = "-"
        with open(attendance_file, "a") as f:
            f.write(f"{name},{date_str},{time_now},ABSENT\n")
        print(f"[ABSENT] {name} marked as absent.")



cap.release()
cv2.destroyAllWindows()
