import cv2
import os
from tkinter import simpledialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
from tkinter import ttk, Scrollbar, RIGHT, Y
import customtkinter as ctk

# Setup CTk appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# --- Core Functions ---

def start_attendance():
    progressbar.set(0.2)
    app.update_idletasks()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trained_model/trainer.yml")
    face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

    label_map = {}
    with open("trained_model/labels.txt") as f:
        for line in f:
            id_, name = line.strip().split(",")
            label_map[int(id_)] = name

    date_str = datetime.now().strftime("%Y-%m-%d")
    attendance_folder = "Attendance"
    os.makedirs(attendance_folder, exist_ok=True)
    attendance_file = os.path.join(attendance_folder, f"{date_str}.csv")

    if not os.path.exists(attendance_file):
        with open(attendance_file, "w") as f:
            f.write("Name,Date,Time,Confidence\n")

    marked_names = set()
    with open(attendance_file, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 1:
                marked_names.add(parts[0])

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face)

            if conf < 60:
                name = label_map.get(id_, "Unknown")
                if name not in marked_names:
                    time_now = datetime.now().strftime("%H:%M:%S")
                    with open(attendance_file, "a") as f:
                        f.write(f"{name},{date_str},{time_now},{round(conf, 2)}\n")
                    marked_names.add(name)
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for id_, name in label_map.items():
        if name not in marked_names:
            with open(attendance_file, "a") as f:
                f.write(f"{name},{date_str},-,ABSENT\n")

    cap.release()
    cv2.destroyAllWindows()
    progressbar.set(1.0)


def show_attendance():
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = f"Attendance/{date_str}.csv"
        df = pd.read_csv(file_path)

        if df.empty:
            messagebox.showinfo("Info", "Attendance file is empty!")
            return

        top = ctk.CTkToplevel()
        top.title(f"Attendance Sheet - {date_str}")
        top.geometry("700x400")

        tree = ttk.Treeview(top, columns=list(df.columns), show='headings')
        tree.pack(side="left", fill="both", expand=True)

        scrollbar = Scrollbar(top, orient="vertical", command=tree.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        tree.configure(yscrollcommand=scrollbar.set)

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    except FileNotFoundError:
        messagebox.showerror("Error", "Attendance file not found!")


def add_person():
    name = simpledialog.askstring("Input", "Enter person's name:")
    if not name:
        messagebox.showerror("Error", "Name cannot be empty!")
        return

    dataset_path = os.path.join("Dataset", name)
    os.makedirs(dataset_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    haar_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            file_path = os.path.join(dataset_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Collecting Images", frame)
        if cv2.waitKey(1) == 27 or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Images saved for {name}")


def train_model():
    dataset_path = "Dataset"
    model_path = "trained_model"
    os.makedirs(model_path, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    haar_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

    label_ids = {}
    current_id = 0
    x_train = []
    y_labels = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces = haar_cascade.detectMultiScale(img, 1.3, 5)
                for (x, y, w, h) in faces:
                    roi = img[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    if not x_train:
        messagebox.showerror("Error", "No training data found!")
        return

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(os.path.join(model_path, "trainer.yml"))
    with open(os.path.join(model_path, "labels.txt"), "w") as f:
        for name, id_ in label_ids.items():
            f.write(f"{id_},{name}\n")
    messagebox.showinfo("Success", "Model trained successfully!")


# --- GUI Setup ---
app = ctk.CTk()
app.geometry("900x600")
app.title("🧠 Face Recognition Attendance System")

sidebar = ctk.CTkFrame(app, width=200, corner_radius=0)
sidebar.pack(side="left", fill="y")

main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(padx=20, pady=20, fill="both", expand=True)

ctk.CTkLabel(sidebar, text="Menu", font=("Segoe UI", 18, "bold")).pack(pady=25)
ctk.CTkButton(sidebar, text="🎯 Start Attendance", command=start_attendance).pack(pady=10)
ctk.CTkButton(sidebar, text="📄 View Attendance", command=show_attendance).pack(pady=10)
ctk.CTkButton(sidebar, text="➕ Add New Person", command=add_person).pack(pady=10)
ctk.CTkButton(sidebar, text="🧠 Train Model", command=train_model).pack(pady=10)

progressbar = ctk.CTkProgressBar(sidebar, orientation="horizontal")
progressbar.set(0)
progressbar.pack(pady=30, fill="x", padx=20)

main_label = ctk.CTkLabel(main_frame, text="Welcome to the Face Recognition System", font=("Segoe UI", 24))
main_label.pack(expand=True)

footer = ctk.CTkLabel(app, text="© 2025 Project by YOU 😎", font=("Segoe UI", 12))
footer.pack(side="bottom", pady=10)

app.mainloop()
