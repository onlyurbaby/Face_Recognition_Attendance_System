import cv2
import os

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

def collect_faces(user_id):
    cam = cv2.VideoCapture(0)
    count = 0

    # Create folder to save face images
    save_path = f"dataset/{user_id}"
    os.makedirs(save_path, exist_ok=True)

    print("[INFO] Starting face data collection. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Collecting Faces - Press 'q' to Stop", frame)

        # Break on 'q' or after 30 images
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Collected {count} images for user: {user_id}")

if __name__ == "__main__":
    user_id = input("Enter User ID or Name: ")
    collect_faces(user_id)
