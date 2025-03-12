import cv2
import os

def collect_faces(person_name, save_path="dataset"):
    person_folder = os.path.join(save_path, person_name)
    os.makedirs(person_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Collecting Faces - Press 's' to Save, 'q' to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save image
            count += 1
            img_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")

        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter person's name: ")
    collect_faces(person_name)
