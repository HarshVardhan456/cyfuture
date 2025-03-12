import cv2
import os

def collect_faces(person_name, save_path="dataset"):
    """Capture and save face images for a person."""
    person_folder = os.path.join(save_path, person_name)
    os.makedirs(person_folder, exist_ok=True)

    # Count existing images to avoid overwriting
    existing_images = len([f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    count = existing_images  # Continue numbering

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        cv2.imshow("Collecting Faces - Press 's' to Save, 'q' to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  
            count += 1
            img_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"✅ Saved: {img_path}")

        elif key == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter person's name: ").strip()
    if person_name:
        collect_faces(person_name)
    else:
        print("❌ Error: No name entered.")
