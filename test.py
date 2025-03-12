from face_detection import detect_faces  
from face_recognition import recognize_faces  
 
detect_faces("dataset/person1.jpg")  
detect_faces("dataset/person2.jpg")   
result, score = recognize_faces("dataset/person1.jpg", "dataset/person2.jpg")  
print(f"Recognition Result: {result} (Score: {round(score, 2)})")  
