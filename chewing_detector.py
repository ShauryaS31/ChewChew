import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Initialize face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def calculate_jaw_distance(landmarks):
    # Get points for upper and lower jaw
    upper_jaw = np.array([landmarks.part(62).x, landmarks.part(62).y])
    lower_jaw = np.array([landmarks.part(66).x, landmarks.part(66).y])
    
    # Calculate Euclidean distance
    return distance.euclidean(upper_jaw, lower_jaw)

def detect_chewing():
    cap = cv2.VideoCapture(0)
    
    # Variables for chewing detection
    jaw_distances = []
    chew_count = 0
    threshold = 2.0  # Adjust this value based on testing
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Calculate jaw distance
            jaw_dist = calculate_jaw_distance(landmarks)
            jaw_distances.append(jaw_dist)
            
            # Keep only last 10 measurements
            if len(jaw_distances) > 10:
                jaw_distances.pop(0)
                
                # Detect chewing motion
                if len(jaw_distances) >= 3:
                    if (max(jaw_distances[-3:]) - min(jaw_distances[-3:])) > threshold:
                        chew_count += 1
            
            # Draw facial landmarks
            for n in range(60, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
        # Display chew count
        cv2.putText(frame, f'Chews: {chew_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Chewing Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_chewing() 