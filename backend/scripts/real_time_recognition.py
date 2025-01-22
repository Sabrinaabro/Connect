import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r"D:\Work\FYP\backend\models\hand_gesture_model.pkl")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def recognize_gestures():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks = np.array(landmarks).reshape(1, -1)

                    prediction = model.predict(landmarks)
                    cv2.putText(image, f"Sign: {prediction[0]}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Real-Time Recognition', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_gestures()
