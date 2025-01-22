import cv2
import mediapipe as mp
import csv
import os

# MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function to collect data from static images
def collect_static_image_data(image_files, label, output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    data = []  # To store the landmark data

    # Initialize MediaPipe hands
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(image_files):
            # Read and process the image
            image = cv2.flip(cv2.imread(file), 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # If landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])  # Collect x, y, z for each landmark
                    data.append([label] + landmarks)  # Add label and landmarks to data

            # Optional: Visualize the landmarks (for debugging)
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            cv2.imwrite(f"/tmp/annotated_image_{label}_{idx}.png", cv2.flip(annotated_image, 1))

    # Save collected data to CSV
    with open(f"{output_dir}/{label}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + [f"x{i}" for i in range(21)] +
                        [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])
        writer.writerows(data)

# Function to collect data from webcam input
def collect_webcam_data(label, output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    data = []  # To store the landmark data

    # Initialize MediaPipe hands
    with mp_hands.Hands(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Process the image
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Draw landmarks if hands are detected
            image.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])  # Collect x, y, z for each landmark
                    data.append([label] + landmarks)  # Add label and landmarks to data

                    # Draw landmarks on the image
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Show the processed image with landmarks
            cv2.imshow('MediaPipe Hands', cv2.flip(image_bgr, 1))

            # Stop collection when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()

    # Save collected data to CSV after closing the webcam
    with open(f"{output_dir}/{label}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + [f"x{i}" for i in range(21)] +
                        [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])
        writer.writerows(data)

# Example usage:
# Collect data for "thumbs_up" from static images in a directory
image_files = ["images/up.jpeg", "images/thumbsup.jpeg"]  # Add paths to your static images
collect_static_image_data(image_files, label="thumbs_up")

# Collect data for "peace_sign" using webcam
collect_webcam_data(label="peace_sign")
