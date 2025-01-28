import cv2
import mediapipe as mp
import csv
import os

# MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function to collect data from static images
# Modified function to collect static image data for multiple labels
def collect_multiple_static_image_data(label_image_map, output_dir="backend/datasets"):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory

    # Initialize MediaPipe hands
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5) as hands:
        for label, image_files in label_image_map.items():
            data = []  # Store landmarks for this label
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

                # Optional: Visualize landmarks
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

            # Save collected data for this label to CSV
            with open(f"{output_dir}/{label}.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["label"] + [f"x{i}" for i in range(21)] +
                                [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])
                writer.writerows(data)


# Modified function to collect webcam data for multiple labels
def collect_multiple_webcam_data(labels, output_dir="backend/datasets"):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    data = {label: [] for label in labels}  # Initialize storage for each label

    # Instructions for the user
    print("Press the following keys to switch between labels:")
    for idx, label in enumerate(labels):
        print(f"{idx + 1}: {label}")
    print("Press 'q' to quit.")

    current_label = labels[0]  # Default to the first label

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

            # Switch labels based on key input
            key = cv2.waitKey(5) & 0xFF
            if ord('1') <= key <= ord(str(len(labels))):  # Switch label by number
                current_label = labels[key - ord('1')]
                print(f"Switched to label: {current_label}")
            elif key == ord('q'):  # Quit
                break

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
                    data[current_label].append([current_label] + landmarks)

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

        cap.release()
        cv2.destroyAllWindows()

    # Save collected data for each label to separate CSV files
    for label, landmarks in data.items():
        with open(f"{output_dir}/{label}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["label"] + [f"x{i}" for i in range(21)] +
                            [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])
            writer.writerows(landmarks)


# Example Usage:

# Collect multiple static image data
# image_files_by_label = {
#     "thumbs_up": ["images/up1.jpeg", "images/up2.jpeg"],
#     "peace_sign": ["images/peace1.jpeg", "images/peace2.jpeg"],
#     "fist": ["images/fist1.jpeg", "images/fist2.jpeg"]
# }
# collect_multiple_static_image_data(image_files_by_label)

# Collect multiple webcam data
collect_multiple_webcam_data(labels=["thumbs_up", "peace_sign", "fist"])

