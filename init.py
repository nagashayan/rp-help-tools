import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam input stream.
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands with configurations.
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame color from BGR to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hands.
        results = hands.process(frame_rgb)

        # Convert frame back to BGR for OpenCV visualization.
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw hand annotations if any are found.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame.
        cv2.imshow('MediaPipe Hands', frame_bgr)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
