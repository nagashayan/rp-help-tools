import cv2
import mediapipe as mp

# Define a function to check if the hand is showing thumbs-up.
def is_thumb_up(landmarks, handedness):
    """
    Check if the gesture is a thumbs-up based on landmarks.
    
    Parameters:
    - landmarks: List of hand landmarks from Mediapipe.
    - handedness: 'Left' or 'Right' hand.
    
    Returns:
    - True if gesture is thumbs-up, False otherwise.
    """
    # Thumb landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    
    # Index finger landmarks
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Check if thumb is extended upwards:
    # - Thumb_tip should be above the other parts of the thumb (y-axis in image space)
    # - Thumb_tip should be to the left (for right hand) or right (for left hand) of the index finger
    if handedness == 'Right':
        return (thumb_tip.y < thumb_mcp.y and  # Thumb tip above MCP
                thumb_tip.x < index_tip.x)     # Thumb tip left of index finger (for right hand)
    else:
        return (thumb_tip.y < thumb_mcp.y and  # Thumb tip above MCP
                thumb_tip.x > index_tip.x)     # Thumb tip right of index finger (for left hand)


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
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                # Get the handedness (Left/Right) from Mediapipe results.
                handedness_label = hand_handedness.classification[0].label

                # Check for the thumbs-up gesture.
                if is_thumb_up(hand_landmarks.landmark, handedness_label):
                    # Display text if a thumbs-up is detected.
                    cv2.putText(frame_bgr, 'Thumbs Up!', (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame.
        cv2.imshow('MediaPipe Hands', frame_bgr)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
