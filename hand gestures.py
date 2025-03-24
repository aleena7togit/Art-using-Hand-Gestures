import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
draw_color = (255, 255, 255)  # Color for drawing
erase_color = (0, 0, 0)       # Color for erasing

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Initialize variables
prev_x, prev_y = 0, 0
is_drawing = True
is_erasing = False
reset_start_time = None
exit_start_time = None
fist_start_time = None
pause_start_time = None
is_paused = False

# Smoothing positions using deque
position_queue = deque(maxlen=5)

# Function to draw lines on canvas
def draw_line(canvas, start, end, color, thickness=2):
    cv2.line(canvas, start, end, color, thickness)

# Function to erase drawn areas on canvas
def erase_area(canvas, center, radius, color):
    cv2.circle(canvas, center, radius, color, -1)

# Function to detect if the hand is in a fist position
def is_fist(landmarks):
    for finger_tip, finger_mcp in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    ]:
        tip = landmarks[finger_tip]
        mcp = landmarks[finger_mcp]
        distance = np.linalg.norm(np.array([tip.x, tip.y]) - np.array([mcp.x, mcp.y]))
        if distance > 0.05:
            return False
    return True

# Function to detect if the hand is open
def is_open_hand(landmarks):
    for finger_tip, finger_mcp in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    ]:
        tip = landmarks[finger_tip]
        mcp = landmarks[finger_mcp]
        distance = np.linalg.norm(np.array([tip.x, tip.y]) - np.array([mcp.x, mcp.y]))
        if distance < 0.1:
            return False
    return True

# Function to detect if both hands are open for a high-five gesture
def is_high_five(results):
    if len(results.multi_hand_landmarks) == 2:
        hand1_open = is_open_hand(results.multi_hand_landmarks[0].landmark)
        hand2_open = is_open_hand(results.multi_hand_landmarks[1].landmark)
        return hand1_open and hand2_open
    return False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hands_count = len(results.multi_hand_landmarks)

        # Handle gestures based on the number of hands detected
        if hands_count == 1:  # Single hand detected
            landmarks = results.multi_hand_landmarks[0].landmark

            # Single-hand high-five to toggle pause/resume
            if is_open_hand(landmarks):
                if pause_start_time is None:
                    pause_start_time = time.time()
                elif time.time() - pause_start_time > 1:
                    is_paused = not is_paused
                    pause_start_time = None
                    print("Paused" if is_paused else "Resumed")
            else:
                pause_start_time = None

            if is_fist(landmarks):  # Single-hand fist for erasing
                is_erasing = True
                is_drawing = False
            else:
                is_erasing = False
                is_drawing = True

            # Get the coordinates of the index finger tip
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Smooth positions
            position_queue.append((index_tip_x, index_tip_y))
            smoothed_position = np.mean(position_queue, axis=0).astype(int)
            index_tip_x, index_tip_y = smoothed_position

            if not is_paused:
                if is_erasing:
                    erase_area(canvas, (index_tip_x, index_tip_y), 30, erase_color)
                elif is_drawing:
                    if prev_x != 0 and prev_y != 0:
                        draw_line(canvas, (prev_x, prev_y), (index_tip_x, index_tip_y), draw_color)
                    prev_x, prev_y = index_tip_x, index_tip_y

        elif hands_count == 2:  # Two hands detected
            landmarks1 = results.multi_hand_landmarks[0].landmark
            landmarks2 = results.multi_hand_landmarks[1].landmark

            if is_fist(landmarks1) and is_fist(landmarks2):  # Dual-hand fist for reset
                if fist_start_time is None:
                    fist_start_time = time.time()
                elif time.time() - fist_start_time > 2:
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                    print("Canvas Reset!")
                    fist_start_time = None
            else:
                fist_start_time = None

        # Check for high-five gesture to exit
        if is_high_five(results):
            if exit_start_time is None:
                exit_start_time = time.time()
            elif time.time() - exit_start_time > 2:
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        else:
            exit_start_time = None

    # Display visual feedback
    if fist_start_time:
        cv2.putText(frame, "RESETTING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif is_paused:
        cv2.putText(frame, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if exit_start_time:
        cv2.putText(frame, "EXITING", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame and canvas
    cv2.imshow('Frame', frame)
    cv2.imshow('Canvas', canvas)

    # Exit the program on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()