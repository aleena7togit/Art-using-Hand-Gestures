import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw_color = (255, 255, 255)  # Color for drawing
erase_color = (0, 0, 0)        # Color for erasing
pointer_color = (0, 255, 0)    # Color for the finger pointer

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Stack to store canvas states for undo
undo_stack = [canvas.copy()]

# Initialize previous position variables
prev_x, prev_y = 0, 0

# Pause state
is_paused = False

# Function to draw lines on canvas
def draw_line(canvas, start, end, color, thickness=2):
    cv2.line(canvas, start, end, color, thickness)

# Function to erase drawn areas on canvas
def erase_area(canvas, center, radius, color):
    cv2.circle(canvas, center, radius, color, -1)

# Main loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame_rgb)

    # Draw landmarks and get hand positions
    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get landmarks of the hand
            landmarks = hand_landmarks.landmark
            
            # Identify if the hand is left or right
            hand_label = results.multi_handedness[hand_index].classification[0].label
            
            # Get the coordinates of the index finger tip
            index_tip_x, index_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            
            # Draw the pointer at the index finger tip
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, pointer_color, -1)
            
            # Get the coordinates of the palm (center of the hand)
            palm_x, palm_y = int(landmarks[mp_hands.HandLandmark.WRIST].x * frame.shape[1]), int(landmarks[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
            
            # Perform actions only if not paused
            if not is_paused:
                # Erase if the left hand is detected
                if hand_label == "Left":
                    erase_area(canvas, (palm_x, palm_y), 50, erase_color)
                # Draw if the right hand is detected
                elif hand_label == "Right":
                    if prev_x != 0 and prev_y != 0:
                        draw_line(canvas, (prev_x, prev_y), (index_tip_x, index_tip_y), draw_color)
                    prev_x, prev_y = index_tip_x, index_tip_y
            else:
                # Reset previous positions when paused
                prev_x, prev_y = 0, 0

    # Display frame and canvas
    cv2.imshow('Frame', frame)
    cv2.imshow('Canvas', canvas)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        # Toggle pause state
        is_paused = not is_paused
        print("Drawing Paused" if is_paused else "Drawing Resumed")

# Release resources
cap.release()
cv2.destroyAllWindows()
