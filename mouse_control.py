import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Get screen dimensions for scalpy -3.10 mouse_control.pying
screen_w, screen_h = pyautogui.size()

# Mouse control flag
is_clicking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb tip and index finger tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized coordinates to screen size
            x_index = int(index_tip.x * screen_w)
            y_index = int(index_tip.y * screen_h)

            # Move mouse
            pyautogui.moveTo(x_index, y_index, duration=0.1)

            # Calculate distance between thumb and index finger
            dist = ((thumb_tip.x - index_tip.x) ** 2 + 
                    (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # Click if fingers are close
            if dist < 0.05 and not is_clicking:
                pyautogui.click()
                is_clicking = True
            elif dist >= 0.05:
                is_clicking = False

    # Show video
    cv2.imshow('Hand Gesture Mouse Control', frame)

    # Exit on ESC key
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()