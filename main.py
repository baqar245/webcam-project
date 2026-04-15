import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
prev_scroll_y = 0

clicking = False
last_click_time = 0

smoothening = 7  # higher = smoother
frame_reduction = 100  # reduces jitter area

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    cam_h, cam_w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw control box
    cv2.rectangle(img, (frame_reduction, frame_reduction),
                  (cam_w - frame_reduction, cam_h - frame_reduction),
                  (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # ===== ESP STYLE BONES =====
            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,200,255), thickness=2)
            )

            # LOCK TEXT
            cv2.putText(img, "LOCKED", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

            # Get landmarks
            lm = handLms.landmark

            index = lm[8]
            middle = lm[12]
            thumb = lm[4]

            # Convert to pixel coords
            x = int(index.x * cam_w)
            y = int(index.y * cam_h)

            # Check fingers up (control mode: index + middle)
            fingers_up = (index.y < lm[6].y) and (middle.y < lm[10].y)

            if fingers_up:
                # Map inside box
                screen_x = np.interp(x,
                                    (frame_reduction, cam_w - frame_reduction),
                                    (0, screen_w))
                screen_y = np.interp(y,
                                    (frame_reduction, cam_h - frame_reduction),
                                    (0, screen_h))

                # Smooth movement
                smooth_x = prev_x + (screen_x - prev_x) / smoothening
                smooth_y = prev_y + (screen_y - prev_y) / smoothening

                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y

                # Draw pointer
                cv2.circle(img, (x, y), 10, (255, 0, 255), -1)

            # ===== PINCH CLICK =====
            distance = math.hypot(
                thumb.x - index.x,
                thumb.y - index.y
            )

            if distance < 0.03:
                if not clicking and time.time() - last_click_time > 0.3:
                    pyautogui.mouseDown()
                    clicking = True
                    last_click_time = time.time()

                    cv2.putText(img, "CLICK", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                if clicking:
                    pyautogui.mouseUp()
                    clicking = False

            # ===== SCROLL =====
            if prev_scroll_y != 0:
                if index.y < prev_scroll_y - 0.02:
                    pyautogui.scroll(60)
                elif index.y > prev_scroll_y + 0.02:
                    pyautogui.scroll(-60)

            prev_scroll_y = index.y

    else:
        cv2.putText(img, "NO HAND", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()