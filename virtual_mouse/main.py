import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import speech_recognition as sr
import threading
import platform

# ===== OS-Specific Volume Control =====
system_os = platform.system()
if system_os == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
elif system_os == "Darwin":
    import subprocess

def set_volume(delta):
    if system_os == "Windows":
        current = volume.GetMasterVolumeLevelScalar()
        new_level = max(0.0, min(1.0, current + delta / 100.0))
        volume.SetMasterVolumeLevelScalar(new_level, None)
    elif system_os == "Darwin":
        subprocess.run(["osascript", "-e", f"set volume output volume {(delta)}"])
    else:
        print("Volume control not supported for this OS.")

# ===== Voice Command Thread =====
def voice_control():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
    while True:
        try:
            with mic as source:
                print("Listening for volume command...")
                audio = r.listen(source, phrase_time_limit=3)
            command = r.recognize_google(audio).lower()
            print(f"Voice Command: {command}")

            if "volume up" in command:
                try:
                    val = int(command.split()[-1])
                    set_volume(val)
                except:
                    set_volume(5)

            elif "volume down" in command:
                try:
                    val = int(command.split()[-1])
                    set_volume(-val)
                except:
                    set_volume(-5)

        except sr.UnknownValueError:
            pass
        except Exception as e:
            print("Voice error:", e)

threading.Thread(target=voice_control, daemon=True).start()

# ===== Hand Tracking =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_x, prev_y = 0, 0
smoothening = 3
speed_multiplier = 1.7
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Cartoon filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    img = cv2.bitwise_and(color, color, mask=edges)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, c = img.shape
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            if lm_list:
                # Mouse Movement
                x1, y1 = lm_list[8]   # Index tip
                x0, y0 = lm_list[4]   # Thumb tip
                screen_x = np.interp(x1, (0, w), (0, screen_w))
                screen_y = np.interp(y1, (0, h), (0, screen_h))
                curr_x = prev_x + ((screen_x - prev_x) * speed_multiplier) / smoothening
                curr_y = prev_y + ((screen_y - prev_y) * speed_multiplier) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Click Gestures
                if abs(x1 - x0) < 40 and abs(y1 - y0) < 40:
                    pyautogui.click()
                    pyautogui.sleep(0.2)

                # Finger State Detection
                fingers = []
                for id in range(1, 5):
                    fingers.append(lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1])

                # Scroll Up (Index + Middle up)
                if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                    pyautogui.scroll(50)

                # Scroll Down (Only Index up)
                if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3]:
                    pyautogui.scroll(-50)

    cv2.imshow("Virtual Mouse + Scroll + Voice Volume", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
