import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np 
import joblib
import math
import keyboard
import time
import pyautogui
from collections import deque, Counter

debug_last=0
def norm_features(hand_landmarks):
    features = np.array([[lm.x,lm.y,lm.z]for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = features[0]
    features = features - wrist

    mid = features[9]
    scale = math.sqrt(mid[0]**2+mid[1]**2+mid[2]**2)

    features = features/scale

    return features.flatten()

gesture_assign={"PALM":"play_pause",
                "FIST":"alt_tab",
                "PEACE":"next track",
                "4-FINGER":"new_tab"}

def execution(gesture):
    if gesture == "play_pause":
        keyboard.send("play/pause media")
    elif gesture == "alt_tab":
        pyautogui.keyDown("alt")
        pyautogui.press("tab")
        pyautogui.keyUp("alt")
    elif gesture == "next track":
        keyboard.send("next track")
    elif gesture=="new_tab":
        #print("TRIGGER:", smo_ges, "->", action)
        #pyautogui.keyDown("ctrl")
        #pyautogui.press("t")
        #pyautogui.keyUp("ctrl")
        keyboard.release("ctrl")
        keyboard.release("shift")
        keyboard.release("alt")
        time.sleep(0.05)
        keyboard.send("ctrl+t")
        print("TRIGGER:", smo_ges, "->", action)
        
    else:
        history.clear()
        hand_since=None


model = joblib.load("model.pkl")
le=joblib.load("labels.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands=mp_hands.Hands(static_image_mode=False,
                     max_num_hands=1,
                     min_detection_confidence=0.6,
                     min_tracking_confidence=0.6)

cam = cv2.VideoCapture(0)

history=deque(maxlen=10)

cooldown=5.0
last_trigger=0

hand_since=None
hand_present_threshold=4.0

confidence_threshold=0.75

while True:
    success,frame= cam.read()
    if not success:
        break
    frame=cv2.flip(frame,1)
    color=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(color)
    raw_ges="NO HAND"
    smo_ges="NO HAND"
    confidence=0.0
    if result.multi_hand_landmarks:
        now=time.time()
        if hand_since is None:
            hand_since=now
        hand_landmarks=result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
        features=norm_features(hand_landmarks)

        prob=model.predict_proba([features])[0]
        max_prob_indx=np.argmax(prob)

        confidence=prob[max_prob_indx]
        raw_ges=le.inverse_transform([max_prob_indx])[0]
        raw_ges = raw_ges.strip().upper().replace(".", "")

        history.append(raw_ges)
        smo_ges=Counter(history).most_common(1)[0][0]  #finds one most common element as (["palm",5], more if needed and entered 2) so 2 [0] needed
        cur_time=time.time()

        if confidence >= confidence_threshold:
            if smo_ges in gesture_assign and (cur_time - last_trigger) > cooldown and (now - hand_since) > hand_present_threshold:
                action = gesture_assign[smo_ges]
                execution(action)
                last_trigger = cur_time
    else:
        history.clear()
    cv2.putText(frame, f"Raw: {raw_ges}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"Smooth: {smo_ges}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Gesture Control",frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break