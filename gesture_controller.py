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
import threading

class Gesture_controller:
    def __init__(self):
        self.model = joblib.load("model.pkl")
        self.lb=joblib.load("labels.pkl")

        self.gesture_assn = {"PALM":"play_pause",
                            "FIST":"alt_tab",
                            "PEACE":"next track",
                            "4-FINGER":"new_tab"}
        
        self.history=deque(maxlen=10)
        self.cooldown = 3.0
        self.last_trigger = 0
        self.hand_since = None
        self.hand_present_threshold = 3.0

        self.confidence_threshold = 0.75
        self.current_gesture = "NO HAND"
        self.current_confidence=0.0
        self.thread= None
        self.running=False
        self.latest_frame=None
    def generate_frame(self):
        while True:
            if self.latest_frame is None:
                continue
            ret,buffer = cv2.imencode('.jpg',self.latest_frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                  b'Content - Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def norm_features(self,hand_landmarks):
        features = np.array([[lm.x,lm.y,lm.z]for lm in hand_landmarks.landmark], dtype=np.float32)
        wrist = features[0]
        features = features - wrist

        mid = features[9]
        scale = math.sqrt(mid[0]**2+mid[1]**2+mid[2]**2)

        features = features/scale

        return features.flatten()
    
    def execute(self,gesture):
        if gesture == "play_pause":
            keyboard.send("play/pause media")
        elif gesture == "alt_tab":
            pyautogui.keyDown("alt")
            pyautogui.press("tab")
            pyautogui.keyUp("alt")
        elif gesture == "next track":
            keyboard.send("next track")
        elif gesture=="new_tab":
            keyboard.release("ctrl")
            keyboard.release("shift")
            keyboard.release("alt")
            time.sleep(0.05)
            keyboard.send("ctrl+t")

    def start(self):
        if not self.running:
            self.running=True
            self.thread = threading.Thread(target=self.run_loop)
            self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def run_loop(self):
        mp_hands=mp.solutions.hands
        
        hands=mp_hands.Hands(static_image_mode=False,
                             max_num_hands=1,
                             min_detection_confidence=0.6,
                             min_tracking_confidence=0.6,)
        
        cam = cv2.VideoCapture(0)

        while self.running:
            success,frame=cam.read()
            if not success:
                break

            frame=cv2.flip(frame,1)
            color=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=hands.process(color)

            raw_ges="NO HAND"
            confidence=0.0

            if result.multi_hand_landmarks:
                now = time.time()

                if self.hand_since is None:
                    self.hand_since=now

                hand_landmarks=result.multi_hand_landmarks[0]
                features = self.norm_features(hand_landmarks)

                prob = self.model.predict_proba([features])[0]
                max_prob_indx = np.argmax(prob)
                confidence=prob[max_prob_indx]

                raw_ges=self.lb.inverse_transform([max_prob_indx])[0]
                raw_ges=raw_ges.strip().upper().replace(".","")

                self.history.append(raw_ges)
                smo_ges=Counter(self.history).most_common(1)[0][0]


                if confidence>=self.confidence_threshold and smo_ges in self.gesture_assn and (now - self.last_trigger)>=self.cooldown and (now - self.hand_since)>=self.hand_present_threshold:
                    action=self.gesture_assn[smo_ges]
                    self.execute(action)
                    self.last_trigger=now
                
                self.current_gesture=smo_ges
                self.current_confidence=confidence

            else:
                self.history.clear()
                self.hand_since = None
                self.current_gesture = "NO HAND"
                self.current_confidence = 0.0
            self.latest_frame=frame.copy()
        cam.release()
    def status(self):
        return{"Gesture":self.current_gesture,
               "Confidence":self.current_confidence,
               "Running":self.running}