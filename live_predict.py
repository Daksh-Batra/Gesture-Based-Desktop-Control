import math
import cv2
import numpy as np
import mediapipe as mp
import joblib


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

model = joblib.load("model.pkl")
le = joblib.load("labels.pkl")

cam = cv2.VideoCapture(0)

def norm_features(hand_landmarks):
    features = np.array([[lm.x,lm.y,lm.z]for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = features[0]
    features = features - wrist

    mid = features[9]
    scale = math.sqrt(mid[0]**2+mid[1]**2+mid[2]**2)

    features = features/scale

    return features.flatten()

while True:
    success, frame = cam.read()
    if not success:
        break
    frame = cv2.flip(frame,1)
    color_cnv=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = hands.process(color_cnv)

    gesture="No Hand"

    if result.multi_hand_landmarks:
        hand_landmarks=result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

        features=norm_features(hand_landmarks)

        prob=model.predict_proba([features])[0]
        max_pred_ind=np.argmax(prob)

        gesture=le.inverse_transform([max_pred_ind])[0]
        pred=prob[max_pred_ind]

        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Confidence: {pred:.2f}", (10, 85),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Live Gesture Recognition", frame)

    key=cv2.waitKey(1)

    if key ==  ord("q"):
        break
