import cv2
import numpy as np
import mediapipe as mp
import math
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def norm_features(hand_landmarks):
    features = np.array([[lm.x,lm.y,lm.z]for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = features[0]
    features = features - wrist

    mid = features[9]
    scale = math.sqrt(mid[0]**2+mid[1]**2+mid[2]**2)

    features = features/scale

    return features.flatten()

cam = cv2.VideoCapture(0)

gesture_name=input("Enter gesture name: (e.g. \"palm\") ").strip().upper()
os.makedirs("dataset",exist_ok=True)
save_path=os.path.join("dataset",f"{gesture_name}.npy")

if os.path.exists(save_path):
    old_data = np.load(save_path)

sample=[]
rec= False
target=200

print("1) Show your gesture in camera")
print("2) Press S to start recording")
print("3) Press Q to quit\n")

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    color_convert = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(color_convert)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]  #reads only first hand, cant switch to 2 hands with this method

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)
            features.append(lm.z)

        features = np.array(features)
        n_features=norm_features(hand_landmarks)

        if rec and len(sample)<target:
            sample.append(n_features)

        status= "Recording" if rec else "IDLE"

        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Status: {status}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Samples: {len(sample)}/{target}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText( frame ,f"Feature length: {len(features)}", (10, 155) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0, 255, 0) , 2)

    cv2.imshow("Hand Landmarks", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if len(sample)>=target:
        break

    if key == ord("s"):
        rec= True
    
if len(sample)>=0:
    sample=np.array(sample, dtype=np.float32)
    np.save(save_path, sample)
    print(f"\nSaved {len(sample)} samples to: {save_path}")
    print("Saved shape:", sample.shape)
else:
    print("\nNo samples collected.")