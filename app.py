from gesture_controller import Gesture_controller
from flask import Flask,jsonify,render_template,Response,request
import json
import subprocess
import cv2
import sys
import os

app = Flask (__name__)  #create new web server

control = Gesture_controller()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(control.generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/gesture")
def gesture():
    files = os.listdir("dataset")

    gesture_L=[]
    for file in files:
        if file.endswith(".npy"):
            gesture_L.append(file.replace(".npy",""))
    return jsonify({"message":gesture_L})


@app.route("/collect",methods=["POST"])
def collect():
    label=request.json.get("label")

    if not label:
        return jsonify({"message":"Gesture Name Not Provided"})
    
    control.stop()
    
    subprocess.run([sys.executable,"data.py"],
                   input=label,
                   text = True)
    
    return jsonify({"message":f"200 reading taken successfully for {label}"})

@app.route("/update_gesture",methods=["POST"])
def update_ges():
    gesture = request.json.get("gesture")
    action = request.json.get("action")

    with open("gesture_assn.json","r") as f:
        configuration=json.load(f)

    configuration[gesture] = action

    with open("gesture_assn.json","w") as f:
        json.dump(configuration,f,indent=4)

    control.gesture_assn=configuration

    return jsonify({"message":"Mapping Updated Succesfully"})

@app.route("/train",methods=["POST"])
def train():
    subprocess.run([sys.executable,"model.py"])

    control.reload_model()
    return jsonify({"message":"Model Retrained Successfully"})

@app.route("/start")
def start():
    control.start()
    return jsonify({"message":"Control Started"})

@app.route("/stop")
def stop():
    control.stop()
    return jsonify({"message":"Control Stopped"})

@app.route("/status")
def status():
    return jsonify(control.status())

if __name__ == "__main__":
    app.run(debug=True)