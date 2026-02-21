from gesture_controller import Gesture_controller
from flask import Flask,jsonify,render_template,Response

app = Flask (__name__)  #create new web server

control = Gesture_controller()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(control.generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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