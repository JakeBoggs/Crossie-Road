"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import cv2
import base64
import numpy as np
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

feed = ""
current_frame = None

@app.route("/", methods=["GET"])
def main():
    return render_template("index.html")

@app.route("/camera", methods=["GET"])
def camera():
    return render_template("camera.html")

@app.route("/api/object-detection", methods=["GET"])
def classify():
    global current_frame

    if current_frame != None:
        results = model(current_frame, size=640) # reduce size=320 for faster inference

        return results.pandas().xyxy[0].to_json(orient="records")

@app.route("/api/depth-perception", methods=["POST"])
def depth():
    global feed
    global current_frame

    if not request.method == "POST":
        return
    
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        current_frame = img

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        output = prediction.cpu().numpy()

        _, buffer = cv2.imencode('.png', np.array(output / np.max(output) * 255, dtype=np.uint8))

        feed = base64.b64encode(buffer)

        return feed

@app.route("/api/feed", methods=["GET"])
def get_feed():
    global feed
    return feed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    transform = midas_transforms.small_transform

    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
