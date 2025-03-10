from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model
try:
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
except Exception as e:
    print("Error loading YOLO model:", e)
    exit(1)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file found"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Create a unique folder for each upload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.join(UPLOAD_FOLDER, f"upload_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        # Save uploaded image
        filename = secure_filename(file.filename)
        image_path = os.path.join(folder_name, filename)
        file.save(image_path)

        # Process image with YOLOv5
        results = model(image_path)

        # Save processed image in the same folder (NO "exp" folder)
        processed_filename = f"processed_{timestamp}.jpg"
        processed_image_path = os.path.join(folder_name, processed_filename)
        results.render()
        for img in results.ims:
            import cv2
            cv2.imwrite(processed_image_path, img)

        if not os.path.exists(processed_image_path):
            return jsonify({"error": "Processed image not found"}), 500

        return jsonify({"processed_image": f"/processed/{folder_name}/{processed_filename}"}), 200

    except Exception as e:
        print("Error in upload_image:", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/processed/uploads/<path:filename>")
def get_processed_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
