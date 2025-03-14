import io
import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import base64

# Load YOLOv5 model
model = YOLO("yolov5s.pt")  # Ensure yolov5s.pt is in the main directory

app = FastAPI()

# Enable CORS for all domains (so external webpages can access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receives an image, processes it with YOLOv5, and returns the labeled image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        # Run YOLO model
        results = model(img_np)

        # Draw bounding boxes on image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert back to image and encode in Base64
        _, img_encoded = cv2.imencode(".jpg", img_np)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        return JSONResponse(content={"message": "Success", "image": img_base64})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
