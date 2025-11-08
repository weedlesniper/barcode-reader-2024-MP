import cv2
import os
from ultralytics import YOLO

# Get the absolute path of the current working directory
base_path = os.path.abspath(os.getcwd())

# Construct the full path to your model file
model_path = os.path.join(base_path, 'data', 'modelv2', 'runs', 'obb', 'train3', 'weights', 'best.pt')

# Load your trained model
model = YOLO(model_path)

# Read the image using OpenCV
colourImage = cv2.imread('validation/test/task1/img4.jpg', cv2.IMREAD_COLOR)

# Ensure output directory exists
output_dir = os.path.join(base_path, 'output', 'task1')
os.makedirs(output_dir, exist_ok=True)

# Set the working directory to your output directory
os.chdir(output_dir)

# Use the model to make predictions on the image and save to the specified directory
results = model.predict(source=colourImage, imgsz=640, save=True, save_dir=output_dir)

print(f"Prediction saved to: {output_dir}")
