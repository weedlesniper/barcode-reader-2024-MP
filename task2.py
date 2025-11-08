

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: [Luc Adams]
# Student ID: 20188193
# Last Modified: October 6, 2024
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from task1_util import *

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def run_task2(directoryPath, config):
    # Define the output directory
    outputDirectory = 'output/task2'
    
    # Clear the contents of the output directory before processing
    clearDirectory(outputDirectory)
    
    # Check if the path is a directory
    if not os.path.isdir(directoryPath):
        print(f"Error: {directoryPath} is not a directory.")
        return
    
    # Iterate over files in the directory
    for filename in os.listdir(directoryPath):
        filePath = os.path.join(directoryPath, filename)
        
        # Process only image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Use a directory under output/task2 for the output of each barcode
            outputDir = os.path.join(outputDirectory, os.path.splitext(filename)[0])
            process_image(filePath, outputDir, config)
            
def process_image(imagePath, outputDir, config):
    # Image name and output directory setup
    imageName = os.path.splitext(os.path.basename(imagePath))[0]
    os.makedirs(outputDir, exist_ok=True)
    
    # Read both the original and grayscale images
    originalImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    if originalImage is None or grayImage is None:
        print(f"Error: Unable to load image {imagePath}")
        return

    # Preprocess the image (binary threshold and morphology operations)
    _, binaryImage = cv2.threshold(grayImage, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours (assuming barcodes are the largest 13)
    contoursByArea = sorted(contours, key=cv2.contourArea, reverse=True)[:13]
    sortedContours = sorted(contoursByArea, key=lambda c: cv2.boundingRect(c)[0])

    # Extract the barcode digits and save them (from the original image)
    for idx, contour in enumerate(sortedContours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the digit region from the original image
        digitRegion = originalImage[y:y + h, x:x + w]
        
        # Save each digit region from the original image
        digitImagePath = os.path.join(outputDir, f'd{idx + 1:02d}.png')
        save_output(digitImagePath, digitRegion, output_type='image')
        
        # Save the bounding box coordinates to a text file
        coordinates = f"{x} {y} {x + w} {y + h}"
        coordinatesPath = os.path.join(outputDir, f'd{idx + 1:02d}.txt')
        save_output(coordinatesPath, coordinates, output_type='txt')
    
    print(f"Processed image {imagePath} and saved digit regions from the original image to {outputDir}")


