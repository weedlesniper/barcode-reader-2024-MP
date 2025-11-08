# Author: [Luc Adams]
# Student ID: 20188193
# Last Modified: October 6, 2024
import os
import numpy as np
import cv2
import tensorflow as tensorflow
import matplotlib.pyplot as plt
from task3_util import *
from task1_util import clearDirectory
def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        content.save(output_path)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def run_task3(image_path, config):
    outputPath = 'output/task3'
    verbose = False
    
    # Clear the output directory before starting the task
    clearDirectory(outputPath)

    # Check if the model already exists, otherwise train it.
    trainedSVM = 'data/model/svm.xml'
    if os.path.exists(trainedSVM):
        svm = cv2.ml.SVM_load(trainedSVM)
    #this elif statement will not run/can't run as the data required isn't in the submission folder 
    elif not os.path.exists(trainedSVM):
        features, labels = formatDataset('data/barcodeDigits/')
        # Train SVM
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
        svm.train(features, cv2.ml.ROW_SAMPLE, labels)
        svm.save(trainedSVM)

    if os.path.isdir(image_path):
        os.makedirs(outputPath, exist_ok=True)
        # For each subdirectory (the individual digit images)
        for subdir in sorted(os.listdir(image_path)):
            subdirPath = os.path.join(image_path, subdir)
            
            if os.path.isdir(subdirPath):
                print(f"Processing images in directory: {subdir}")
                
                # # Create a subdirectory for the current barcode in the output directory
                outputSubdir = os.path.join(outputPath, subdir)
                os.makedirs(outputSubdir, exist_ok=True)
                
                idx = 0  # Initialize the index manually
                for imgName in sorted(os.listdir(subdirPath)):
                    imgPath = os.path.join(subdirPath, imgName)

                    # Check if the file is an image
                    if imgName.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        # Load the test image in grayscale
                        testImage = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                        #displayImage("image", testImage, cmap="gray")
                        if testImage is not None:
                            # Resize test image to 64x64 if not already
                            testImage = preprocessImage(testImage)
                            # Flatten the test image to a 1D array
                            testImageFlattened = testImage.flatten().astype(np.float32).reshape(1, -1)

                            # Predict the digit using the SVM model
                            result = svm.predict(testImageFlattened)[1]
                            predictedDigit = int(result[0][0])
                            print(f"Predicted digit for image '{imgName}': {predictedDigit}")

                            # Increment index for image processing
                            idx += 1
                            txtFileName = f"d{str(idx).zfill(2)}.txt"  # Use the incremented index for naming

                            # Save the result to the appropriate text file
                            with open(os.path.join(outputSubdir, txtFileName), 'w') as f:
                                f.write(str(predictedDigit))
                        else:
                            print(f"Error: Could not load image '{imgName}'!")
                    elif imgName.lower().endswith('.txt'):
                        # Skip .txt files and continue without incrementing idx
                        pass
    else:
        print(f"Error: '{image_path}' is not a valid directory!")
