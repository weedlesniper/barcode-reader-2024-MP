# Author: [Luc Adams]
# Student ID: 20188193
# Last Modified: October 6, 2024
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocessImage(imageToProcess):
    """
    Function: preprocess image 
    Purpose: to apply pre-processing to an image, which will either be a extracted digit we're predicting 
             or an image that has been passed in from the training dataset. 
    """
    #gaussian blur + thresholding
    blurredImage = cv2.GaussianBlur(imageToProcess, (5, 5), 0)
    _, thresholdedImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    #using erosion to 'shrink' the digits 
    kernel = np.ones((3, 2), np.uint8)
    erordedImage = cv2.erode(thresholdedImage, kernel, iterations=1)
    
    #adding a border around the image so that they are more consistent.
    padding = 4
    borderImage = cv2.copyMakeBorder(erordedImage, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    
    resizedImage = cv2.resize(borderImage, (64, 64))
    return resizedImage

def formatDataset(datasetPath):
    """
    Function: formatDataset
    Purpose: traverses the structure of the barcodeDigits directory that is a specially created dataset for training
    my SVM model. This function is responsibel for traversing through the various sub folders (/0, /1 etc) and processing 
    the features and labels so that they can be used by the SVM training methods. 
    """
    features = []
    labels = []
    
    # iterating over each digit folder
    for label in os.listdir(datasetPath):
        labelFolder = os.path.join(datasetPath, label)
        if os.path.isdir(labelFolder):
            # iterating over each individual digit image
            for imageName in os.listdir(labelFolder):
                imagePath = os.path.join(labelFolder, imageName)
                digitImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                
                if digitImage is not None:
                    # Resize image to 64x64
                    image = preprocessImage(digitImage)
                    
                    # Flatten the image to a 1D array
                    image_flattened = image.flatten()
                    
                    # Append the feature and label
                    features.append(image_flattened)
                    labels.append(int(label))
    
    # Convert lists to numpy arrays
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    return features, labels

def displayImage(title, img, cmap=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()