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

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"

# Author: [Luc Adams]
# Student ID: 20188193
# Last Modified: October 6, 2024
from task1_util import *
import os
import cv2
import numpy as np

verbose = False


def saveOutput(output_path, content, output_type = 'txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
    elif output_type == 'image':
        cv2.imwrite(output_path, content)

def run_task1(image_directory, config):
    """
    Processes all image files in the given directory for barcode detection.

    Parameters:
    -----------
    image_directory : str
        Path to the directory containing image files.
    config : dict
        Configuration settings (currently not used in the function).

    Function Steps:
    ---------------
    1. Deletes existing JPG files in the visualisation directory.
    2. Iterates through each image in the directory.
    3. If the file is a valid image format, it calls the `processImage` function to process it.
    4. Handles and skips images that raise any exceptions (e.g., no barcode found).

    Outputs:
    --------- 
    Nothing explicitly, as the `processImage` function is responsible for saving output to the respective folders. 
    """
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    visualisationDirectory = 'data/visualisation'
    outputDirectory = 'output/task1'

    clearDirectory(outputDirectory)
    clearDirectory(visualisationDirectory)

    # Process the images
    for image_name in sorted(os.listdir(image_directory)):
        image_path = os.path.join(image_directory, image_name)

        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                processImage(image_path)
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                # Log the error and move on to the next image
                print(f"Error processing {image_name}: {str(e)}")
                continue  # Skip to the next image

def processImage(image_path):
    """
    Processes an image to detect and extract barcode digits.

    Parameters:
    -----------
    image_path : str
        Path to the image file.

    Function Steps:
    ---------------
    1. Reads the image, converts it to grayscale, and preprocesses it for contour detection.
    2. Uses a YOLO model to detect barcode bounding box coordinates.
    3. Extracts barcode contours and checks for image rotation. If rotated, adjusts contours accordingly.
    4. Extracts the digit region from the barcode, handles upside-down images, and saves the cropped digit region.
    5. Optionally saves intermediate visualizations if verbose mode is enabled.
    
    Outputs:
    --------
    - Saves cropped digit region image and its coordinates to a text file.
    
    Raises:
    -------
    - FileNotFoundError: If the YOLO model file is missing.
    - RuntimeError: If no barcode is found.
    """
    print(f"{GREEN}_____________________________________________________________________")
    print(f"Beginning of processing for {image_path}{RESET}")
    
    visualisationDirectory = 'data/visualisation'
    colourImage = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if colourImage is None:
        return
    # extract the filename without the file extension
    fileName = os.path.basename(image_path)
    image_height, image_width = colourImage.shape[:2]
    centre = (image_width // 2, image_height // 2)

    #extract number
    number = fileName.split('.jpg')[0].split('img')[-1]

    # construct output paths for coordinate and image
    final_output_path = f'output/task1/barcode{number}.png'
    finalBarcodePath = f'output/task1/img{number}.txt'

    
    #for screen demo
    show_bgr("1. Original (BGR)", colourImage)


    # convert to grayscale
    bgr = cv2.cvtColor(colourImage, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    #for screen demo
    beforeRotation = preprocess(grayImage)
    show_gray("2. Preprocess: Canny + Close", beforeRotation)

    contours, _ = cv2.findContours(beforeRotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 15

    # Filter contours based on area
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    # using base directory so program can navigate to YOLO file location
    baseDirectory = os.path.dirname(os.path.abspath(__file__))
    relativeModelPath = os.path.normpath('data/model/best.pt')
    pathToModel = os.path.join(baseDirectory, relativeModelPath)

    if not os.path.exists(pathToModel):
        raise FileNotFoundError(f"Model file not found at: {pathToModel}")

    print(f"{YELLOW}Using YOLO model to retrieve Barcode Bounding Box Coordinates in {image_path}...{RESET}")
    barcodeCoords = getBarcodeCoordinatesFromYolo(image_path, 'data/model/best.pt', visualisationDirectory, verbose)
    if(len(barcodeCoords) != 0):
        print(f"{GREEN}Coordinates Retrieved Successfully from {image_path} {RESET}")
    else:
        print(f"{GREEN}No barcode found in image {image_path} {RESET}")
        raise RuntimeError(f"No barcode found in image {image_path}")
    
    #convert YOLO results to type integer
    points = np.array(barcodeCoords, dtype=np.int32)

    #for screen demo
    barcodeOverlay = colourImage.copy()
    cv2.polylines(barcodeOverlay, [points.reshape(-1,1,2)], True, (0,255,0), 3)
    show_bgr("3. YOLO OBB on Image", barcodeOverlay)

    #draw the extracted points for visualisation purposes 
    if verbose:
        barcodeDrawnImage = colourImage.copy()
        cv2.polylines(barcodeDrawnImage, [points], isClosed=True, color=(0, 255, 0), thickness=4)
        outputBarcodeBox = r'data/visualisation/1.1_afterBarcodeDrawn.jpg'
        saveOutput(outputBarcodeBox, barcodeDrawnImage, output_type='image')
    
    # extract contours within the barcode bounding box and draw them on a copy
    if verbose: #code for visualisation if verbose flag set 
        outputPath = 'data/visualisation/1.2filtered_contours_from_OBB.jpg'
        filteredContours = getBarcodeContoursFromOBB(points, contours, colourImage.shape[:2], outputPath)
        contourDemo = np.zeros_like(colourImage)
        cv2.drawContours(contourDemo, filteredContours, -1, (0, 255, 0), 2)
        saveOutput(outputPath, contourDemo, output_type='image')
    else: #else we still need the filteredContours
        filteredContours = getBarcodeContoursFromOBB(points, contours, colourImage.shape[:2])
    
    #for screen demo
    obbMaskVis = colourImage.copy()
    cv2.drawContours(obbMaskVis, filteredContours, -1, (0,255,255), 2)
    show_bgr("4. Contours inside YOLO OBB", obbMaskVis)
  
    # Rotate the image and determine if rotation occurred
    rotatedImage, isRotated, rotationMatrix = rotateImage(filteredContours, colourImage.copy())
    
    #for screen demo
    if isRotated:
        show_bgr("7. Rotated to horizontal", rotatedImage)
    else:
        show_bgr("7. No rotation needed", colourImage)
    
    if not isRotated:
        digitContours, upsideDown = extractDigitContours(contours, barcodeCoords, imagePath=image_path)
        print(f"Extracting Digit Contours from the image, no rotation occurred")
        x, y, w, h = cv2.boundingRect(np.vstack(digitContours))
        x1, y1 = x, y
        x2, y2 = x + w, y
        x3, y3 = x + w, y + h
        x4, y4 = x, y + h
        digitRegionCopy = colourImage.copy()
        
        if verbose:
            cv2.drawContours(digitRegionCopy, digitContours, -1, (255, 0, 0), 2)
            cv2.rectangle(digitRegionCopy, (x1, y1), (x3, y3), (0, 255, 255), 2)
            region = r'data/visualisation/1.4_afterRecognisedRegion.jpg'
            saveOutput(region, digitRegionCopy, output_type='image')

        # Cropping barcode digit region
        croppedRegion = colourImage[y:y + h, x:x + w]  
        if upsideDown:
            # Flip the cropped region both vertically and horizontally to make it right-side up
            croppedRegion = cv2.flip(croppedRegion, -1)  
        saveOutput(final_output_path, croppedRegion, output_type='image')

        if verbose:
            outputFilteredContoursBelow = r'data/visualisation/1.6_digitContours.jpg'
            saveOutput(outputFilteredContoursBelow, digitRegionCopy, output_type='image')
        
        # Save the coordinates to a text file
        with open(finalBarcodePath, 'w') as file:
            # Write the coordinates in the desired format
            # Assuming you have the values of x1, y1, x2, y2, x3, y3, x4, y4
            file.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4}")
            
        #confirmCoordinates(colourImage, (x1, y1, x2, y2, x3, y3, x4, y4))
        if verbose:
            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            points = points.reshape((-1, 1, 2))  # Reshape for use with cv2.polylines
            cv2.polylines(colourImage, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            saveOutput(r'data/visualisation/1.7whoareyou', colourImage, output_type='image')
            

        print(f"{GREEN}Finished processing{image_path}{RESET}")

    else:
        #image was rotated
        postRotation = r'data/visualisation/1.7_rotatedimage.jpg'
        saveOutput(postRotation, rotatedImage, output_type='image')

        # Preprocess the rotated image to obtain a binary or thresholded image for contour detection
        afterRotation = preprocess(rotatedImage)
        # Create a copy of the rotated image for drawing contours
        testContours = rotatedImage.copy()
        # Find contours on the preprocessed image
        postRotationContours, _ = cv2.findContours(afterRotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        postRotationContours = [contour for contour in postRotationContours if cv2.contourArea(contour) > min_contour_area]

        # for screen demo
        contours_vis = rotatedImage.copy()
        cv2.drawContours(contours_vis, postRotationContours, -1, (0, 255, 0), 2)
        show_bgr("8. Contours after rotation", contours_vis)

        # Visualize the found contours
        if verbose:
            cv2.drawContours(testContours, postRotationContours, -1, (0, 0, 255), 2)  # Draw in red color
            contoursVisualisationOutput = r'data/visualisation/2.1_contours_visualization_after_rotation.jpg'
            saveOutput(contoursVisualisationOutput, testContours, output_type='image')
            print(f"Contours visualized and saved to: {contoursVisualisationOutput}")

        # Obtain barcode coordinates from YOLO
        barcodeCoords = getBarcodeCoordinatesFromYolo(postRotation, 'data/model/best.pt', 'data/visualisation', verbose)
        rotatedPoints = np.array(barcodeCoords, dtype=np.int32)

        yolo_overlay = rotatedImage.copy()
        cv2.polylines(yolo_overlay, [rotatedPoints.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=3)
        show_bgr("9. YOLO OBB on rotated image", yolo_overlay)

        if verbose:
            # Save the visualization of the YOLO detected bounding box
            yoloBoundingBoxOutput = r'data/visualisation/2.0_yolo_detected_bounding_box.jpg'
            yoloBarcodeImage = rotatedImage.copy()
            cv2.polylines(yoloBarcodeImage, [rotatedPoints], isClosed=True, color=(255, 0, 0), thickness=4)
            saveOutput(yoloBoundingBoxOutput, yoloBarcodeImage, output_type='image')
            print(f"YOLO detected bounding box saved to: {yoloBoundingBoxOutput}")

            barcodeDrawnImage = rotatedImage.copy()
            cv2.polylines(barcodeDrawnImage, [rotatedPoints], isClosed=True, color=(0, 255, 0), thickness=4)
            outputBarcodeBox = r'data/visualisation/2.1_afterBarcodeDrawnOnRotated.jpg'
            saveOutput(outputBarcodeBox, barcodeDrawnImage, output_type='image')
                
            contoursVisualization = rotatedImage.copy()
            rota = rotatedImage.copy()
            postRotationContoursVisualisation = r'data/visualisation/2.2_postrotationcontours.jpg'
            # Draw the postRotationContours on the copied image
            cv2.drawContours(contoursVisualization, postRotationContours, -1, (0, 255, 0), 2)  # Draw contours in green
            saveOutput(postRotationContoursVisualisation, contoursVisualization, output_type='image')

        # Extract contours within the barcode bounding box and draw them on a copy
        rotatedFilteredContours = getBarcodeContoursFromOBB(rotatedPoints, postRotationContours, rotatedImage.shape[:2],'', 0.5)
        if verbose:
            beforeCheckingAffineContours = 'data/visualisation/BEFOREPASSINGTOAFFINECHECK.jpg'
            cv2.drawContours(rota, rotatedFilteredContours, -1, (0, 255, 0), 2)
            saveOutput(beforeCheckingAffineContours, rota, output_type='image')

        # if we're checking affine transformation
        if checkAffineDistortion(rotatedFilteredContours, afterRotation.copy()):
            print(f"{CYAN}Image is affine performing homography.{RESET}")
            
            topLeft, topRight, bottomRight, bottomLeft = extractConvexHullCorners(rotatedFilteredContours)

            # Arrange the corners in order
            rotatedPoints = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype="float32")
            
            # Apply homography and warp the image using the extracted corners
            warpedFullImage, homographyMatrix = performHomographyAndWarp(rotatedPoints, rotatedImage, verbose)
            originalWarp = warpedFullImage.copy()

            # Save and display the rectified full image
            rectifiedFullOutput = r'data/visualisation/rectified_full_image_preserved_context.jpg'
            cv2.imwrite(rectifiedFullOutput, warpedFullImage)

            if verbose:
                plt.imshow(cv2.cvtColor(warpedFullImage, cv2.COLOR_BGR2RGB))
                plt.title("Rectified Full Image with Preserved Context")
                plt.show()

            # Apply preprocessing after homography transformation
            warpedFullImage = preprocess(warpedFullImage)
            
            # Find contours on the preprocessed, warped image
            postAffineContours, _ = cv2.findContours(warpedFullImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            postAffineContours = [contour for contour in postAffineContours if cv2.contourArea(contour) > min_contour_area]
            
            # Use YOLO to retrieve barcode bounding box coordinates
            barcodeCoords = getBarcodeCoordinatesFromYolo(rectifiedFullOutput, 'data/model/best.pt', visualisationDirectory, verbose)

            # Convert YOLO results to integer type for further processing
            points = np.array(barcodeCoords, dtype=np.int32)

            # Extract barcode contours within the bounding box
            postAffineFilteredContours = getBarcodeContoursFromOBB(points, postAffineContours, warpedFullImage.shape[:2])

            # Extract digit contours from the barcode region
            postAffineDigitContours, upsideDown = extractDigitContours(postAffineContours, barcodeCoords, imagePath=rectifiedFullOutput)

            x, y, w, h = cv2.boundingRect(np.vstack(postAffineDigitContours))
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h

            # Cropping barcode digit region
            croppedRegion = originalWarp[y:y + h, x:x + w]  
            if upsideDown:
                croppedRegion = cv2.flip(croppedRegion, -1)  
            saveOutput(final_output_path, croppedRegion, output_type='image')


            if verbose:
                outputFilteredContoursBelow = r'data/visualisation/3.6_postAffineDigitContours.jpg'
                saveOutput(outputFilteredContoursBelow, digitRegionCopy, output_type='image')

            # Reverse engineer the coordinates
            originalPoints = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
            reversedPoints = reverseEngineerCoordinates(homographyMatrix, rotationMatrix, originalPoints)
            #confirmCoordinates(colourImage, reversedPoints)
            with open(finalBarcodePath, 'w') as file:
                # Writing the reversed coordinates in the desired format (x1, y1, x2, y2, x3, y3, x4, y4)
                file.write(','.join([str(p) for p in reversedPoints]))
            print(f"{GREEN} Finished processing {image_path}{RESET}")

            show_bgr("10. Cropped digits (final, rectified)", croppedRegion)
        # no affine
        else:
            print(f"{CYAN}Image is rotated, but not affine, skipping homography transformation.{RESET}")
            
            extractedContours, upsideDown = extractDigitContours(postRotationContours, barcodeCoords, imagePath=postRotation)
            #had to pass the rotation matrix into this function to reverse engineer coordinates on original image. 
            calculateRotatedBarcodeCoordinates(colourImage, extractedContours, centre, rotationMatrix, finalBarcodePath)
            x, y, w, h = cv2.boundingRect(np.vstack(extractedContours))
            wholeImageWithBarcodeBoundingBox = rotatedImage.copy()
            cv2.rectangle(wholeImageWithBarcodeBoundingBox, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow bounding box
            if verbose: 
                outputWholeImageWithBox = r'data/visualisation/3.0_whole_image_with_bounding_box.jpg'
                saveOutput(outputWholeImageWithBox, wholeImageWithBarcodeBoundingBox, output_type='image')
            show_bgr("10. Whole image with barcode bounding box", wholeImageWithBarcodeBoundingBox)

            extractedBarcodeRegion = rotatedImage.copy()
            croppedRegion = extractedBarcodeRegion[y:y + h, x:x + w]

            if upsideDown:
                croppedRegion = cv2.flip(croppedRegion, -1)  

            saveOutput(final_output_path, croppedRegion, output_type='image')
            print(f"{GREEN} Finished processing {image_path}{RESET}")

            show_side_by_side("Final Comparison", rotatedImage, croppedRegion)

    
    