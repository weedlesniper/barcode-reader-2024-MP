# Author: [Luc Adams]
# Student ID: 20188193
# Last Modified: October 6, 2024
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'
MAGENTA = "\033[95m"

def displayImage(title, image, cmap = None):
    plt.figure(figsize = (10, 6))
    plt.imshow(image, cmap = cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def clearDirectory(outputDirectory):
    """
    Clears the contents of the specified directory. If the directory does not exist,
    it will be created.
    
    Args:
        outputDirectory (str): The path of the directory to clear.
    """
    if os.path.exists(outputDirectory):
        # Walk through the directory tree and delete files and directories
        for root, dirs, files in os.walk(outputDirectory, topdown=False):
            #clear files
            for name in files:
                filePath = os.path.join(root, name)
                try:
                    os.remove(filePath)
                except Exception as e:
                    print(f"Failed to delete file {filePath}. Reason: {e}")
            # clear directories 
            for name in dirs:
                dirPath = os.path.join(root, name)
                try:
                    os.rmdir(dirPath)
                except Exception as e:
                    print(f"Failed to delete directory {dirPath}. Reason: {e}")
    else:
        # Create the directory if it does not exist
        os.makedirs(outputDirectory)

def preprocess(grayImage):
    """
    Preprocesses a grayscale image to enhance edges and close gaps.

    Parameters:
    -----------
    grayImage : numpy.ndarray
        Grayscale image to be preprocessed.

    Function Steps:
    ---------------
    1. Applies a Gaussian blur to the input image to reduce noise.
    2. Performs Canny edge detection to highlight the edges in the image.
    3. Applies a morphological closing operation to fill small gaps in detected edges using a rectangular kernel.

    Returns:
    --------
    numpy.ndarray:
        Preprocessed image with edges enhanced and gaps closed.
    """
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

    edges = cv2.Canny(blurredImage, 50, 150)

    kernel = np.ones((3, 2), np.uint8)

    closedImage = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closedImage

def getBarcodeCoordinatesFromYolo(imagePath, pathToModel, outputDirectory, verbose):
    """
    Detects barcode coordinates from an image using a YOLO model.

    Parameters:
    -----------
    imagePath : str
        Path to the image file for barcode detection.
    pathToModel : str
        Path to the YOLO model file.
    outputDirectory : str
        Directory where the output image with the drawn barcode bounding box will be saved (if verbose is enabled).
    verbose : bool
        Flag to indicate whether intermediate visualizations should be saved.

    Function Steps:
    ---------------
    1. Loads the YOLO model from the specified path.
    2. Reads the image from the given path.
    3. Runs the YOLO model to predict barcode bounding box coordinates (using oriented bounding boxes).
    4. If no barcode is detected, raises a ValueError.
    5. Extracts the barcode coordinates from the prediction results.
    6. If verbose mode is enabled, draws the barcode bounding box on the image and saves it to the specified output directory.

    Returns:
    --------
    list of tuples:
        A list of tuples representing the four coordinates of the detected barcode bounding box.

    Raises:
    -------
    - ValueError: If the image cannot be loaded or no barcode is detected.
    """
    #loading the YOLO model using the absolute path
    absModelPath = os.path.abspath(pathToModel)
    model = YOLO(absModelPath)

    predictionImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if predictionImage is None:
        raise ValueError(f"Error: Failed to load image from {imagePath}")
    
    os.makedirs(outputDirectory, exist_ok=True)
    results = model.predict(source=imagePath, verbose = False, imgsz=640)
    
    #results are an array, because yolo allows the functionality to pass multiple prediction images
    #we just need the first result since we're passing one at a time
    result = results[0]

    # check if there exists a barcode bounding box 
    if result.obb is None or len(result.obb) == 0:
        print(f"{RED}No barcode detected in {imagePath}, proceeding to next image")
        raise ValueError(f"No barcode detected in the image using OBBs.{RESET}")

    # Extract the YOLO obb results
    obbPredictions = result.obb.xyxyxyxy.cpu().numpy()
    barcodeCoords = obbPredictions[0]  
    barcodeCoords = barcodeCoords.astype(int)

    # representing the barcodeCoords as points for visualisation
    x1, y1 = barcodeCoords[0]
    x2, y2 = barcodeCoords[1]
    x3, y3 = barcodeCoords[2]
    x4, y4 = barcodeCoords[3]

    if verbose:
        predictionImageCopy = predictionImage.copy()
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(predictionImageCopy, [points], isClosed=True, color=(0, 255, 255), thickness=2)
        outputBarcodeBox = os.path.join(outputDirectory, '1.3barcode_detected_by_yolo.jpg')
        cv2.imwrite(outputBarcodeBox, predictionImageCopy)
        
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def getBarcodeContoursFromOBB(points, contours, imageSize, outputPath = '', threshold=0.5):
    """
    Filters contours based on whether they lie within the barcode's bounding box (OBB).

    Parameters:
    -----------
    points : numpy.ndarray
        Points representing the barcode's bounding box.
    contours : list of numpy.ndarray
        List of contours detected in the image.
    imageSize : tuple
        Size of the image (height, width) for creating the blank mask.
    output_path : str, optional
        Path to save the visualization of the mask and contours (default is '').
    threshold : float, optional
        Percentage threshold of contour points that must lie within the bounding box to be included (default is 0.5).

    Function Steps:
    ---------------
    1. Creates a blank mask and fills it with the barcode's bounding box (polygon).
    2. Iterates through each contour, counting how many points lie within the barcode's bounding box.
    3. Filters the contours based on the specified percentage threshold.
    4. If `output_path` is provided, visualizes the mask with all contours (in red) and filtered contours (in green), and saves the result.

    Returns:
    --------
    list of numpy.ndarray:
        List of filtered contours that meet the threshold of points inside the barcode's bounding box (hopefully the barcode lines, but maybe some outliers)
    """
    filteredContours = []
    # iterate through each contour to determine if its within the barcode bounding box
    for contour in contours: 
        withinBarcodeCount = 0
        for point in contour:
            x, y = point[0]  
            if cv2.pointPolygonTest(np.array(points), (float(x), float(y)), False) >= 0:
                withinBarcodeCount += 1
        
        #  the percentage of points inside the polygon
        percentageInsideBarcodeMask = withinBarcodeCount / len(contour)

        # if the percentage meets the threshold, keep the contour
        if percentageInsideBarcodeMask >= threshold:
            filteredContours.append(contour)
    
    if outputPath:
        #visual representation of function's operation if verbose enabled
        blankMask = np.zeros(imageSize, dtype=np.uint8)
        cv2.fillPoly(blankMask, pts=[points], color=255) 
        visualMaskCopy = cv2.cvtColor(blankMask, cv2.COLOR_GRAY2BGR)  
        cv2.drawContours(visualMaskCopy, contours, -1, (0, 0, 255), 1)
        cv2.drawContours(visualMaskCopy, filteredContours, -1, (0, 255, 0), 2)
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        cv2.imwrite(outputPath, visualMaskCopy)

    return filteredContours

def rotateImage(filteredBarcodeContours, image):
    """
    Rotates an image based on the orientation of the detected barcode contours to make the longest side horizontal.

    Parameters:
    -----------
    filteredBarcodeContours : list of numpy.ndarray
        List of barcode contours filtered from the image.
    image : numpy.ndarray
        The original image to be rotated.

    Returns:
    --------
    tuple:
        - rotatedImage (numpy.ndarray): The rotated image.
        - rotated (bool): Flag indicating if the image was rotated.
        - rotationMatrix (numpy.ndarray): The 2x3 rotation matrix used to rotate the image.
    """
    try:
        allPoints = np.vstack(filteredBarcodeContours).astype(np.float32)
        currentImage = image.copy()

        rotated = False 
        rotationAmount = 0

        # calculate the minimum enclosing rectangle
        enclosingRectangle = cv2.minAreaRect(allPoints)
        (width, height) = enclosingRectangle[1]

        boxPoints = cv2.boxPoints(enclosingRectangle)
        boxPoints = np.int0(boxPoints)

        # calculate the sides of the enclosing rectangle
        side1 = np.linalg.norm(boxPoints[0] - boxPoints[1])  
        side2 = np.linalg.norm(boxPoints[1] - boxPoints[2])  
        side3 = np.linalg.norm(boxPoints[2] - boxPoints[3])  
        side4 = np.linalg.norm(boxPoints[3] - boxPoints[0])

        # identify the longest side
        sideLengths = [(side1, (boxPoints[0], boxPoints[1])),
                       (side2, (boxPoints[1], boxPoints[2])),
                       (side3, (boxPoints[2], boxPoints[3])),
                       (side4, (boxPoints[3], boxPoints[0]))]
        
        longestSide = max(sideLengths, key=lambda x: x[0])
        longestPoints = longestSide[1]

        # Determine the angle of the longest side relative to the horizontal axis
        point1, point2 = longestPoints
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Determine the rotation amount
        if angle < -90:
            rotationAmount = 180 + angle  
        elif angle > 90:
            rotationAmount = angle - 180 
        else:
            if -2 < angle < 2:  # Angle is approximately 'straight'
                return currentImage, rotated, None  # No rotation needed
            else:
                rotationAmount = angle

        # Rotate the image by the calculated amount
        center = (currentImage.shape[1] // 2, currentImage.shape[0] // 2)
        rotationMatrix = cv2.getRotationMatrix2D(center, rotationAmount, 1.0)
        
        rotatedImage = cv2.warpAffine(currentImage, rotationMatrix, (currentImage.shape[1], currentImage.shape[0]))
        rotated = True

        return rotatedImage, rotated, rotationMatrix

    except Exception as e:
        print(f"Error occurred during rotation: {str(e)}")
        raise

def extractBoundingBox(coords, showPlot, image_path=None):
    """
    Function: extractBoundingBox
    Purpose: Returns the coordinates of a new bounding box, based on the coordinates of the original OBB found by YOLO. The intention
    with this bounding box is to increase the search area slightly above and below, so as to include the barcode digits.
    Output: Boundingbox Perimeter, halfway point of aforementioned bounding box. Also returns top and bottom quarter points for further contour filtering.
    """
    expansionFactor = 0.08
    # Extract bounding box coordinates
    if isinstance(coords, list) and all(isinstance(point, tuple) and len(point) == 2 for point in coords):
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        barcodeLeft = min(x_coords)
        barcodeRight = max(x_coords)
        barcodeTop = min(y_coords)
        barcodeBottom = max(y_coords)
    else:
        raise ValueError(f"Expected coords to be a list of 4 corner points (x, y), but got: {coords}")

    # Calculate the original height and expanded bounding box
    barcodeHeight = barcodeBottom - barcodeTop
    expansionAmount = expansionFactor * barcodeHeight if isinstance(expansionFactor, float) else expansionFactor
    expandedTop = max(0, barcodeTop - expansionAmount)
    expandedBottom = barcodeBottom + expansionAmount

    # Calculate the halfway point, top quarter, and bottom quarter
    halfwayPoint = expandedTop + (expandedBottom - expandedTop) / 2
    topQuarterPoint = expandedTop + (halfwayPoint - expandedTop) / 2
    bottomQuarterPoint = halfwayPoint + (expandedBottom - halfwayPoint) / 2

    # Plotting the bounding box if enabled
    if showPlot:
        if image_path is None:
            raise ValueError("image_path must be provided when plotting is enabled.")
        plt.figure(figsize=(10, 8))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  

        # Draw the expanded bounding box
        plt.plot([barcodeLeft, barcodeRight], [expandedTop, expandedTop], color='blue', linestyle='--', linewidth=2, label='Expanded Bounding Box')
        plt.plot([barcodeLeft, barcodeRight], [expandedBottom, expandedBottom], color='blue', linestyle='--', linewidth=2)
        plt.plot([barcodeLeft, barcodeLeft], [expandedTop, expandedBottom], color='blue', linestyle='--', linewidth=2)
        plt.plot([barcodeRight, barcodeRight], [expandedTop, expandedBottom], color='blue', linestyle='--', linewidth=2)

        # Draw the original bounding box
        plt.plot([barcodeLeft, barcodeRight], [barcodeTop, barcodeTop], color='red', linestyle='-', linewidth=2, label='Original Bounding Box')
        plt.plot([barcodeLeft, barcodeRight], [barcodeBottom, barcodeBottom], color='red', linestyle='-', linewidth=2)
        plt.plot([barcodeLeft, barcodeLeft], [barcodeTop, barcodeBottom], color='red', linestyle='-', linewidth=2)
        plt.plot([barcodeRight, barcodeRight], [barcodeTop, barcodeBottom], color='red', linestyle='-', linewidth=2)

        # Draw the halfway line
        plt.plot([barcodeLeft, barcodeRight], [halfwayPoint, halfwayPoint], color='green', linestyle='-', linewidth=2, label='Halfway Point')

        # Draw the top quarter and bottom quarter lines
        plt.plot([barcodeLeft, barcodeRight], [topQuarterPoint, topQuarterPoint], color='orange', linestyle='-', linewidth=2, label='Top Quarter Point')
        plt.plot([barcodeLeft, barcodeRight], [bottomQuarterPoint, bottomQuarterPoint], color='purple', linestyle='-', linewidth=2, label='Bottom Quarter Point')

        # Set plot properties
        plt.title('Bounding Box with Quarter and Halfway Points')
        plt.legend()
        plt.axis('on')
        plt.show()

    return barcodeLeft, barcodeRight, expandedTop, expandedBottom, halfwayPoint, topQuarterPoint, bottomQuarterPoint

def filterContoursFromExtendedSearchSpace(contours, extended_search_left, extended_search_right, min_y, max_y, halfwayPoint, showPlot=False, imagePath=None):
    """
    Filters contours based on their overlap with the extended search area and plots the results if showPlot is True.
    
    Parameters:
    - contours: List of all original contours
    - extended_search_left: Left boundary of the extended search area
    - extended_search_right: Right boundary of the extended search area
    - min_y: Minimum y-coordinate of the extended search area
    - max_y: Maximum y-coordinate of the extended search area
    - halfwayPoint: The y-coordinate that divides the top and bottom sections
    - showPlot: Boolean flag to toggle plotting
    - imagePath: Path to the image (required if showPlot is True)

    Returns:
    - newContours: List of contours that are filtered within the extended search area
    """
    newContours = []

    # Load the image only once if plotting is enabled
    if showPlot and imagePath is not None:
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image at path: {imagePath}")
        
        # Open a single figure for displaying all contours and the extended search space
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display

        # Draw the extended search space as a rectangle
        plt.plot([extended_search_left, extended_search_right], [min_y, min_y], color='purple', linestyle='--', linewidth=2)  # Top line
        plt.plot([extended_search_left, extended_search_right], [max_y, max_y], color='purple', linestyle='--', linewidth=2)  # Bottom line
        plt.plot([extended_search_left, extended_search_left], [min_y, max_y], color='purple', linestyle='--', linewidth=2)  # Left line
        plt.plot([extended_search_right, extended_search_right], [min_y, max_y], color='purple', linestyle='--', linewidth=2)  # Right line

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the contour's bounding box falls within or overlaps with the extended search area
        overlap_left = max(extended_search_left, x)
        overlap_right = min(extended_search_right, x + w)
        overlap_top = max(min_y, y)
        overlap_bottom = min(max_y, y + h)

        if overlap_left < overlap_right and overlap_top < overlap_bottom:
            overlap_width = overlap_right - overlap_left
            overlap_height = overlap_bottom - overlap_top
            overlap_area = overlap_width * overlap_height
        else:
            overlap_area = 0  # No valid overlap

        contour_area = w * h

        # Check if at least 90% of the contour's area falls within the extended search area
        if overlap_area >= 0.80 * contour_area:
            newContours.append(contour)

        # Plot the contour if plotting is enabled
        if showPlot and imagePath is not None:
            color = 'blue' if overlap_area >= 0.80 * contour_area else 'red'
            plt.plot([x, x + w], [y, y], color=color, linestyle='-', linewidth=2)  # Top line
            plt.plot([x, x + w], [y + h, y + h], color=color, linestyle='-', linewidth=2)  # Bottom line
            plt.plot([x, x], [y, y + h], color=color, linestyle='-', linewidth=2)  # Left line
            plt.plot([x + w, x + w], [y, y + h], color=color, linestyle='-', linewidth=2)  # Right line

    # Display the combined plot if plotting is enabled
    if showPlot and imagePath is not None:
        plt.title('Contours and Extended Search Area (Blue: Inside, Red: Outside, Purple: Search Area)')
        plt.axis('on')
        plt.show()
        plt.close()  # Close the figure to release memory

    return newContours

def filterContoursWithinExtendedBarcode(contours, barcodeLeft, barcodeRight, expandedTop, expandedBottom, showPlot, imagePath):
    """
    Function: filterContoursWithinExtendedBarcode
    Purpose: Since we've increased the search space, we now look for any contours that are within the new search space.
    Now we filter contours based on whether they are within the bottom 1/5th or top 1/5th of the expanded bounding box.
    """
    # Define filtering thresholds
    min_aspect_ratio = 0.2  # Minimum aspect ratio to filter out barcode lines
    min_width = 5  # Minimum width to filter out very thin lines
    overlap_threshold = 0.9  # The required percentage overlap with the bounding box

    top_digit_contours = []  # List to store contours in the top 1/5th of the bounding box
    bottom_digit_contours = []  # List to store contours in the bottom 1/5th of the bounding box

    # Calculate the cutoffs for the top 1/5th and bottom 1/5th of the expanded bounding box
    top_cutoff = expandedTop + (expandedBottom - expandedTop) / 5
    bottom_cutoff = expandedBottom - (expandedBottom - expandedTop) / 5

    # Load the image if plotting is enabled
    if showPlot and imagePath is not None:
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image at path: {imagePath}")
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
        
        # Draw the extended bounding box
        plt.plot([barcodeLeft, barcodeRight], [expandedTop, expandedTop], color='cyan', linestyle='--', linewidth=2, label='Extended Bounding Box')  # Top line
        plt.plot([barcodeLeft, barcodeRight], [expandedBottom, expandedBottom], color='cyan', linestyle='--', linewidth=2)  # Bottom line
        plt.plot([barcodeLeft, barcodeLeft], [expandedTop, expandedBottom], color='cyan', linestyle='--', linewidth=2)  # Left line
        plt.plot([barcodeRight, barcodeRight], [expandedTop, expandedBottom], color='cyan', linestyle='--', linewidth=2)  # Right line

    # Iterate through all contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the total area of the contour's bounding rectangle
        contour_area = w * h
        
        # Calculate the overlap area between the contour and the expanded bounding box
        overlap_left = max(barcodeLeft, x)
        overlap_right = min(barcodeRight, x + w)
        overlap_top = max(expandedTop, y)
        overlap_bottom = min(expandedBottom, y + h)
        
        if overlap_left < overlap_right and overlap_top < overlap_bottom:
            overlap_width = overlap_right - overlap_left
            overlap_height = overlap_bottom - overlap_top
            overlap_area = overlap_width * overlap_height
        else:
            overlap_area = 0  # No valid overlap
        
        # Check if at least 90% of the contour is within the expanded bounding box
        overlap_percentage = overlap_area / contour_area
        if overlap_percentage < overlap_threshold:
            continue  # Skip contours that do not meet the 90% overlap requirement

        # Calculate aspect ratio and filter out barcode lines
        aspect_ratio = w / h
        if aspect_ratio < min_aspect_ratio or w < min_width:
            continue  # Skip this contour

        # Determine if the contour is within the top 1/5th or bottom 1/5th of the expanded bounding box
        contourCenterY = y + h / 2
        if contourCenterY < top_cutoff:
            top_digit_contours.append(contour)  # Add contour to top contours list (top 1/5th)
            contour_color = 'red'  # Contours in the top 1/5th
        elif contourCenterY > bottom_cutoff:
            bottom_digit_contours.append(contour)  # Add contour to bottom contours list (bottom 1/5th)
            contour_color = 'orange'  # Contours in the bottom 1/5th
        else:
            continue  # Skip contours that are not in the top or bottom 1/5th

        # Draw the contour if plotting is enabled
        if showPlot:
            plt.plot([x, x + w], [y, y], color=contour_color, linewidth=1)  # Top side of the contour
            plt.plot([x, x + w], [y + h, y + h], color=contour_color, linewidth=1)  # Bottom side of the contour
            plt.plot([x, x], [y, y + h], color=contour_color, linewidth=1)  # Left side of the contour
            plt.plot([x + w, x + w], [y, y + h], color=contour_color, linewidth=1)  # Right side of the contour

    # Display the plot with the bounding box
    if showPlot:
        plt.title('Contours Within the Extended Bounding Box')
        plt.axis('on')
        plt.legend()
        plt.show()

    return top_digit_contours, bottom_digit_contours

def showUpsideDownResult(showPlot, imagePath, top_digit_contours, bottom_digit_contours):
    # Plot the result
    if showPlot and imagePath is not None:
        # Load the image
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image at path: {imagePath}")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display

        # Draw bounding boxes around the top contours
        contour_color_top = 'red'
        for contour in top_digit_contours:
            x, y, w, h = cv2.boundingRect(contour)
            plt.plot([x, x + w], [y, y], color=contour_color_top, linewidth=2)       # Top line
            plt.plot([x, x + w], [y + h, y + h], color=contour_color_top, linewidth=2)  # Bottom line
            plt.plot([x, x], [y, y + h], color=contour_color_top, linewidth=2)       # Left line
            plt.plot([x + w, x + w], [y, y + h], color=contour_color_top, linewidth=2)  # Right line

        # Draw bounding boxes around the bottom contours
        contour_color_bottom = 'orange'
        for contour in bottom_digit_contours:
            x, y, w, h = cv2.boundingRect(contour)
            plt.plot([x, x + w], [y, y], color=contour_color_bottom, linewidth=2)       # Top line
            plt.plot([x, x + w], [y + h, y + h], color=contour_color_bottom, linewidth=2)  # Bottom line
            plt.plot([x, x], [y, y + h], color=contour_color_bottom, linewidth=2)       # Left line
            plt.plot([x + w, x + w], [y, y + h], color=contour_color_bottom, linewidth=2)  # Right line

        plt.title('showUpsideDownResult')
        plt.axis('on')
        plt.show()

def createExtendedSearchRegion(digit_contours, upsideDown, min_y, max_y, imagePath, showPlot=False):
    """
    Creates the extended search region based on the upsideDown flag and optionally plots the search area with the image.
    Also plots a bounding box around the existing digit contours.

    Parameters:
    - digit_contours: List of contours representing the digits
    - upsideDown: Boolean indicating whether the image is upside down
    - min_y: The minimum y-coordinate of the digit contours
    - max_y: The maximum y-coordinate of the digit contours
    - imagePath: Path to the image file
    - showPlot: Boolean indicating whether to plot the extended search area

    Returns:
    - extended_search_left: Left boundary of the extended search area
    - extended_search_right: Right boundary of the extended search area
    """
    
    # Calculate the bounding box for the existing digit contours
    x_values = [cv2.boundingRect(c)[0] for c in digit_contours]
    y_values = [cv2.boundingRect(c)[1] for c in digit_contours]
    widths = [cv2.boundingRect(c)[2] for c in digit_contours]
    heights = [cv2.boundingRect(c)[3] for c in digit_contours]

    # Determine the combined bounding box that contains all digit contours
    bounding_box_left = min(x_values)
    bounding_box_right = max([x + w for x, w in zip(x_values, widths)])
    bounding_box_top = min(y_values)
    bounding_box_bottom = max([y + h for y, h in zip(y_values, heights)])

    # Calculate the width of the digit contours bounding box
    bounding_box_width = bounding_box_right - bounding_box_left

    # Define the extended search area based on 25% of the bounding box's width
    extension_length = 0.25 * bounding_box_width  # 25% of the digit contours' bounding box width

    # Adjust the search area based on the upsideDown flag
    if upsideDown:

        # Search to the right of the current filtered contours' bounding box
        extended_search_left = bounding_box_right  # Start from the right edge of the filtered contours
        extended_search_right = bounding_box_right + extension_length  # Extend by 25% of the bounding box width

    elif not upsideDown:
        # Search to the left of the current filtered contours' bounding box
        extended_search_right = bounding_box_left  # Start from the left edge of the filtered contours
        extended_search_left = max(0, bounding_box_left - extension_length)  # Extend by 25% of the bounding box width to the left
    
    # Plot the extended search area and the existing digit contour bounding box if showPlot is enabled
    if showPlot:
        # Load the image
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image at path: {imagePath}")
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
        
        # Plot the bounding box of the existing digit contours in green
        plt.plot([bounding_box_left, bounding_box_right], [bounding_box_top, bounding_box_top], color='green', linestyle='-', linewidth=2, label='Digit Contours Bounding Box')  # Top
        plt.plot([bounding_box_left, bounding_box_right], [bounding_box_bottom, bounding_box_bottom], color='green', linestyle='-', linewidth=2)  # Bottom
        plt.plot([bounding_box_left, bounding_box_left], [bounding_box_top, bounding_box_bottom], color='green', linestyle='-', linewidth=2)  # Left
        plt.plot([bounding_box_right, bounding_box_right], [bounding_box_top, bounding_box_bottom], color='green', linestyle='-', linewidth=2)  # Right
        
        # Plot the extended search area in purple using the bounding box's top and bottom values
        plt.plot([extended_search_left, extended_search_right], [bounding_box_top, bounding_box_top], color='purple', linestyle='--', linewidth=2, label='Extended Search Area')  # Top
        plt.plot([extended_search_left, extended_search_right], [bounding_box_bottom, bounding_box_bottom], color='purple', linestyle='--', linewidth=2)  # Bottom
        plt.plot([extended_search_left, extended_search_left], [bounding_box_top, bounding_box_bottom], color='purple', linestyle='--', linewidth=2)  # Left
        plt.plot([extended_search_right, extended_search_right], [bounding_box_top, bounding_box_bottom], color='purple', linestyle='--', linewidth=2)  # Right

        plt.title('createExtendedSearchRegion')
        plt.legend()
        plt.axis('on')
        plt.show()

    return extended_search_left, extended_search_right

def extractDigitContours(contours, coords, imagePath):
    """
    Function: extractDigitContours 
    Purpose: To take in an already positioned barcode and return the numerical values at the top or bottom
    """
    upsideDown = False
    verbose = False
    barcodeLeft, barcodeRight, expandedTop, expandedBottom, halfwayPoint, topQuarter, bottomQuarter = extractBoundingBox(coords, verbose, imagePath)
    
    top_digit_contours, bottom_digit_contours = filterContoursWithinExtendedBarcode(contours, barcodeLeft, barcodeRight, expandedTop, expandedBottom, verbose, imagePath)
    # Now we have the contours that exist within the extended bounding box. 
    if len(top_digit_contours) >= 12 or len(bottom_digit_contours) >= 12:
        if len(top_digit_contours) == 0: 
            upsideDown = False
        elif len(bottom_digit_contours) == 0:
            upsideDown = True
        else:
            # Calculate the average area of top and bottom contours
            avg_top_area = np.mean([cv2.contourArea(c) for c in top_digit_contours])
            avg_bottom_area = np.mean([cv2.contourArea(c) for c in bottom_digit_contours])

            # Compare the average areas with a tolerance ratio
            if avg_top_area > avg_bottom_area * 1.1:  # Use a 10% tolerance ratio
                upsideDown = True
            elif avg_bottom_area > avg_top_area * 1.1:
                upsideDown = False
            else:
                upsideDown = None  # Set to None if it's ambiguous

    else:
        upsideDown = None  # Indicate that the decision could not be made
        raise RuntimeError("can't figure out if upsidedown or not")
    
    showPlot = verbose
    showUpsideDownResult(showPlot, imagePath, top_digit_contours, bottom_digit_contours)

    # Calculate the height range (min_y to max_y) of the filtered digit contours
    if upsideDown:
        min_y = min([cv2.boundingRect(c)[1] for c in top_digit_contours])
        max_y = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in top_digit_contours])
    elif not upsideDown:
        min_y = min([cv2.boundingRect(c)[1] for c in bottom_digit_contours])
        max_y = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in bottom_digit_contours])

    # Determine the extended search area based on whether the barcode is upside down or not
    if upsideDown:
        extended_search_left, extended_search_right = createExtendedSearchRegion(top_digit_contours, upsideDown,  min_y, max_y, imagePath, verbose)
        filtered_digit_contours = top_digit_contours
    elif not upsideDown:
        extended_search_left, extended_search_right = createExtendedSearchRegion(bottom_digit_contours, upsideDown, min_y, max_y, imagePath, verbose)
        filtered_digit_contours = bottom_digit_contours
    else:
        return top_digit_contours, bottom_digit_contours  # If no orientation could be determined, return early
    
    # Use the new filtering function with plotting support
    showPlot = verbose  # Set to True if you want to visualize the results
    newContours = filterContoursFromExtendedSearchSpace(
        contours,
        extended_search_left,
        extended_search_right,
        min_y,
        max_y,
        halfwayPoint,
        showPlot=showPlot,
        imagePath=imagePath
    )
    filtered_digit_contours.extend(newContours)
    if len(filtered_digit_contours) < 10:
        raise RuntimeError(f"{RED}filtered digit contours should be at least 12 long, representing the barcode digits, there is likely no barcode{RESET}")
    return filtered_digit_contours, upsideDown

def confirmCoordinates(image, coordinates):
    """
    Display an image with a polygon outlined using the provided 8 coordinates (representing 4 points).
    
    Args:
    - image: The image to display (as a NumPy array).
    - coordinates: A tuple or list of 8 values representing (x1, y1, x2, y2, x3, y3, x4, y4)
      where each pair (x, y) represents a corner of the polygon.
    
    Example:
    - coordinates = (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    # Make a copy of the image to avoid modifying the original
    image_copy = image.copy()
    
    # Ensure that coordinates contain 8 values (4 points)
    if len(coordinates) != 8:
        raise ValueError("Coordinates must contain 8 values (x1, y1, x2, y2, x3, y3, x4, y4).")
    
    # Reshape the coordinates into a 4-point array for the polygon
    points = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
    
    # Convert the list of points to a NumPy array for use with OpenCV
    points_array = np.array(points, np.int32).reshape((-1, 1, 2))
    
    # Draw the polygon with the 4 points
    cv2.polylines(image_copy, [points_array], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Display the image with the polygon using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper color display
    plt.title('Image with Polygon (4 points)')
    plt.axis('on')
    plt.show()

def calculateRotatedBarcodeCoordinates(colourImage, contours, centre, rotationMatrix, finalBarcodeFilePath):
    """
    Calculates the coordinates of the bounding box for the barcode in the original image
    by reverse engineering the transformation using the inverse of the rotation matrix.

    Parameters:
    -----------
    - colourImage : numpy.ndarray
        The original color image.
    - contours : list of numpy.ndarray
        List of contours detected for the barcode.
    - centre : tuple
        The center of rotation (typically the center of the image).
    - rotationMatrix : numpy.ndarray
        The 2x3 rotation matrix used for rotating the image.
    - finalBarcodeFilePath : str
        The file path to save the calculated coordinates.

    Output:
    -------
    - Saves the reverse-engineered barcode coordinates to the provided file path. Only works on image that has been rotated,
    not on an image that has been affine transformed too. 
    """
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    topLeft = (x, y)
    topRight = (x + w, y)
    bottomRight = (x + w, y + h)
    bottomLeft = (x, y + h)
    
    #convert contour bounding box points to coordinates 
    coords = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.float32).reshape(-1, 2)

    #inverse the coordinates to retrieve the original coordinates 
    inverseRotationMatrix = cv2.invertAffineTransform(rotationMatrix)
    rotatedCoordinates = cv2.transform(np.array([coords]), inverseRotationMatrix)[0]
    
    # converting points into output format specified. 
    coordString = ','.join([f'{int(pt[0])},{int(pt[1])}' for pt in rotatedCoordinates])

    with open(finalBarcodeFilePath, 'w') as file:
        file.write(coordString)

    coordinates = list(map(int, coordString.split(',')))
    
    #confirmCoordinates(colourImage, coordinates)

def checkAffineDistortion(rotatedContours, image):
    """
    Check for affine distortion in an image based on the angles of detected barcode contours.

    This function calculates the angles of individual contours in the image and computes the median angle.
    It assumes that the barcode lines should be vertical (near 90°), so if the median angle deviates
    significantly from this value, the function concludes that affine distortion may be present.

    Parameters:
    -----------
    rotatedContours : list of numpy.ndarray
        A list of barcode contours that have been rotated or transformed.
        
    image : numpy.ndarray
        The input image on which the barcode contours are located. Used for visualizing the contour angles.

    Returns:
    --------
    bool:
        - Returns True if affine distortion is detected (i.e., if the median contour angle is significantly different from 90°).
        - Returns False if no affine distortion is detected (i.e., if the median contour angle is close to 90°).
    """
    angles = []

    # This function is sensitive to noise, so we're doing aditional noise filtering. 
    minContourLength = 10
    visImage = image.copy()

    #calculate each angle of contours
    for contour in rotatedContours:
        if cv2.arcLength(contour, True) < minContourLength:
            continue  # kkip small, noisy contours

        # fit a straight line to each contour
        line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = line[0], line[1], line[2], line[3]
        
        # calculate the angle of the line (in degrees) relative to the horizontal axis
        angle = np.degrees(np.arctan2(vy, vx))
        
        # normalize angle to be within [-90, 90] range
        # probably not needed since its after horizontal rotation
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        angles.append(angle)
        
        # visualization code for the contour and its angle on the image
        startPoint = (int(x - vx * 100), int(y - vy * 100))
        endPoint = (int(x + vx * 100), int(y + vy * 100))
        cv2.line(visImage, startPoint, endPoint, (255, 0, 0), 2)
        cv2.putText(visImage, f"{float(angle):.2f}°", (startPoint[0], startPoint[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # calculate the median angle of all contours
    if len(angles) > 0:
        # use the median to reduce the influence of outliers
        medianAngle = np.median(angles)
        #print(f"Median angle of barcode contours: {medianAngle:.2f}°")
    else:
        print("No valid contours found.")
        return False

    # check if the median angle is within the allowable tolerance range
    if (80 <= abs(medianAngle) <= 95):  
        return False  # no affine distortion
    else:
        return True  # affine distortion detected
    
def extractConvexHullCorners(contours):
    """
    Extracts the convex hull and identifies the four corners of the barcode region.
    We use this function to determine the original shape of the barcode in the affine image, so that we can determine the source 
    points that are transformed using homography. 
    Parameters:
    -----------
    contours : list of numpy.ndarray
        List of contours from which the convex hull will be calculated.

    Returns:
    --------
    tuple:
        - topLeft (tuple): Coordinates of the top-left corner of the bounding box.
        - topRight (tuple): Coordinates of the top-right corner of the bounding box.
        - bottomRight (tuple): Coordinates of the bottom-right corner of the bounding box.
        - bottomLeft (tuple): Coordinates of the bottom-left corner of the bounding box.
    """
    allPoints = np.vstack(contours)

    # create a convex hull around the outside of the contours. 
    hull = cv2.convexHull(allPoints)

    #need to reshape to determine the 4 points
    hullPoints = hull.reshape(-1, 2)  

    # code to find points from flattened points 
    s = hullPoints.sum(axis=1)
    diff = np.diff(hullPoints, axis=1)

    topLeft = hullPoints[np.argmin(s)]
    bottomRight = hullPoints[np.argmax(s)]
    topRight = hullPoints[np.argmin(diff)]
    bottomLeft = hullPoints[np.argmax(diff)]

    # return the four corners
    return topLeft, topRight, bottomRight, bottomLeft

def performHomographyAndWarp(rotatedPoints, rotatedImage, verbose=False):
    """
    Performs homography transformation and warps the perspective to straighten the barcode region.

    Parameters:
    -----------
    rotatedPoints : numpy.ndarray
        Points representing the corners of the bounding box to be transformed.
    rotatedImage : numpy.ndarray
        The image to be warped.
    verbose : bool
        If True, display the warped image after transformation.

    Returns:
    --------
    tuple:
        - warpedImage : numpy.ndarray
            The warped image after homography.
        - homographyMatrix : numpy.ndarray
            The homography matrix used for the transformation.
    """
    # define the destination points based on the bounding box of the barcode
    # i wanted to experiment with different ways of determining 
    # minx and miny, but i ran out of time. 
    minX, minY = np.min(rotatedPoints, axis=0)
    maxX, maxY = np.max(rotatedPoints, axis=0)

    barcodeWidth = int(maxX - minX)
    barcodeHeight = int(maxY - minY)

    # define destination points as a rectangle preserving the aspect ratio of the barcode
    dstPoints = np.array([
        [minX, minY],  # TL
        [minX + barcodeWidth - 1, minY],  # TR
        [minX + barcodeWidth - 1, minY + barcodeHeight - 1],  # BR
        [minX, minY + barcodeHeight - 1]  # BL
    ], dtype="float32")

    #compute the homography matrix
    homographyMatrix, status = cv2.findHomography(rotatedPoints, dstPoints)

    # apply the homography transformation to the entire image
    warpedImage = cv2.warpPerspective(rotatedImage, homographyMatrix, (rotatedImage.shape[1], rotatedImage.shape[0]))

    # display the warped image if verbose
    if verbose:
        plt.imshow(cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB))
        plt.title("Warped Image with Preserved Context")
        plt.show()

    return warpedImage, homographyMatrix

def reverseEngineerCoordinates(homographyMatrix, rotationMatrix, points):
    """
    Reverse engineer the coordinates by applying the inverse of the homography and rotation matrices.
    If we were just to extract the coordinates without applying any transformation, then the result would
    be a region that doesn't describe the barcode on the original image. 

    Args:
    - homographyMatrix: The homography matrix used in the transformation.
    - rotationMatrix: The rotation matrix used in the transformation.
    - points: The points to reverse engineer (e.g., top-left, top-right, etc.).

    Returns:
    - Reversed points as a flat list of 8 values (x1, y1, x2, y2, x3, y3, x4, y4).
    """
    # inverse the homography and rotation matrices
    invHomography = np.linalg.inv(homographyMatrix)
    invRotation = np.linalg.inv(np.vstack([rotationMatrix, [0, 0, 1]]))[:2, :]  # Convert 2x3 to 3x3 for inversion

    # Apply the inverse homography first, then inverse rotation
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    
    # Apply inverse homography matrix
    homographyReversedPoints = cv2.perspectiveTransform(np.array([points]), invHomography)[0]

    # Apply inverse rotation matrix
    reversedPoints = cv2.transform(np.array([homographyReversedPoints]), invRotation)[0]

    # Return the reversed points as a flat list of 8 values
    reversedPoints = np.round(reversedPoints).astype(int).flatten()
    return reversedPoints
