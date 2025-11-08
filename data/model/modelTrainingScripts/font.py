import cv2
import os 
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generateBarcodeImages(outputDir='data/barcodeDigits', numSamples=1000):
    os.makedirs(outputDir, exist_ok=True)
    fontPath = 'data/fonts/ocr/font/OCR-B.ttf'

    for digit in range(10):
        digitDir = os.path.join(outputDir, str(digit))
        os.makedirs(digitDir, exist_ok=True)
        for i in range(numSamples):
            canvaSize = 200 
            img = Image.new('L', (canvaSize, canvaSize), color=255)
            draw = ImageDraw.Draw(img)
            
            fontSize = np.random.randint(65, 75) # Choose random font size to simulate real data.
            font = ImageFont.truetype(fontPath, size=fontSize)

            text = str(digit)

            x = canvaSize / 2
            y = canvaSize / 2

            draw.text((x, y), text, fill=0, font=font, anchor="mm")  # Add the text to the centre of the image
            angle = np.random.uniform(-10, 10) # Choose a random roation to add to number
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True, center=(img.width / 2, img.height / 2), fillcolor=255)

            left = (img.width - 64) / 2
            top = (img.height - 64) / 2
            right = left + 64
            bottom = top + 64
            
            img = img.crop((left, top, right, bottom)) # Crop the image so that it is 64 x 64

            imgArray = np.array(img).astype(np.float32) # allows openCV to edit it as it needs to be numpy array.

            noniseSTDDeviation = 5
            noise = np.random.normal(0, noniseSTDDeviation, imgArray.shape).astype(np.float32)
            imgArray += noise # apply random noise to simulate real life.
            
            imgArray = np.clip(imgArray, 0, 255)
            imgArray = imgArray.astype(np.uint8)

            imgPath = os.path.join(digitDir, f'{digit}_{i}.png')
            cv2.imwrite(imgPath, imgArray)

generateBarcodeImages()