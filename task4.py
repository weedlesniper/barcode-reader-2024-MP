
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
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"

import os
from task1 import run_task1
from task2 import run_task2
from task3 import run_task3
from task1_util import clearDirectory
def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        content.save(output_path)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def run_task4(image_dir, config):
    outputFilePath = 'output/task4'
    clearDirectory(outputFilePath)

    # Step 1: Run task1 to detect and preprocess the barcode
    print(f"{MAGENTA}TASK 1 Barcode Digit Extraction \n ___________________________________________________________________{RESET}")
    run_task1(image_dir, config)
    
    # Step 2: Run task2 to extract individual digit images
    print(f"{MAGENTA}TASK 2 Digit Segmentation\n ___________________________________________________________________{RESET}")
    run_task2('output/task1', config)
    
    # Step 3: Run task3 to predict the digits from the segmented barcode
    print(f"{MAGENTA}TASK 3 Digit Recognition \n ___________________________________________________________________{RESET}")
    run_task3('output/task2', config)
    
    # Create output directory for task4
    print(f"{MAGENTA}TASK 4 Completing the pipeline \n ___________________________________________________________________{RESET}")
    os.makedirs(outputFilePath, exist_ok=True)
    
    # Step 4: Concatenate predicted digits for each barcode directory in task3
    task3_output_dir = 'output/task3'

    # Iterate through each image file in the validation directory
    image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]
    
    for idx, image_file in enumerate(image_files):
        image_name = os.path.splitext(image_file)[0]  # Get the base name without extension
        
        # Get the corresponding barcode directory (e.g., barcode1, barcode2)
        barcode_dir_path = os.path.join(task3_output_dir, f'barcode{idx + 1}')
        
        # If the barcode directory doesn't exist (i.e., no barcode detected), handle the case
        if not os.path.exists(barcode_dir_path):
            print(f"{YELLOW}No barcode detected for {image_name}!{RESET}")
            continue
        
        print(f"Processing barcode for image: {image_name}")
        barcode_number = ""
        
        # Iterate through each .txt file in the barcode directory (predicted digits)
        for txt_file in sorted(os.listdir(barcode_dir_path)):
            if txt_file.endswith('.txt'):
                txt_file_path = os.path.join(barcode_dir_path, txt_file)
                
                # Read the predicted digit from each .txt file
                with open(txt_file_path, 'r') as f:
                    digit = f.read().strip()  # Strip whitespace/newlines
                    barcode_number += digit
        
        # Save the full predicted barcode as barcodeX = imgY.txt in the output folder
        barcode_output_file = os.path.join(outputFilePath, f'barcode{idx + 1}.txt')
        with open(barcode_output_file, 'w') as output_file:
            output_file.write(f"{barcode_number}")
        print(f"Barcode prediction saved: barcode{idx + 1} = {barcode_number}")

    print(f"{MAGENTA}TASK 4 completed! Predictions saved in {outputFilePath}{RESET}")



