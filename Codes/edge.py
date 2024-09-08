'''
Adarsh Gourab Das
2141004066
'''

import cv2
import numpy as np
import os

input_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Sample Dataset'
output_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Saved Images\T2Edge'

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'): 
        # Load the image
        image_path = os.path.join(input_directory, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f'Could not read image: {filename}')
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 100, 200)

        # Save the edge-detected image
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, edges)

        print(f'Processed and saved edges for: {filename}')

print('All images processed.')