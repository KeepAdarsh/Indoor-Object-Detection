'''
Adarsh Gourab Das
2141004066
'''

import cv2
import numpy as np
import os

input_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Sample Dataset'
output_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Saved Images\T3Transformation'

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        image_path = os.path.join(input_directory, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f'Could not read image: {filename}')
            continue

        resized_image = cv2.resize(image, (300, 300)) 

        # Rotate the image by 45 degrees
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip

        # Save the transformed images
        cv2.imwrite(os.path.join(output_directory, f'resized_{filename}'), resized_image)
        cv2.imwrite(os.path.join(output_directory, f'rotated_{filename}'), rotated_image)
        cv2.imwrite(os.path.join(output_directory, f'flipped_{filename}'), flipped_image)

        print(f'Processed and saved transformations for: {filename}')

print('All images processed.')