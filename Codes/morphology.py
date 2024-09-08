'''
Adarsh Gourab Das
2141004066
'''

import cv2
import os
import numpy as np

input_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Sample Dataset'
output_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Saved Images\T5Morphology'

os.makedirs(output_directory, exist_ok=True)

operation = cv2.MORPH_OPEN 
kernel_size = (5, 5)  
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        image_path = os.path.join(input_directory, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f'Could not read image: {filename}')
            continue

        # Apply the morphological operation
        morphed_image = cv2.morphologyEx(image, operation, kernel)

        # Save the morphed image
        output_path = os.path.join(output_directory, f'morphed_{filename}')
        cv2.imwrite(output_path, morphed_image)

        print(f'Processed and saved morphed image for: {filename}')

print('All images processed.')