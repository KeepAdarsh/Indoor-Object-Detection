'''
Adarsh Gourab Das
2141004066
'''

import cv2
import numpy as np
import os

input_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Sample Dataset'
output_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Saved Images\T1Filter'

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'): 
        image_path = os.path.join(input_directory, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f'Could not read image: {filename}')
            continue

        # Define a 5x5 averaging kernel for blurring
        kernel_blur = np.ones((5, 5), np.float32) / 25

        # Apply the averaging kernel
        blurred_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_blur)

        # Save the filtered image
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, blurred_image)

        print(f'Processed and saved: {filename}')

print('All images processed.')