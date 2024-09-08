'''
Adarsh Gourab Das
2141004066
'''

import cv2
import numpy as np
import os

input_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Sample Dataset'
output_directory = 'D:\Personal Projects\Celebal Technologies\submissions\Week 10\Saved Images\T4Enhancement'

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        image_path = os.path.join(input_directory, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f'Could not read image: {filename}')
            continue

        alpha = 1.5  # Contrast control (1.0 for original image)
        beta = 30  # Brightness control (0-100)
        enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Apply Gaussian blur for noise reduction
        kernel_size = (5, 5)
        sigma = 0
        blurred_image = cv2.GaussianBlur(enhanced_image, kernel_size, sigma)

        # Apply histogram equalization for contrast enhancement
        gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray)

        # Sharpen the image using unsharp masking
        kernel = np.array([[-1, -1, -1], 
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened_image = cv2.filter2D(equalized_image, -1, kernel)

        # Save the enhanced image
        output_path = os.path.join(output_directory, f'enhanced_{filename}')
        cv2.imwrite(output_path, sharpened_image)

        print(f'Processed and saved enhanced image for: {filename}')

print('All images processed.')