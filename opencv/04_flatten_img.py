import cv2
import numpy as np

# Load an image (color image)
image = cv2.imread('images/fruit.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the shape of the image
print('Original shape:', gray_image.shape)  # Output will be (height, width, channels)

# Flatten the image into a 1D array
flattened_image = gray_image.flatten()

# Display the shape of the flattened image
print('Flattened shape:', flattened_image.shape)  # Output will be a 1D array, (total_pixels,)

# Example of accessing the first 10 elements of the flattened image
print(flattened_image[:10])
