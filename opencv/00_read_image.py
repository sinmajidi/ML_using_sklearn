import cv2

# Load an image from a file
image = cv2.imread('images/fruit.jpg')

print(image.shape)
print(image)
# Display the image in a window
cv2.imshow('Loaded Image', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
