import cv2

# Load an image
image = cv2.imread('images/fruit.jpg')

# Resize the image to 300x300 pixels
resized_image = cv2.resize(image, (300, 300))

print(resized_image.shape)

# Display the resized image
cv2.imshow('Resized Image', resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
