import cv2
import matplotlib.pyplot as plt
from projects.number_detectors.main import model,scaler
# Load your custom image using OpenCV
custom_image_path = "./download (2).png"
custom_image = cv2.imread(custom_image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to match the input shape of your model (8x8 for digits dataset)
custom_image_resized = cv2.resize(custom_image, (8, 8))

# Flatten the resized image array to match the input format of your model
custom_image_flattened = custom_image_resized.flatten().reshape(1, -1)

# Normalize the flattened image using the same scaler used for training
custom_image_normalized = scaler.transform(custom_image_flattened)

# Make predictions using your trained model
predicted_label = model.predict(custom_image_normalized)

# Print the predicted label
print("Predicted Label:", predicted_label[0])

# Show the custom image
# plt.gray()
# plt.imshow(custom_image_resized)
# plt.title("Custom Image")
# plt.axis('off')
# plt.show()
