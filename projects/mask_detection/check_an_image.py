import cv2
import numpy as np
import pickle

# Load the trained model and scaler
with open('mask_detector_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to predict if a picture has a mask or not
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img_resized = cv2.resize(img, (100, 100))  # Resize image
    img_flattened = img_resized.flatten()  # Flatten to 1D array
    img_scaled = scaler.transform([img_flattened])  # Scale the image
    prediction = model.predict(img_scaled)  # Predict
    return "With Mask" if prediction[0] == 1 else "Without Mask"

# Example usage
image_path = 'without_mask2.jpg'  # Specify your image path here
result = predict_image(image_path)
print(f'The image is: {result}')
