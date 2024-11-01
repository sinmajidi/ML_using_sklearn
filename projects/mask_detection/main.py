import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# Function to load and process images from a specified folder in grayscale
def load_images_from_folder(main_folder, img_size=(100, 100)):
    images = []
    labels = []
    
    for class_name in os.listdir(main_folder):
        class_folder = os.path.join(main_folder, class_name)
        if os.path.isdir(class_folder):
            label = 1 if class_name == 'WithMask' else 0  # 1 for with mask, 0 for without mask
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is not None:
                    img_resized = cv2.resize(img, img_size)  # Resize image
                    img_flattened = img_resized.flatten()  # Flatten to 1D array
                    images.append(img_flattened)
                    labels.append(label)  # Use the folder name as the label
    
    return np.array(images), np.array(labels)

# Load datasets from the specified folders
train_folder_path = 'Face Mask Dataset/Train'
val_folder_path = 'Face Mask Dataset/Validation'
test_folder_path = 'Face Mask Dataset/Test'

X_train, y_train = load_images_from_folder(train_folder_path)
X_val, y_val = load_images_from_folder(val_folder_path)
X_test, y_test = load_images_from_folder(test_folder_path)

# Print shapes of the loaded datasets
print("Training data shape:", X_train.shape, y_train.shape)
print("Validation data shape:", X_val.shape, y_val.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Standardize the features (X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize and train the MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=100)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print('Validation accuracy:', val_accuracy * 100, '%')

# Make predictions on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Testing accuracy:', test_accuracy * 100, '%')


# Save the trained model to a file using pickle
with open('mask_detector_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler as well
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved!")