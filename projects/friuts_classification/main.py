import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Function to load and process images from a folder and assign labels based on subfolder names
def load_images_from_folder(main_folder, img_size=(300, 300)):
    images = []
    labels = []
    classes = os.listdir(main_folder)  # Each sub-folder corresponds to a class label
    
    for class_name in classes:
        class_folder = os.path.join(main_folder, class_name)
        if os.path.isdir(class_folder):  # Ensure it's a folder
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is not None:
                    img_resized = cv2.resize(img, img_size)  # Resize image to 300x300
                    img_flattened = img_resized.flatten()    # Flatten to 1D array
                    images.append(img_flattened)
                    labels.append(class_name)  # Use the folder name as the label
    
    return np.array(images), np.array(labels)  # Return images and labels as NumPy arrays

# Load training data
train_folder_path = 'dataset/train'
X_train, y_train = load_images_from_folder(train_folder_path)

# Load testing data
test_folder_path = 'dataset/test'
X_test, y_test = load_images_from_folder(test_folder_path)
print(X_train.shape,y_train.shape)
print(f"Loaded {X_train.shape[0]} training images of shape {X_train.shape[1]}.")
print(f"Loaded {X_test.shape[0]} testing images of shape {X_test.shape[1]}.")

# Standardize the features (X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initialize and train the MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=32, max_iter=50)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate accuracy
print('Training accuracy:', accuracy_score(y_train, y_train_pred) * 100, '%')
print('Testing accuracy:', accuracy_score(y_test, y_test_pred) * 100, '%')

# Example prediction with a single image from training set
print('Prediction for one sample:', model.predict([X_test[10]]),y_test[10])
