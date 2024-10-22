
# Machine Learning and Image Processing with Python

This repository contains Python scripts that demonstrate machine learning techniques using `scikit-learn` and image processing using `cvzone` and `OpenCV`.

## Requirements

Install the required packages by running:
```bash
pip install -r requirements.txt
```

### `requirements.txt`:
```
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.1
matplotlib==3.9.0
cvzone==1.6.1
opencv-python==4.9.0.80
mediapipe==0.9.1.0
```

## File Overview

### Machine Learning Scripts:
- **00_numpy.py**: Basic NumPy operations.
- **01_pandas.py**: Data manipulation with Pandas.
- **03_matplotlib.py**: Data visualization with Matplotlib.
- **04_linear_regression.py**: Linear regression example.
- **05_decision_tree_and_randomforest.py**: Decision tree and random forest classifiers.
- **06_logestic_regression.py**: Logistic regression for classification.
- **07_simple_neuron.py**: Implementation of a simple neuron.
- **08_simple_neuron_train.py**: Training a basic neural network.
- **09_ANN_classification.py**: ANN for classification tasks.
- **10_ANN_pca.py**: PCA combined with ANN.
- **11_ANN_regression.py**: ANN for regression tasks.

### Image Processing Scripts:
- **12_hand_detection.py**: Hand detection using `cvzone` and `OpenCV`.
- **12_1_virtualkeyboard.py**: Virtual keyboard controlled by hand gestures.
- **13_facemesh.py**: Face landmark detection.
- **14_pose_detection.py**: Pose detection using `cvzone`.

### Dataset:
- **heart_statlog_cleveland_hungary_final.csv**: Dataset for heart disease classification.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run any script: `python <filename>.py`
