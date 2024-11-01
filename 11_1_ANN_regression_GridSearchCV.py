import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X = np.arange(-10, 10, 0.05).reshape(-1, 1)
y = 5 * X**3 + X**2 + X + 1
noise = np.random.normal(0, 1000, y.shape)  
y_noisy = y + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

# Define the MLPRegressor
mlp = MLPRegressor()

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (128, 128), (256,), (128, 128, 128)],
    'max_iter': [100, 200, 500]
}

# Create the GridSearchCV object with Mean Squared Error as the scoring method
grid_search = GridSearchCV(mlp, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Calculate Mean Squared Error
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f'Training MSE: {train_mse:.2f}')
print(f'Test MSE: {test_mse:.2f}')

# Display best parameters found by GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

# Plotting Real vs Predicted for Test Data
plt.scatter(X_test, y_test, color='blue', label='Real Data')
plt.scatter(X_test, y_pred_test, color='red', label='Predicted Data')
plt.title('Real vs Predicted Data (Test Set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Plotting Real vs Predicted for Train Data
plt.scatter(X_train, y_train, color='blue', label='Real Data')
plt.scatter(X_train, y_pred_train, color='red', label='Predicted Data')
plt.title('Real vs Predicted Data (Train Set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
