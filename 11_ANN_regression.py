import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X = np.arange(-10, 10, 0.05).reshape(-1,1)

y =5*X**3 + X**2 + X + 1
noise = np.random.normal(0, 1000, y.shape)  
y_noisy = y + noise  


X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2)
mlp = MLPRegressor(hidden_layer_sizes=(128,128,128), activation='relu', max_iter=500)
mlp.fit(X_train, y_train)
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')


plt.scatter(X_test, y_test, color='blue', label='Real Data')
plt.scatter(X_test, y_pred_test, color='red', label='Predicted Data')
plt.title('Real vs Predicted Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


plt.scatter(X_train, y_train, color='blue', label='Real Data')
plt.scatter(X_train, y_pred_train, color='red', label='Predicted Data')
plt.title('Real vs Predicted Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()