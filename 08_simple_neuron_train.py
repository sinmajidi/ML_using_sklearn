import random
class Neuron:
    def __init__(self):
        self.slope = random.random()  # Initialize slope randomly
        self.bias = random.random()   # Initialize bias randomly

    def forward(self, x):
        # Weighted sum of inputs
        weighted_sum = x * self.slope + self.bias
        # Activation of the weighted sum
        return weighted_sum

# Function to generate training data
def generate_data(x, y):
    data = []
    for i in range(len(x)):
        data.append((x[i], y[i]))  # Generate data as (x, y)
    return data

# Train the neuron
def train_neuron(neuron, data, learning_rate, epochs):
    for _ in range(epochs):
        for x, y in data:
            # Forward pass
            output = neuron.forward(x)
            # Update weights based on error
            neuron.slope += learning_rate * (y - output) * x
            neuron.bias += learning_rate * (y - output)

# Given values of x and y
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [-2 * xi - 5 for xi in x]

# Create neuron
neuron = Neuron()

# Generate training data
data = generate_data(x, y)

# Train the neuron
learning_rate = 0.01
epochs = 10000
train_neuron(neuron, data, learning_rate, epochs)

# Print the learned slope and bias
print("Learned Slope (m):", neuron.slope)
print("Learned Bias (b):", neuron.bias)


#
# import random
#
# class Neuron:
#     def __init__(self):
#         self.slope_x2 = random.random()  # Initialize slope for x^2 randomly
#         self.slope_x1 = random.random()  # Initialize slope for x randomly
#         self.bias = random.random()      # Initialize bias randomly
#
#     def forward(self, x):
#         # Weighted sum of inputs
#         weighted_sum = x**2 * self.slope_x2 + x * self.slope_x1 + self.bias
#         # Activation of the weighted sum
#         return weighted_sum
#
# # Function to generate training data
# def generate_data(x):
#     y = [(-200 * xi**2 )+ (-500 * xi) - 10 for xi in x]  # True values of y for given x
#     return list(zip(x, y))
#
# # Train the neuron
# def train_neuron(neuron, data, learning_rate, epochs):
#     for _ in range(epochs):
#         for x, y_true in data:
#             # Forward pass
#             y_pred = neuron.forward(x)
#             # Update weights based on error
#             neuron.slope_x2 += learning_rate * (y_true - y_pred) * (x ** 2)
#             neuron.slope_x1 += learning_rate * (y_true - y_pred) * x
#             neuron.bias += learning_rate * (y_true - y_pred)
#
# # Given values of x
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# # Create neuron
# neuron = Neuron()
#
# # Generate training data
# data = generate_data(x)
#
# # Train the neuron
# learning_rate = 0.0001  # Adjust learning rate
# epochs = 100000          # Adjust number of epochs
# train_neuron(neuron, data, learning_rate, epochs)
#
# # Print the learned slopes and bias
# print("Learned Slope for x^2 (m2):", neuron.slope_x2)
# print("Learned Slope for x (m1):", neuron.slope_x1)
# print("Learned Bias (b):", neuron.bias)
