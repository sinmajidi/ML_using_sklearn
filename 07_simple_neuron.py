import random


class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, input_val):
        # Weighted sum of input
        return input_val * self.weight + self.bias


# Example usage:
if __name__ == "__main__":
    # Create neuron with weight and bias
    weight = random.random()
    bias = 0
    neuron = Neuron(weight, bias)

    # Input value
    input_val = 0.5

    # Perform forward pass
    output = neuron.forward(input_val)
    print("Output:", output)
