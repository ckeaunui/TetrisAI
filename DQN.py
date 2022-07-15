import numpy as np
from Tetris_env import Tetris
from collections import deque


class Layer:
    def __init__(self, layer_size):
        self.weights = np.random.rand(layer_size) - 0.5
        self.bias = 0
        self.layer_size = layer_size

    def forward(self, X):  # Return unactivated Z
        Z = X.dot(self.weights) + self.bias
        return Z

    def activation_ReLU(self, Z):  # Returns activated Z
        return np.maximum(Z, 0)

    def softmax(self, Z):  # Activation for output layer
        return np.exp(Z) / np.sum(np.exp(Z))

    def loss(self, Z, Y):
        loss = Y - Z

    def one_hot(self, Y):
        one_hot_y = np.zeros((Y.size, Y.max() + 1))

    def back_prop_error(self, dZ, expected):
        errors = list()
        for i in reversed(range(len(self.layer_size))):
            errors.append(self.weights[i] * dZ[i])

    def print_layer(self):
        print("Weights", self.weights)
        print("Bias: ", self.bias)


class DNN:
    def __init__(self, input_layer_size, num_hidden_layers, hidden_layer_size, output_layer_size):
        self.active_layer = 0
        self.layers = np.empty(num_hidden_layers + 1, dtype=object)
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.layers[0] = Layer(input_layer_size)
        for i in range(1, num_hidden_layers):
            layer_i = Layer(size_hidden_layer)
            self.layers[i] = layer_i
        self.layers[-1] = Layer(output_layer_size)

    def print_layers(self):
        for i in range(len(self.layers)):
            self.layers[i].print_layer()

    def forward_step(self, X):
        Z_unactivated = self.layers[self.active_layer].forwward(X)
        Z_activated = self.layers[self.active_layer].activation_ReLU(Z_unactivated)
        return Z_activated

    def get_best_action(self, state):
        self.forward_step(state)


input_layer_size = 234
size_hidden_layer = 16
num_hidden_layers = 4
output_layer_size = 6

dnn = DNN(input_layer_size, num_hidden_layers, size_hidden_layer, output_layer_size)
dnn.print_layers()

input_layer = Layer(input_layer_size)

env = Tetris()
iterations = 100000

while env.playing:
    state = env.get_state_as_arr()
    dnn.get_best_action(state)


env.render()

