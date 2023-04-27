import numpy as np
from scipy import signal
import math


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((n_neurons, 1))

    def forward(self, X):
        pass  # Sets Z

    def d_activation(self, Z):
        pass

    def backprop(self, a_prev, y, alpha):
        pass

    def get_one_hot(self, index):
        pass


class Dense(Layer):
    def forward(self, X):
        self.Z = np.sum(X * self.weights + self.biases, axis=1)
        self.A = np.maximum(0, self.Z)

    def d_activation(self, Z):
        return (Z > 0) * 1

    def backprop(self, a_prev, y, alpha):
        d_sigma = self.d_activation(self.Z)
        a_prev = np.array(a_prev)
        dC_dZ = 2 * d_sigma * (y - self.A)  # = dA/dZ * dC/DA
        dC_dZ = dC_dZ.reshape(self.n_neurons, 1)
        dC_dW = dC_dZ * a_prev
        dC_dB = dC_dZ
        dC_dA = np.dot(self.weights.T, dC_dZ)
        a_prime = (a_prev.T - alpha * dC_dA).reshape(1, self.n_inputs)
        return a_prime, dC_dW, dC_dB

    def get_one_hot(self, index):
        one_hot = np.zeros((self.n_neurons, 1))
        one_hot[index] = 1
        return one_hot.T


class OutputLinear(Layer):
    def forward(self, X):
        self.Z = np.sum(X * self.weights + self.biases, axis=1)
        self.A = self.Z

    def d_activation(self, Z):
        return np.ones((self.n_neurons, 1))

    def backprop(self, a_prev, y, alpha):
        d_sigma = self.d_activation(self.Z)
        a_prev = np.array(a_prev)
        dC_dZ = 2 * d_sigma * (y - self.A)  # = dA/dZ * dC/DA
        dC_dZ = dC_dZ.reshape(self.n_neurons, 1)
        dC_dW = dC_dZ * a_prev
        dC_dB = dC_dZ
        dC_dA = np.dot(self.weights.T, dC_dZ)
        a_prime = (a_prev.T - alpha * dC_dA).reshape(1, self.n_inputs)
        return a_prime, dC_dW, dC_dB

    def get_one_hot(self, index):
        one_hot = np.zeros((self.n_neurons, 1))
        one_hot[index] = 1
        return one_hot.T


class OutputSigmoid(Layer):
    def forward(self, X):
        self.Z = np.sum(X * self.weights + self.biases, axis=1)
        self.A = 1 / (1 + np.exp(-self.Z))
        self.A /= self.n_neurons

    def d_activation(self, Z):
        fx = 1 / (1 + np.exp(-Z))
        return fx * (1 - fx)

    def backprop(self, a_prev, y, alpha):
        d_sigma = self.d_activation(self.Z)
        a_prev = np.array(a_prev)
        dC_dZ = 2 * d_sigma * (y - self.A)  # = dA/dZ * dC/DA
        dC_dZ = dC_dZ.reshape(self.n_neurons, 1)
        dC_dW = dC_dZ * a_prev
        dC_dB = dC_dZ
        dC_dA = np.dot(self.weights.T, dC_dZ)
        a_prime = (a_prev.T - alpha * dC_dA).reshape(1, self.n_inputs)
        return a_prime, dC_dW, dC_dB

    def get_one_hot(self, index):
        one_hot = np.zeros((self.n_neurons, 1))
        one_hot[index] = 1
        return one_hot.T


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.feature_map = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                print(self.input[j].shape)
                print(self.kernels[i, j].shape)
                self.feature_map[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        self.max_pooling(self.feature_map)
        self.Z = self.max_pool.flatten()
        self.forward_activation(self.Z)
        return self.A

    def max_pooling(self, feature_map):  # Assumes even dimensions.  Occurs if input image is of even dimensions
        if len(feature_map[0]) % 2 == 1:  # Feature map odd dimensions
            raise ValueError("Dimensions Error: Input image sizes must be even")
        max_pool_size = math.ceil(len(feature_map[0]) / 2)
        print("shape: ", feature_map.shape)
        self.max_pool = np.zeros((feature_map.shape[0], max_pool_size, max_pool_size))
        print("Mshape", self.max_pool.shape)
        for batch in range(feature_map.shape[0]):
            for i in range(max_pool_size):
                for j in range(max_pool_size):
                    pool = [feature_map[batch][i][j], feature_map[batch][i][j+1],
                            feature_map[batch][i+1][j], feature_map[batch][i+1][j+1]]
                    self.max_pool[batch][i][j] = np.max(pool)
        print("MAX", self.max_pool)

    def forward_activation(self, Z):
        self.A = np.maximum(0, Z)

    def d_activation(self, Z):
        self.d_sigma = (Z > 0) * 1

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        print("input shape", self.input.shape)
        print("kernel shape", self.kernels.shape)
        print("grad shape", output_gradient.shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                print("input shape", self.input.shape)
                print("kernel shape", self.kernels.shape)
                print("grad shape", output_gradient.shape)
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class DNN:
    def __init__(self):
        self.alpha = 0.05
        self.gamma = 0.05
        self.layers = []
        self.net_len = 0

    def build_layer(self, layer):
        self.layers.append(layer)
        self.net_len += 1

    def build_nn(self, n_neurons_input, n_hidden_layers, n_neurons_hidden, n_neurons_output):  # Build a fully connected nn
        self.layers.insert(0, Dense(n_neurons_input, n_neurons_hidden))
        for l in range(1, n_hidden_layers):
            self.layers.insert(l, Dense(n_neurons_hidden, n_neurons_hidden))
        self.layers.insert(n_hidden_layers, OutputSigmoid(n_neurons_hidden, n_neurons_output))

    def forward(self, X):
        for i in range(len(self.layers)):
            self.layers[i].forward_Z(X)
            self.layers[i].forward_activation(self.layers[i].Z)
            X = np.array([self.layers[i].A])

    def backprop(self, X, Y):
        dC_dW = []
        dC_dB = []
        for i in reversed(range(len(self.layers))):
            if i == 0:
                a_prev = X
            else:
                a_prev = np.array([self.layers[i - 1].A])
            Y, dw, db = self.layers[i].backprop(a_prev, Y, self.alpha)
            dC_dW.insert(0, dw)
            dC_dB.insert(0, db)
        return dC_dW, dC_dB

    def update_policy(self, dC_dW, dC_dB):
        for i in range(self.net_len):
            self.layers[i].weights -= self.alpha * dC_dW[i]
            self.layers[i].biases -= self.alpha * dC_dB[i]

    def get_policy(self):
        W = []
        B = []
        for i in range(self.net_len):
            W.append(self.layers[i].weights)
            B.append(self.layers[i].biases)
        return W, B

    def print_policy(self):
        for i in range(self.net_len):
            print("Layer", i)
            print("Weights: \n", self.layers[i].weights)
            print("Biases: \n", self.layers[i].biases)

    def get_action(self, q_values):
        move = np.argmax(q_values)
        return move



"""
image = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 11, 0, 0, 11, 0, 0],
                  [0, 0, 11, 0, 0, 11, 0, 0],
                  [0, 0, 11, 11, 11, 11, 0, 0],
                  [0, 0, 0, 0, 0, 11, 0, 0],
                  [0, 0, 0, 0, 0, 11, 0, 0],
                  [0, 0, 0, 0, 0, 11, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

# Channel size of one since its grayscale.  Only one channel on img so only one filter

dnn = DNN()
dnn.build_layer(Conv(8))  # Output size = (X/2 - 1)^2 = flattened max_pool length
dnn.build_layer(FullConnection(9, 3))
dnn.build_layer(FullConnection(3, 3))
dnn.build_layer(OutputSigmoid(3, 10))

X = image
Y = dnn.layers[-1].get_one_hot(4)

for layer in range(dnn.net_len):
    dnn.layers[layer].forward_Z(X)
    Z = dnn.layers[layer].Z
    dnn.layers[layer].forward_activation(Z)
    A = dnn.layers[layer].A
    X = A

# dnn.backprop(X, Y)

dnn.layers[0].backprop(image, Y, 0.1)
"""

