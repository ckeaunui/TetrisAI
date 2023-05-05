import random
import numpy as np
from collections import deque
import math
from shapes import Paper, Rectangle, Oval, Arrow, Text


class Layer:
    def __init__(self, input_size, output_size):
        # self.weights = 3 * (np.random.rand(output_size, input_size) - 0.5)
        # self.biases = 2 * (np.random.rand(output_size) - 0.5)
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.ones((output_size, input_size))
        self.biases = np.zeros(output_size)
        self.Z = np.zeros(output_size)
        self.activations = np.zeros(output_size)

    def forward(self, X):
        self.Z = np.dot(X, self.weights.T) + self.biases
        return self.Z

    def activation_ReLU(self, Z):  # Returns activated Z
        a = np.maximum(Z, 0)
        return a

    def activation_linear(self, Z):  # Activation for output layer
        a = Z
        return a

    def d_activation_ReLU(self, Z):
        return (Z > 0) * 1  # Bool into 0 or 1

    def d_activation_linear(self, Z):
        return 1

    def print_layer(self):
        print("Weights: ", self.weights)
        print("Bias: ", self.biases)


class DNN:
    def __init__(self, size_input_layer, num_hidden_layers, size_hidden_layer, size_output_layer):
        self.active_layer = 0
        self.size_input_layer = size_input_layer
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layer = size_hidden_layer
        self.size_output_layer = size_output_layer
        self.layers = np.empty(self.num_hidden_layers + 1, dtype=object)  # Hidden layers, one output layer, ignore input layer
        self.reset()

    def reset(self):
        self.active_layer = 0
        self.layers = np.empty(self.num_hidden_layers + 1, dtype=object)
        self.size_hidden_layer = self.size_hidden_layer
        self.num_hidden_layers = self.num_hidden_layers
        self.layers[0] = Layer(self.size_input_layer, self.size_hidden_layer)
        for i in range(1, self.num_hidden_layers):
            layer_i = Layer(self.size_hidden_layer, self.size_hidden_layer)
            self.layers[i] = layer_i
        self.layers[self.num_hidden_layers] = Layer(self.size_hidden_layer, self.size_output_layer)

    def print_layers(self):
        for i in range(len(self.layers)):
            print("Layer: ", i)
            self.layers[i].print_layer()

    def forward_step(self, X):
        Z_unactivated = np.array(self.layers[self.active_layer].forward(X))
        if self.active_layer == self.num_hidden_layers + 1:
            Z_activated = Z_unactivated
        else:
            Z_activated = self.layers[self.active_layer].activation_ReLU(Z_unactivated)
        self.layers[self.active_layer].activations = Z_activated
        return Z_activated

    def forward(self, state):
        q_values = np.asarray(state)
        for i in range(self.num_hidden_layers + 1):
            q_values = self.forward_step(q_values.T)
            self.activations = q_values
            self.active_layer += 1
        return q_values

    def get_action(self, q_values):
        move = np.argmax(q_values)
        self.active_layer = 0
        return move

    def get_loss(self, y_pred, y):  # Returns a vector of each nodes error within current layer
        loss = (np.asarray(y_pred) - np.asarray(y)) ** 2
        return loss  # a vector of how wrong each element in the layer is

    def get_policy(self):
        weights = []
        biases = []
        for l in range(len(self.layers)):
            weights.append(self.layers[l].weights)
            biases.append(self.layers[l].biases)
        return weights, biases

    def set_policy(self, policy):
        return

    def backprop(self, x, y):  # Full cycle for backprop.  Returns what this network wants
        alpha = 0.9  # Learning rate
        self.active_layer = self.num_hidden_layers

        while self.active_layer > -1:  # For each layer in reversed order
            j_size = self.layers[self.active_layer].output_size
            k_size = self.layers[self.active_layer].input_size
            dC_da_prev_l = np.zeros((k_size,))  # (k, 1) self.layers[self.active_layer - 1]

            for j in range(j_size):  # for each node in L
                a_j = self.layers[self.active_layer].activations[j]
                Z_j = self.layers[self.active_layer].Z[j]

                if self.active_layer == self.num_hidden_layers:
                    d_sigma_j = self.layers[self.active_layer].d_activation_linear(Z_j)

                else:
                    d_sigma_j = self.layers[self.active_layer].d_activation_ReLU(Z_j)

                dC_db_j = d_sigma_j * 2 * (a_j - y[j])
                print(dC_db_j)

                # Update bias
                self.layers[self.active_layer].biases[j] -= alpha * dC_db_j
                for k in range(k_size):  # For each node in L-1
                    if self.active_layer == 0:
                        a_prev = x[k]  # Input x for layer 0
                    else:
                        a_prev = self.layers[self.active_layer - 1].activations[k]
                    w_jk = self.layers[self.active_layer].weights[j][k]

                    # j is index of right layer, k is index of left node
                    dC_dW_jk = a_prev * d_sigma_j * 2 * (a_j - y[j])
                    dC_dx_k = w_jk * d_sigma_j * 2 * (a_j - y[j])  # X = activation of node in L-1
                    dC_da_prev_l[k] += dC_dx_k
                    self.layers[self.active_layer].weights[j][k] -= alpha * dC_dW_jk

            if self.active_layer != 0:
                self.active_layer -= 1
                a = self.layers[self.active_layer].activations
                y = a - alpha * dC_da_prev_l  # New predicted
            else:
                return


# dnn = DNN(input_layer_size, num_hidden_layers, size_hidden_layer, output_layer_size)
dnn = DNN(5, 2, 1, 2)
x = [1, 0, 1, 2, 3]  # Input
j = 0


while j < 5:
    q = [0, 21]  # q = correct q values
    q_pred = dnn.forward(x)  # Last layer predicted q values

    loss = np.sum(dnn.get_loss(q_pred, q))  # Loss is constant since activation doesnt change
    print("Loss: ", loss)
    print("q", q_pred)
    dnn.backprop(x, q)
    # print("Q", dnn.layers[-1].activations)
    w, b = dnn.get_policy()
    j += 1






