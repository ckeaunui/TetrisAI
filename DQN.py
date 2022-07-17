import random

import numpy as np
import pygame
from Tetris_env import Tetris
from collections import deque


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.biases = np.zeros(output_size)
        self.output_size = output_size

    def forward(self, X):  # Not working
        Z = []
        for i in range(self.output_size):
            Z.append(np.dot(X, self.weights[i]) + self.biases[0])
        return Z

    def activation_ReLU(self, Z):  # Returns activated Z
        return np.maximum(Z, 0)

    def activation_sigmoid(self, Z):  # Returns output activations
        return 1 / (1 + np.exp(-Z))

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
        print("Weights: ", self.weights)
        print("Bias: ", self.bias)


class DNN:
    def __init__(self, input_layer_size, num_hidden_layers, size_hidden_layer, output_layer_size):
        self.active_layer = 0
        self.input_layer_size = input_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layer = size_hidden_layer
        self.output_layer_size = output_layer_size
        self.layers = np.empty(self.num_hidden_layers + 2, dtype=object)  # Hidden layers, one input, one output layer
        self.reset()

    def reset(self):
        self.active_layer = 0
        self.layers = np.empty(self.num_hidden_layers + 2, dtype=object)
        self.size_hidden_layer = self.size_hidden_layer
        self.num_hidden_layers = self.num_hidden_layers
        self.layers[0] = Layer(self.input_layer_size, self.size_hidden_layer)
        for i in range(1, self.num_hidden_layers):
            layer_i = Layer(self.size_hidden_layer, self.size_hidden_layer)
            self.layers[i] = layer_i
        self.layers[self.num_hidden_layers] = Layer(self.size_hidden_layer, self.output_layer_size)

    def set_layer(self, input_size, output_size, bias=0):
        self.layers[self.active_layer].weights = np.empty(input_size * output_size)
        self.layers[self.active_layer].bias = bias

    def print_layers(self):
        for i in range(len(self.layers)):
            self.layers[i].print_layer()

    def forward_step(self, X):
        Z_unactivated = np.array(self.layers[self.active_layer].forward(X))
        if self.active_layer == self.num_hidden_layers + 1:
            Z_activated = self.layers[self.active_layer].activation_ReLU(Z_unactivated)
        else:
            Z_activated = self.layers[self.active_layer].activation_sigmoid(Z_unactivated)
        self.active_layer += 1
        return Z_activated

    def get_best_action(self, state):
        Z = state
        # print(input_layer_activation)
        for i in range(self.num_hidden_layers + 1):
            Z = self.forward_step(Z.T)
        move = np.argmax(Z)
        self.active_layer = 0
        return move

    def get_loss(self):
        return 1

