import random

import numpy as np
import pygame
from Tetris_env import Tetris
from collections import deque


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size * output_size) - 0.5
        self.bias = 0
        self.num_nodes = output_size

    def forward(self, X):  # Not working
        print("X", X.shape)
        print("W", self.weights.shape)
        Z = np.dot(X, self.weights) + self.bias
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
        print("Weights", self.weights)
        print("Bias: ", self.bias)


class DNN:
    def __init__(self, input_layer_size, num_hidden_layers, size_hidden_layer, output_layer_size):
        self.active_layer = 0
        self.layers = np.empty(num_hidden_layers + 1, dtype=object)
        self.size_hidden_layer = size_hidden_layer
        self.num_hidden_layers = num_hidden_layers
        self.layers[0] = Layer(input_layer_size)
        for i in range(1, num_hidden_layers):
            layer_i = Layer(size_hidden_layer)
            self.layers[i] = layer_i
        self.layers[-1] = Layer(output_layer_size)

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

        input_layer_activation = self.forward_step(state)
        layer_1_activation = self.forward_step(input_layer_activation)
        layer_2_activation = self.forward_step(layer_1_activation)
        layer_3_activation = self.forward_step(layer_2_activation)
        layer_4_activation = self.forward_step(layer_3_activation)
        output_layer_activation = self.forward_step(layer_4_activation)
        move = np.argmax(output_layer_activation)
        print(move)
        print(len(output_layer_activation))


        return move

