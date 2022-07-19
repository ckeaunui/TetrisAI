import random

import numpy as np
import pygame
from Tetris_env import Tetris
from collections import deque
import math
from shapes import Paper, Rectangle, Oval, Arrow, Text
# Install Graphviz


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = 3 * (np.random.rand(output_size, input_size) - 0.5)
        self.biases = 2 * (np.random.rand(output_size) - 0.5)
        self.output_size = output_size

    def get_layer_size(self):
        return self.output_size

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
    def __init__(self, size_input_layer, num_hidden_layers, size_hidden_layer, size_output_layer):
        self.active_layer = 0
        self.size_input_layer = size_input_layer
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layer = size_hidden_layer
        self.size_output_layer = size_output_layer
        self.layers = np.empty(self.num_hidden_layers + 1, dtype=object)  # Hidden layers, one input, one output layer
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

    def backprop(self, y, y_prime):  # Y is the score, y prime is the desired value
        # Find partial derivative: dE/dy with respect to the input dE/dX, X is the layers input
        # E = y_prime, E is a scalar value, Y, X are vectors  (Soon each will go up by one dim for batches)
        learning_rate = 0.99 
        return y - y_prime

    def get_best_action(self, state):
        Z = state
        for i in range(self.num_hidden_layers + 1):
            Z = self.forward_step(Z.T)
        move = np.argmax(Z)
        self.active_layer = 0
        return move

    def get_loss(self):
        return 1

    def draw_nn(self):  # Draw the neural network to display its policy (red/black = negative, white/green = positive)
        board_height = 750
        board_width = 750
        node_size = (board_height / self.size_input_layer+1) * 0.5
        paper = Paper(width=board_width, height=board_height)
        num_layers = len(self.layers)

        # Get locations to draw to
        input_coords = []
        for i in range(self.size_input_layer):
            y = i / self.size_input_layer * board_height + node_size/2
            input_coords.append([node_size/2, y])
        coords = []
        for i in range(num_layers):
            num_nodes = self.layers[i].get_layer_size()
            x_shift = (board_width / num_layers) / 2 - node_size/2
            x = (i + 1) / (num_layers + 1) * board_width + x_shift #+ node_size/2
            layer_coords = []
            for j in range(num_nodes):
                y_shift = (board_height / num_nodes) / 2 - node_size/2
                y = j / num_nodes * board_height + y_shift
                layer_coords.append([x, y])
            coords.append(layer_coords)

        # Draw input nodes
        layer_size = self.size_input_layer
        for j in range(layer_size):  # For each current node
            x, y = input_coords[j]
            for k in range(self.size_hidden_layer):  # For each next node
                x2, y2 = coords[0][k]
                arrow = Arrow()
                if self.layers[0].weights[0][j] >= 0:
                    arrow.set_color(color='green')  # Link color
                else:
                    arrow.set_color(color='red')
                width = 1 / (1 + np.exp(math.fabs(-self.layers[0].weights[0][j])))
                arrow.set_width(width)
                arrow.draw(x + node_size / 2, y + node_size / 2, x2 + node_size / 2, y2 + node_size / 2)
            node = Oval()
            node.set_x(x)
            node.set_y(y)
            node.set_color('gray')
            node.set_width(node_size)
            node.set_height(node_size)
            node.draw()

        # Draw rest
        for i in range(len(coords)):  # For each layer
            for j in range(len(coords[i])):  # For each node
                print(i, j)
                x, y = coords[i][j]
                if i+1 in range(len(coords)):  # If there is a next layer
                    for k in range(len(coords[i+1])):  # For each weight
                        x2, y2 = coords[i+1][k]
                        # print(x, y, "-->", x2, y2)
                        arrow = Arrow()
                        if self.layers[i].weights[j][k] >= 0:
                            arrow.set_color(color='green')  # Link color
                        else:
                            arrow.set_color(color='red')
                        width = 1 / (1 + np.exp(-math.fabs(self.layers[i].weights[j][k])))  # Width
                        arrow.set_width(2 * width)
                        arrow.draw(x+node_size/2, y+node_size/2, x2+node_size/2, y2+node_size/2)
                color = 'white' if self.layers[i].biases[j] > 0 else 'black'
                node = Oval()
                node.set_x(x)
                node.set_y(y)
                node.set_color(color)
                node.set_width(node_size)
                node.set_height(node_size)
                node.draw()

                # Draw node bias on top of node
                """bias = Text()
                color = 'white' if color == 'black' else 'black'
                bias.set_color(color)
                bias.set_x(x+node_size/2)
                bias.set_y(y+node_size/2)
                bias.draw(np.round(self.layers[i].biases[j], 2))"""
        paper.display()
        return


# dnn = DNN(input_layer_size, num_hidden_layers, size_hidden_layer, output_layer_size)
dnn = DNN(20, 5, 16, 6)
dnn.draw_nn()


