from Tetris_env import Tetris
from DQN import DNN, Layer

import random
import pygame
import numpy as np
from collections import deque


input_layer_size = 265
size_hidden_layer = 16
num_hidden_layers = 4
output_layer_size = 6




iterations = 25
scores = []
for i in range(iterations):

    dnn = DNN(input_layer_size, num_hidden_layers, size_hidden_layer, output_layer_size)
    # dnn.print_layers()

    input_layer = Layer(input_layer_size)

    env = Tetris()
    env.render()
    pygame.display.init()
    env.draw_board()
    while env.playing:
        state = env.get_state_as_arr()
        #print(state)
        move = dnn.get_best_action(state)  # Most of the time, but add a random factor

        # move = random.randint(0, 6)
        reward = env.execute_curr_state(move)

        env.draw_board()

        # dnn.get_best_action(state)
    scores.append(env.score)
print("Scores: ", scores)
