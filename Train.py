from Tetris_env import Tetris
from DQN import DNN, Layer

import random
import pygame
import numpy as np
from collections import deque

input_layer_size = 267
size_hidden_layer = 16
num_hidden_layers = 4
output_layer_size = 6
iterations = 25
scores = []

pygame.display.init()
env = Tetris()
env.render()
env.draw_board()
dnn = DNN(input_layer_size, num_hidden_layers, size_hidden_layer, output_layer_size)

for i in range(iterations):

    while env.playing:
        state = env.get_state_as_arr()
        action = dnn.get_best_action(state)  # Most of the time, but add a random factor
        reward = env.execute_curr_state(action)
        env.draw_board()
        # print(action, state)

    print("test")
    dnn.backprop()
    dnn.draw_nn(colored=False)
    scores.append(env.score)
    dnn.reset()
pygame.quit()
print("Scores: ", scores)
