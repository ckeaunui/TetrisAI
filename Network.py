from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from keras.models import load_model
from collections import deque
import random
import numpy as np


class Network:
    # Create a new network with set hyper parameters
    def __init__(self, input_size=4, layer_sizes=None, output_size=1, activations=None,
                 loss='mse', optimizer='adam', discount=0.95, exp_replay_size=15000, pop_replay_rand_moves=1000,
                 minibatch_size=512, epsilon=1, epsilon_min=0, episode_end_decay=1000):
        if layer_sizes is None:
            layer_sizes = [32, 32]
        if activations is None:
            activations = ['relu', 'relu', 'linear']
        self.input_size = input_size
        self.layer_sizes = layer_sizes  # Excluding input layer 0
        self.output_size = output_size
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        self.discount = discount

        self.exp_replay = deque(maxlen=exp_replay_size)
        self.pop_replay_rand_moves = pop_replay_rand_moves
        self.minibatch_size = minibatch_size

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.episode_end_decay = episode_end_decay
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.episode_end_decay

        self.model = self._build_model()

    # Compile the models
    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_dim=self.input_size, activation=self.activations[0]))
        for i in range(1, len(self.layer_sizes)):
            model.add(Dense(self.layer_sizes[i], activation=self.activations[i]))
        model.add(Dense(1, activation=self.activations[-1]))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    # Take in board state, returns each states predicted Q value
    def predict(self, input_data):  # input dictionary {(x, rotation): [next_state], ... }
        return self.model.predict(input_data, verbose=0)

    def get_best_action(self, actions, q_values):
        max_value = max(q_values)
        best_actions = []
        for i in range(len(q_values)):
            if q_values[i] == max_value:
                best_actions.append(actions[i])
        best = random.randint(0, len(best_actions)-1)  # In the case several moves have the same q value, pick randomly
        return best_actions[best]

    # Returns a random batch of samples from the exp replay
    def get_batch(self):
        return random.sample(list(self.exp_replay), self.minibatch_size)

    # Train the model on the experience replay data
    def train(self, epochs):
        minibatch = self.get_batch()
        next_states = np.array([transition[2] for transition in minibatch])
        next_qs = self.model.predict(next_states, verbose=0)
        X = []
        y = []

        for i, (current_state, action, _, reward, done) in enumerate(minibatch):
            if not done:
                new_q = reward + self.discount * next_qs[i][0]
            else:
                new_q = reward
            X.append(current_state)
            y.append(new_q)
        self.model.fit(np.array(X), np.array(y), epochs=epochs, verbose=0)

        # Let exploration probability decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay * self.epsilon
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    # Save trained model
    def save(self, filename='Trained_model.h5'):
        self.model.save(filename)

    # Load new model
    def load(self, filename):
        self.model = load_model(filename)
