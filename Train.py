from Tetris_env import Tetris, Tetrino
from Network import Network
import copy
import random
import numpy as np
from tqdm import tqdm


state_size = 4
layer_sizes = [32, 32]
output_size = 1
activations = ['relu', 'relu', 'linear']
loss = 'mse'
optimizer = 'adam'
discount = 0.95
exp_replay_size = 20000
pop_replay_rand_moves = 1000
minibatch_size = 256
epsilon = 0.1
epsilon_min = 1e-3
episode_end_decay = 1000
epochs = 1
replay_min_train_size = 1000
num_training_episodes = 2000
save_every = 250

agent = Network(input_size=state_size, layer_sizes=layer_sizes, output_size=output_size, activations=activations, loss=loss,
                optimizer=optimizer, discount=discount, exp_replay_size=exp_replay_size, pop_replay_rand_moves=pop_replay_rand_moves,
                minibatch_size=minibatch_size, epsilon=epsilon, epsilon_min=epsilon_min, episode_end_decay=episode_end_decay)
# agent.load("Trained_model.h5")  # Retrain the current trained model

# target_net = copy.deepcopy(agent)
env = Tetris()

high_score = 0
games_played = 0
avg_score = 0
min_terminate_score = 250000
scores = []
training = True

recent_high_score = 0
recent_50 = 0

for i in tqdm(range(num_training_episodes)):
    # Prepare a new Tetris environment
    env.reset()
    state = [0] * state_size
    done = False

    # Play
    while not done:
        next_states = env.get_next_states(env.game_board)
        execute_random_move = random.random() <= epsilon
        actions = list(next_states.keys())

        if execute_random_move or len(agent.exp_replay) < pop_replay_rand_moves:
            move = random.randint(0, len(next_states)-1)
            action = actions[move]

        else:
            q_values = agent.predict(list(next_states.values())) # ->
            action = agent.get_best_action(actions, q_values) # ->

        reward, done = env.execute_action(action, show=False)
        agent.exp_replay.append([state, action, next_states.get(action), reward, done])

        if not done:
            state = next_states.get(action)

    scores.append(env.score)
    recent_50 += env.score
    if env.score > recent_high_score:
        recent_high_score = env.score

    if games_played % 50 == 0:
        print("High Score last 50 episodes: ", recent_high_score)
        print("Avg: ", recent_50 / 50)
        recent_high_score = 0
        recent_50 = 0

    # Train
    if len(agent.exp_replay) > replay_min_train_size:
        agent.train(epochs=epochs)
    games_played += 1

    if games_played % save_every == 0:
        filename = "Models/Ep" + str(games_played) + ":" + str(max(scores)) + ".h5"
        agent.save(filename)
        print("scores", scores)
        print("high score: ", max(scores))
        scores = []


