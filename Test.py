from Tetris_env import Tetris
from Network import Network

env = Tetris()
agent = Network()
agent.load("Trained_model.h5")
show = True
while env.playing:
    next_states = env.get_next_states(env.game_board)
    actions = list(next_states.keys())
    q_values = agent.predict(list(next_states.values()))  # ->
    action = agent.get_best_action(actions, q_values)  # ->
    reward, done = env.execute_action(action, show)
    if done:
        env.playing = False

print(env.score)
