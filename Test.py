from Tetris_env import Tetris
from Network import Network
import matplotlib.pyplot as plt

show = True  # Show visuals of the game being play
scores = []
num_test_runs = 50

agent = Network()
agent.load("trained_model.h5")

for i in range(num_test_runs):
    env = Tetris()
    while env.playing:
        next_states = env.get_next_states(env.game_board)
        actions = list(next_states.keys())
        q_values = agent.predict(list(next_states.values()))  # ->
        action = agent.get_best_action(actions, q_values)  # ->
        reward, done = env.execute_action(action, show)
        if done:
            env.playing = False
    print("Score:", env.score)
    scores.append(env.score)

max_score = max(scores)
avg = sum(scores)/num_test_runs
print("Max:", max_score, "\tAvg:", avg)

plt.plot(scores)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()






