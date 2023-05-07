![tetris](https://user-images.githubusercontent.com/47498710/236661913-fadc7114-12df-46e5-8f87-1f656e55ef99.gif)


Test the pre-trained model over 50 episodes by running Test.py, or train a new model with Train.py.  The model is saved
every 250 training episodes in the 'Models' folder.

# Functional Requirements

> Python 3.10

> Run tensorflow backend to train keras networks.

> Rapidly handle large mattrix multiplication operations

# Architecture
![Agent_flowchart](https://user-images.githubusercontent.com/47498710/236660271-8a8ca922-dbe0-4add-94c7-594bc4d31357.jpg)

# Methods

The environment the agent will train on is a version of NES Tetris made in Python.  The agent will explore the environment and record its' observations at each timestep in an experience replay with the format <s, a, r, s', done>. The first 1,000 actions the model performs are fully random so that the experience replay will initially be populated by a diverse data set. 
 Afterwards, the model will predict the Q-value or expected reward of each next state, and take the action which corresponds with the highest value.  

When agents learn through self exploration they are susceptible to exclusively following the first decent path they discover, often  resulting in better states never being explored.  To help prevent this, a randomness factor called an exploration rate was included \cite{dimitri}. This value determines the likelihood of the next action being performed randomly instead of what the agent predicts to be the best move.  The exploration rate \(\epsilon\) will start at 0.1 and steadily decay until reaching a value of \(1e^{-3}\) after 1500 training episodes.  This allows the earlier states to be thoroughly explored which prevents getting trapped in local minima, while also allowing the model to train through exploiting its policy once it has become refined.

The boards state will be defined by a vector with the attributes: number of rows cleared by the previous action, number of holes, bumpiness (sum of height differences between adjacent columns), and total height.  It should be noted that the speed Tetrinos fall at increases with the level and therefore may seem important to include in the state, however the agent is able to place pieces  faster than the highest gravity level, so its policy will not be dependent on the game speed.

The model will use a Deep Q-Network to train itself to predict the Q-values for each possible (s,a) pair in a given state.  The neural network is comprised of an input layer, two fully connected hidden layers of size 32 with ReLU activation functions, and a fully connected output layer of size 1 with a linear activation function.  The input layer accepts a vector of size 4 to represent the state, and propagates the values forward throughout the network.  The input layer does not perform any computations on the input data. The output layer contains only one node, and uses the linear activation function to predict a states q-value.  The agent will use the adam optimizer to train by minimizing a mean squared error cost function on the predicted and actual Q-values.  The cost function will have a discount factor of 0.95.  The weights and biases will be tuned with a learning rate of 1e^{-3} .

An alternative approach to solving this problem is to predict Q-values with a convolutional neural network (CNN), which takes a scaled down image of the board as its input.  This has the advantage of not relying on hand picked data to represent the state, allowing it to find more subtle patterns during its training.  It is also computationally fast, and excels at processing grid-like data.  However as the prior work by Sai showed, these networks are not well suited to play Tetris due to spacial invariance.  A CNN may be able to perform well in this environment, but it is not the best fit model for solving the given task


# Evaluation Metrics

One metric I considered evaluating the model on was the number of rows it was able to clear in a single run.  The number of rows are correlated to both the level and score reached, and is a good indicator of the models ability to compactly organize a random assortment of Tetrinos.  However, one issue with this metric is it treats all line clears as equal.  Clearing several lines at once awards bonus points, with 4 lines cleared in a single move (aka a Tetris) awarding the most.  When humans play we have the added difficulty of reaction time, making the optimal strategy going for Tetris' exclusively.  If the total rows cleared was the metric the model was evaluated on, it would have no incentive to learn a policy which postpones immediate reward in favor of a larger future reward, and therefore could not learn to play optimally.  

Another metric that could judge the models quality is the level it reached.  However, the level increases after every 10 lines cleared, making it a flawed metric for the same reasons as the total number of rows cleared.

The metric used to assess the models overall performance will be the scores it can reach.  The way score is calculated can be seen in figure 2:
<img width="418" alt="Tetris_scores" src="https://user-images.githubusercontent.com/47498710/236660765-5f2ca31b-66bc-49a2-b5a1-e29cfe047593.png">

The score is an appropriate metric to use because maximizing it is the ultimate goal of Tetris.  Furthermore this is the metric humans base their own performance on, making it easy to compare the model to humans.  To get an accurate measurement of the models score, 50 games of Tetris will be ran with the results being displayed in a graph.  

# Results and Discussion

<img width="425" alt="Training_results" src="https://user-images.githubusercontent.com/47498710/236660777-d9f9adfb-7b40-4a55-9ede-d13f867eb58d.png">
<img width="631" alt="Test_results" src="https://user-images.githubusercontent.com/47498710/236660791-56ce6a1b-ecf1-41ac-a4dc-acb77e209ffe.png">

The results obtained from training were incredibly strong.  After training the model over 2,000 episodes, a policy was reached which is able to survive for extended periods of time.  After 50 test episodes, the model received an average score of 6.31 million with a high score of 25.64 million (See figure 4).  For reference, the Tetris world record for a human is 2.94 million.  These results greatly exceeded my expectations.  The model makes many moves which are not great, yet manages to consistently recover from bad positions and survive well into the later levels.

However, it should be noted that a large part of the models success comes from the ability to place pieces immediately, allowing it to ignore the increasing rate at which blocks fall.  At the quickest level(15) a tetrino will drop one tile every \(7e^{-3}\) seconds, but the model can move a tetrino to the correct horizontal position in an average time of \(3.5e^{-5}\) seconds.  Once there the model pushes the piece down itself, so gravity will never affect the tetrinos.  Gravity is the most difficult aspect of the game for humans to handle, yet the  model does not find survival at top speeds any more difficult than during the early stages.   

As figure 4 shows, the models performance is not the most consistent across episodes.  Sometime runs such as test run 50 scored as low as 60,000, while other runs such as 27 were able to significantly surpass the best human players.  This variance is likely a result of the randomness in the pieces the model receives.  Although the model has discovered a solid policy, receiving the right or wrong piece in a critical position could drastically change the position of the board, making its performance somewhat reliant on luck.

One way I would improve this model in the future is implementing next piece information, which would allow the model to see several sequential pieces rather than just the active piece.  This would increase the amount of (s,a) pairs for each state by a factor of ~23 per piece, but greatly increase the computer's ability to organize Tetrinos.

Another adaptation I would make in the future is giving the model the ability to store pieces.  Early iterations of the model ran slowly and had a difficult time training, so removing the ability to store pieces cut the number of (s,a) pairs in half.  This helped improve performance initially, but after further optimization was implemented, the model ran fast enough to handle this added action.  This would greatly improve the models performance, since it would give it twice as many options on where to place the next piece.

Another way I would improve upon the model is to use A* pathfinding to move the tetrinos into place.  This would allow the computer to make more advanced moves than are currently possible.  The current path finding algorithm is very elementary, and functions by moving the piece to the correct horizontal position before dropping it.  With A* the agent would have greater control of how the tetrino is placed, allowing it to perform moves such as block shifting, T-spins, and wall kicks.  Access to these additional actions should help increase the models ability to organize pieces.

