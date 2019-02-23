### Report

By running the file `Navigation.ipynb` one can reproduce my results! 
Both the kernel and the dqn_agent file is based on Udacity nanodegree solutions.

The agent is defined in `dqn_agent.py` where I defined 2 q-networks - a local and a target - for a normal q-learning. The Gamma parameter is 0.99 (1 step future Q value is only considered with 99% weight), the target q-network is updated with Polyak averaging with Tau parameter of 0.001.

First I tried to create a Q-network that has a mu and a sigma parameter so I would be able to force the Policy network for exploration though it was not necessary for the solution. I left that code part in the `dqn_agent.py` file.

The final model architecture for the Q-value estimation builds up from 7 feedforward layers with sizes of [512, 256, 128, 64, 32, 32, 32] each followed by an ELU nonlieanarity except for the last layer where no nonlinearity was used. 
The first 2 layers are followed by a Dropout layer with 0.1 rate in order to make the model more robust. 

The update rule for the Q-function is a bit different from the normal Q-learning method: I have chosen to use the Policy network predictions on next_state to calculate the probabilities of each action and have weighted the different Q-values from the target Q-function with these values. 
This way I was able to defeat the model from overestimating the target values. After some episodes the Policy network became capable of predicting the highest valued action so the method became more-and-more close to the max(Q_target) calculation.

I have chosen to build a Policy network that predicted the maximum value of the Q-network. The loss function is cross-entropy. This way I was also able to make a stochastic policy acting as I have a probability for each action

The Q-network and the Policy network shares the same structure.

### Results

I was able to solve the environment in 51 episodes. Looking at the chart I think that with different seed it would be possible to solve it below 50 episodes. 

![Solved in 51 episodes][image2]

The saved weights are `policy.pth` and `qnetwork.pth`. 
