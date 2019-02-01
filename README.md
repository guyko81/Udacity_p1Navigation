[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://github.com/guyko81/Udacity_p1Navigation/blob/master/episodes51.png "Solved in 51 episodes"
# Project 1: Navigation

### Introduction

This project is a Udacity Reinforcement Learning Nanodegree project, where we had to train an agent to navigate (and collect bananas!) in a large, square world. Below is a trained agent collecting bananas.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of our agent was to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent had to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent had to get an average score of +13 over 100 consecutive episodes.

### Getting Started

In order to use this model one has to download and install the Udacity files provided below. The instructions are the following:

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Report

By running the file `Navigation.ipynb` one can reproduce my results!  

The agent is defined in `dqn_agent.py` where I defined 2 q-networks - a local and a target - for a normal q-learning. The Gamma parameter is 0.99 (1 step future Q value is only considered with 99% weight), the target q-network is updated with Polyak averaging with Tau parameter of 0.001.

I have chosen to build a Policy network that predicted the maximum value of the Q-network. The loss function is cross-entropy. This way I was able to:

1. make a stochastic policy acting as I have a probability for each action
2. the target q-network of next-step can be calculated as the weighted average of values from target q-network - in this way it is possible to avoid Q value overestimation

The Q-network and the Policy network shares the same structure.

First I tried to create a Q-network that has a mu and a sigma parameter so I would be able to force the Policy network for exploration though it was not necessary for the solution. I left that code part in the `dqn_agent.py` file.

### Results

I was able to solve the environment in 51 episodes. Looking at the chart I think that with different seed it would be possible to solve it below 50 episodes. 

The saved weights are `policy.pth` and `qnetwork.pth`. 
