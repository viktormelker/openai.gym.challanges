# openai.gym.challanges
This repository contains attempts to solve various challanges from https://gym.openai.com/

## Environment
Use python 3.6, because tensorflow does not work with 3.7

## Launch simulation
I.e.
```sh
python -m app.lunar_lander.lunar_lander_v2_keras
```

## Running in docker
To avoid having to set up [keras](https://keras.io/) and [tensorflow](https://www.tensorflow.org/) with dependencies the simulations can be run in docker (without GUI).

To enter a shell in a docker image perform the following steps:

1. `docker build . -t keras_image`

2. `docker run -it keras_image bash`

After this the simulations can be launched as normally.


## Policies
Several different policies are implemented in the [policy](app/policy.py). They all have different advantages.

### RandomPolicy
Takes random actions all the time, can be used for benchmarking.

### SimplePolicy
Write it!

### QTablePolicy
Implements [Q-learning](https://en.wikipedia.org/wiki/Q-learning) and keeps a (Q) value for each possible state. This value gets updated as the algorithm learns more about the problem.
This policy works well when there is a discrete number (which is not too big) of states. When the states are not discrete or too many use [DQN](#DQNAgent) or [DDQN]() instead.

### DQNAgent
Implements a `Deep Q network` (DQN) algorithm. This is similar to the [QTablePolicy](#QTablePolicy) except that instead of storing a (Q) value for every possible state it approximates using a (deep) [Neural network](https://en.wikipedia.org/wiki/Artificial_neural_network). The network takes the state as input and generates the Q value as output.
In the case of the QTablePolicy the Q table was continuously updated as we gained more knowledge about the environment. In the DQN case the neural network is continuously updated to better approximate the proper Q value.

#### Features
1. This agent uses [Experience Replay](#Experience-replay) to improve performance.
2. The weights of the model can be saved after training.
3. A [potential function](#Potential-function) can be used.


### DDQNAgent
Write it up!

### Experience replay
Write it!

### Potential function
A potential function is a function that puts a value on a specific state of the environment. The value should represent how "good" it is to be in that state.
Using a potential function is good when the environment is complicated and it is hard/unlikely that the agent will reach a state which gives a reward using random exploration. The potential function will aid the algorithm to know which states are more likely to lead to a future reward.

#### Potential function values
The absolute value of a potential function is not important but the difference in value between 2 consequtive states is important. This is because the potential of one state is always compared to that of another.
When using `Q-learning` this `potential difference` (between states) is used in combination with the reward that can be received. Therefore it is important that the `potential difference` is not (much?) bigger than the true rewards so that it overshadows them. After all the only purpose of the `potential function` is to guide the algorithm to the rewards. The values must also be "big enough" to be relevant in comparison with the rewards.

