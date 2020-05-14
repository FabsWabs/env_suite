# env_suite
The project **env_suite** is a collection of environments to test Reinforcement Learning Agents on. The structure of these environments is equal to the ones presented in the [OpenAI Gym][open] tool kit.

## Contents of this document
 - [env_suite](#env_suite)
    - [Basics](#basics)
    - [Installation](#installation)
    - [Environments](#environments)
    - [Tests](#tests)
    - [Citing](#citing-the-project)
    - [Contribute](#contribute)

## Basics
The package is designed to help you testing your Reinforcement Learning (RL) Agents. The counterpart to the RL Agent is the Environment the Agent is interacting with. 
Such as the environments presented in the [OpenAI Gym][open] tool kit, the environments included in this package are of the Env. For the purpose of interacting with an environment one should know the following class methods:
 - reset(self): Reset the environment's state. Returns observation.
 - step(self, action): Step the environment by one timestep. Returns observation, reward, done, info.
 - render(self, mode='human'): Render one frame of the environment. The default mode will do something human friendly, such as pop up a window.

## Installation
The simplest way of installing is:
```sh
$ git clone https://github.com/smilefab/env_suite.git
$ cd env_suite
$ pip install -e .
```
Right now the package is only available on GitHub. A minimal install of the packaged version directly from PyPI isn't planned but may be possible in the future.

## Environments
### pushBox
The goal of the pushBox environment is - as the name suggests - to push a box to a goal square, which is randomly placed in the environment. The world is therefore discretized into n * n grid cells. The can move up, down, left or right. If the agent runs against a wall, it's position simply won't change. If the box gets pushed next to a wall, the environment reaches a terminal state as it isn't solvable anymore.
There are two modes for the observation to choose from:
 - mode = 'vector': The observation includes the number of cells (in both directions) between the agent and the box, and between the box and the goal. 
 - mode = 'image': As an observation, an image of the current state gets returned as an *RGB-Array*. This can be used for training Agents based on *Convolutional Neural Networks* (CNN)
 
![gif not available](https://github.com/smilefab/env_suite/blob/master/data/videos/DQN_pushBox.gif | width=200)

<img src="https://github.com/smilefab/env_suite/blob/master/data/videos/DQN_pushBox.gif" width="300" height=320">


The animation above shows a DQN-agent solving the environment

### controlTableLine
In the controlTableLine environment the agent has to follow a twodimensional trajectory by applying forces and thereby changing it's momentum. As an observation the agent gets returned:

 - x , y :  current x- and y coordinates in metres, each entry is element of [-1,1]
 - x' , y' : current velocity in metres per second
 - dx , dy : distance to next point of the trajectory in metres, each entry is element of [-2,2]

Based on the observation, the agent can apply forces in x- and y-direction.

![Alt Text](https://github.com/smilefab/env_suite/blob/master/data/videos/PPO_controlTableLine.gif)

The animation above shows a PPO-agent following the trajectory.

## Tests
A number of example scripts is supplied within the *tests*-Folder. The interact*.py scripts for example allow the user to directly interact with the environment. The show*.py scripts are written to demonstrate the pretrained agents. Feel free to tinker around with the implementations.
**NOTE**: For a large part of the testing scripts you need to have [Stable Baselines][stable] installed.

## Citing the Project
To cite this repository in publications:

```
@misc{env_suite,
  author = {Wahren,Fabian},
  title = {env_suite},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/smilefab/env_suite}},
}
```

## Contribute

For feedback, suggestions or questions feel free to contact me at: fabianwahren@gmail.com

[//]: #
 [open]: <https://github.com/openai/gym>
 [stable]: <https://github.com/hill-a/stable-baselines>