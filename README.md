# Self-Driving-GAIL

In this project, an end-to-end autonomous driving system in implemented. A deep learning model is trained to imitate an expert driver's behavior using Generative Adversarial Imitation Learninig (GAIL) algorithm [[1]](#1). The project is developed with [PyTorch](https://pytorch.org/) and [CARLA Simulator](https://carla.org/).

This is the final project for my BSc in Computer Engineering at Amirkabir University of Technology (AUT), September 2022.

## System's Inputs and Outputs

The system works in and end-to-end manner [[2]](#2). At each moment, the model takes as inputs the images of three RGB cameras, a high-level navigational command (that instructs the vehicle to "turn left", "turn right", or "go straight") [[3]](#3), and the current speed of the vehicle. Based on these inputs, the model directly produces three control signals, namely, Throttle, Steer, and Brake, which are used to drive the car.

<div align="center">
    <img src="figures/blackbox.png" width="500" alt="blackbox">
</div>

## Camera Setup

The camera setup is as follows: There is one front camera and two wide-angle cameras on the left and right sides of the vehicle.

Left Camera             |  Front Camera        |  Right Camera
:-------------------------:|:-------------------------:|:-------------------------:
![Left Camera](figures/left-camera.png)  |  ![Front Camera](figures/front-camera.png) | ![Right Camera](figures/right-camera.png)


## System Architecture

The architecure is based on the Reinforcement Learning loop. The agent learns to drive from interations with the environment and also, from the expert driving dataset. The Actor-Critic algorithm, PPO [[4]](#4), is used to implement the agent. 

<div align="center">
    <img src="figures/system.png" width="500" alt="system">
</div>

Three learning signals are used to train the model:
- **Generative Adversarial Imitaion Learning**: The model learns from the rewards predicted by a Discriminator that tries to distinguish bewteen expert driving behavior and that of the agent.
- **Behavioral Clonining**: The agent has direct access to the expert dataset and learns to imitate expert decisions through supervised learning [[5]](#5).
- **Explicit Rewards**: In order to help the model avoid obstacles and keep the vehicle in its lane, a negative reward is produced whenever a lane invasion occures.

## Network Architecure

**Agent Network**: A shared network is used for Actor and Critic modules of the PPO agent. The model processes the camera images separately, and fuses then these sensory informations to get a unified state vector. Then, based on the given high-level command, one head of the model is chosen to generate the control signals and the state value.

<br>

<div align="center">
    <img src="figures/actor-critic.png" width="700" alt="actor-critic">
</div>

<br>

**Discriminator Network**: This is similar to the Agent network, but takes the action as an input as well, and outputs the probability of the state-action pair belonging to the expert dataset.

<div align="center">
    <img src="figures/disc.png" width="700" alt="disc">
</div>

## Dataset

The training and testing is performed in Town 2 of CARLA simluator. The model is trained on intersections 1-6, and intersections 7-8 are used for testing the performance of the system on unseen routes.

<div align="center">
    <img src="figures/town-2.jpg" width="300" alt="town-2">
</div>

<br>
At each intersection, the expert data is gathered on two left turns and two right turns (in total, 24 turns). CARLA's open-source navigation agent and PID controller are used to generate the expert data automatically.

<br>
For simplicity, it is assumed that there's no traffic on the roads and traffic lights are not taken into consideration.

## Sample Results

**Long Route:** A simple route in Town 2 which includes turns the model has not seen during training. We can observe that the model has learned to follow high-level commands and drives safely.

https://user-images.githubusercontent.com/36497794/229384560-59c0bf97-97eb-4109-93a2-4fef26303c4e.mp4

<br>

**Noisy Steering Wheel:** In this experiment, the trained model is tested on a simulated vehicle whose steering wheel randomly turns right. As we can see, the system is able to react quickly and keep the vehicle in its lane. 

https://user-images.githubusercontent.com/36497794/229385027-5c74754b-a69c-4f39-a32e-6077b3b5d2bd.mp4

<br>
What's interesting is that the model has not seen this type of error in the expert dataset or in the training environment. But due to the inherent exploration present in the stochastic policy of PPO, and also, the explicit reward explained before, the model has learned by itself to handle such errors.

## How to Run
A guide to setup the training environment and run the codes, along with the trained models will be added soon...

## Reference Papers
- <a id="1">[1]</a> [Generative Adversarial Imitation Learning, NIPS (2016)](https://arxiv.org/abs/1606.03476)
- <a id="2">[2]</a> [End to End Learning for Self-Driving Cars, arXiv (2017)](https://arxiv.org/abs/1604.07316)
- <a id="3">[3]</a> [End-to-end Driving via Conditional Imitation Learning, ICRA (2018)](https://arxiv.org/abs/1710.02410)
- <a id="4">[4]</a> [Proximal Policy Optimization Algorithms, arXiv (2017)](https://arxiv.org/abs/1707.06347)
- <a id="5">[5]</a> [Augmenting GAIL with BC for sample efficient imitation learning, PMLR (2021)](https://arxiv.org/abs/2001.07798)
- [Generative Adversarial Imitation Learning for End-to-End Autonomous Driving on Urban Environments, SSCI (2021)](https://arxiv.org/abs/2110.08586)
