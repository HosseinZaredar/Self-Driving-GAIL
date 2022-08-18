import numpy as np
import torch

speeds = np.load('expert_data/eps_0/expert_speeds.npy')
commands = np.load('expert_data/eps_0/expert_commands.npy')
obs = np.load('expert_data/eps_0/expert_states.npy')
actions = np.load('expert_data/eps_0/expert_actions.npy')

print(speeds.dtype)
print(commands.dtype)
print(obs.dtype)
print(actions.dtype)

