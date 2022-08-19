import numpy as np

speeds = np.load('expert_data/eps_0/expert_speeds.npy')
commands = np.load('expert_data/eps_0/expert_commands.npy')
obs = np.load('expert_data/eps_0/expert_states.npy')
actions = np.load('expert_data/eps_0/expert_actions.npy')

print(speeds.dtype, speeds.shape)
print(commands.dtype, commands.shape)
print(obs.dtype, obs.shape)
print(actions.dtype, actions.shape)

