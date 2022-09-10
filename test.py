import numpy as np

cmd = np.load('expert_data/eps_0/expert_commands.npy')

for i in range(cmd.shape[0]):
    print(i, cmd[i])
