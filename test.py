import numpy as np
from carla.libcarla import Location


speeds = np.load('expert_data/eps_0/expert_speeds.npy')
print(speeds)

l = Location(x=0.0, y=0.0, z=0.0)
print(l)
l.x += 1
print(l)