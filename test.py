# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # image = np.load('expert_data/obs_0.npy')
# # plt.imshow(image)
# # plt.show()
#
# expert_commands = np.load(os.path.join('expert_data', 'eps_0', 'expert_commands.npy'))
# print(expert_commands)
# #
# # import torch
# #
# # print(torch.__version__)
#
# # import dns.resolver
# #
# # my_resolver = dns.resolver.Resolver()
# #
# # # 8.8.8.8 is Google's public DNS server
# # my_resolver.nameservers = ['178.22.122.100', '185.51.200.2']
# #
# # answers = my_resolver.resolve('google.com')
# # for rdata in answers:
# #     print('Host', rdata.exchange, 'has preference', rdata.preference)

import torch

w = torch.tensor([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
])

o1 = torch.rand((1, 12))
o2 = torch.rand((5, 4))
o3 = torch.rand((5, 4))


out = torch.mul(w[:, 0:1], o1[:, 0:3])
print(w)
print(o1[:, 0:3])
# print(o2)
# print(o3)
print(out)
