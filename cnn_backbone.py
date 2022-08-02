import torch
import torch.nn as nn
import numpy as np


# layer initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class CNNBackbone(nn.Module):
    def __init__(self, n_channels):
        super(CNNBackbone, self).__init__()
        self.conv = nn.Sequential(
            Scale(1 / 255),
            layer_init(nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        W, H = 256, 144
        for i in range(4):
            W = (W - 4) // 2 + 1
            H = (H - 4) // 2 + 1
        fc_dim = 256 * W * H

        self.fc = nn.Sequential(
            layer_init(nn.Linear(fc_dim, 512)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))
