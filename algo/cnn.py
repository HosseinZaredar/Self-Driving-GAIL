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


class CNN(nn.Module):
    def __init__(self, n_channels, dropout=False):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            Scale(1 / 255),
            layer_init(nn.Conv2d(in_channels=n_channels, out_channels=3*16, kernel_size=4, stride=2, groups=3)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=3*16, out_channels=3*32, kernel_size=4, stride=2, groups=3)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=3*32, out_channels=3*64, kernel_size=4, stride=2, groups=3)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=3*64, out_channels=3*128, kernel_size=4, stride=2, groups=3)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

        W, H = 256, 112
        for i in range(4):
            W = (W - 4) // 2 + 1
            H = (H - 4) // 2 + 1
        fc_dim = 3 * 128 * W * H

        self.fc = nn.Sequential(
            layer_init(nn.Linear(fc_dim, 512)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.fc(self.conv(x))
