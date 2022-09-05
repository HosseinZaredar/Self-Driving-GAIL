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
        self.conv1 = nn.Sequential(
            Scale(1 / 255),
            layer_init(nn.Conv2d(in_channels=n_channels//3, out_channels=32, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

        self.conv2 = nn.Sequential(
            Scale(1 / 255),
            layer_init(nn.Conv2d(in_channels=n_channels//3, out_channels=32, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

        self.conv3 = nn.Sequential(
            Scale(1 / 255),
            layer_init(nn.Conv2d(in_channels=n_channels//3, out_channels=32, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
            layer_init(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

        W, H = 256, 144
        for i in range(4):
            W = (W - 4) // 2 + 1
            H = (H - 4) // 2 + 1
        fc_dim = 3 * 256 * W * H

        self.fc = nn.Sequential(
            layer_init(nn.Linear(fc_dim, 512)),
            nn.LeakyReLU(),
            nn.Dropout(0.3) if dropout else nn.Identity()
        )

    def forward(self, x):
        c1 = self.conv1(x[:, 0:3])
        c2 = self.conv2(x[:, 3:6])
        c3 = self.conv3(x[:, 6:9])
        return self.fc(torch.cat([c1, c2, c3], dim=1))
