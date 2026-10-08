import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RotorNet(nn.Module):
    def __init__(self, hidden=5, width=64, tau_min=0.0, tau_max=1.0):
        super().__init__()
        self.register_buffer("t0", torch.tensor(float(tau_min)))
        self.register_buffer("t1", torch.tensor(float(tau_max)))
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(hidden - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        z = 2.0 * (t - self.t0) / (self.t1 - self.t0 + 1e-12) - 1.0
        return self.net(z)
