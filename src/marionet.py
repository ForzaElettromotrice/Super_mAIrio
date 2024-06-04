import torch
from torch import nn
import numpy as np

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        channels = input_dim[0]
        
        def build_cnn(c, output_dim):
            return nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
        )

        self.online = build_cnn(channels, output_dim)
        self.target = build_cnn(channels, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.online.to(self.device)
        self.target.to(self.device)

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        return self.target(input)
        