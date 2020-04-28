import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Alexnet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.features = torchvision.models.alexnet(pretrained=True, progress=True).features
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
