import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# input image size is 32 x 32 CIFAR 10

class VGG16b(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.features = torchvision.models.vgg16(pretrained=True, progress=True).features
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
