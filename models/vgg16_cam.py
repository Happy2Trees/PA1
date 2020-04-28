import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



# with GAP
class VGG16_cam2(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.features = torchvision.models.vgg16(pretrained=True, progress=True).features
        # Global Average Pooling
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
