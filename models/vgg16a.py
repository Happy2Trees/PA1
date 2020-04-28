import torch
import torch.nn as nn
import torch.nn.functional as F


# https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273
# input image size is 32 x 32 CIFAR 10

def conv2d(in_channels, out_channels, kernel_size, stride=1,
           padding=0, dilation=1, groups=1,
           bias=True, padding_mode='zeros'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode),
        nn.ReLU(True)
    )


class VGG16a(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.features = nn.Sequential(
            conv2d(3, 64, 3, padding=1),
            conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d((2, 2)),

            conv2d(64, 128, 3, padding=1),
            conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d((2, 2)),

            conv2d(128, 256, 3, padding=1),
            conv2d(256, 256, 3, padding=1),
            conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d((2, 2)),

            conv2d(256, 512, 3, padding=1),
            conv2d(512, 512, 3, padding=1),
            conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d((2, 2)),

            conv2d(512, 512, 3, padding=1),
            conv2d(512, 512, 3, padding=1),
            conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
