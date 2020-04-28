import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from models.alexnet import Alexnet
from models.vgg16_cam import VGG16_cam2
from models.vgg16a import VGG16a
from models.vgg16b import VGG16b

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def check_model(model):
    models = ['vgg16a', 'vgg16b', 'alexnet', 'vgg16a_cam', 'vgg16b_cam', 'alexnet_cam']
    if model == 'vgg16a':
        return VGG16a()
    elif model == 'vgg16b':
        return VGG16b()
    elif model == 'vgg16_cam2':
        return VGG16_cam2()
    else:
        return Alexnet()


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_today():
    now = time.localtime()
    output = '{:02d}-{:02d}-{:02d}-{:02d}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    return output

def make_folder(folder_name):
    Path(folder_name).mkdir(parents=True, exist_ok=True)


def images_to_probs(net, images):
    '''
    reference = https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())

    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, batch_size):
    '''
    reference = https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(7, 2))
    for idx in np.arange(10):
        ax = fig.add_subplot(1, 10, idx+1, xticks=[], yticks=[])
        plt_imshow(images[idx].cpu())
        ax.set_title("{0},\n{1:.1f}%\n{2}".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def plt_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img * 0.5 + 0.5     # unnormalize
    npimg = img.numpy()
    output = np.transpose(npimg, (1, 2, 0))
    plt.imshow(output)
