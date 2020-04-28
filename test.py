import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from utils import utils
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='./results/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, required=True, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for test')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for test')
parser.add_argument('--num_class', type=int, default=10, help='number of classes to classify of datasets')
parser.add_argument('--model', required=True, type=str, help='beta parameters for adam')
parser.add_argument('--img_size', required=True, type=int, help='img size for test')
parser.add_argument('--classes', default=3, type=int, help='top classes to print')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    args = parser.parse_args()

    # make folder for present
    today = utils.get_today()
    filename = today + '_test'
    save_dir = Path(args.save_dir) / filename
    utils.make_folder(save_dir)

    # data transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    # load CIFAR10 datasets
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    # test loader
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)


    # check model and use GPU
    model = utils.check_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(args.pretrained))
    model.eval()

    # training section
    # calculate for top images and test results
    correct = 0
    total = 0
    class_correct = list(0. for i in range(args.num_class))
    class_total = list(0. for i in range(args.num_class))

    # for the top 3 image
    img_class = []
    scores_class = []
    t_class = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            # using cuda library to train
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # softmax for the outputs
            scores = F.softmax(outputs, 1)

            # prediction for wide classes and total classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            check = (predicted == labels).squeeze()

            # check condition, its class is equal to ground truth labels
            condition = (predicted == labels) & (predicted == args.classes)
            condition = condition.cpu().numpy()

            # if condition is checked, append this
            for j in range(args.batch_size):
                if condition[j]:
                    img_class.append(inputs[j].cpu())
                    scores_class.append(scores[j, predicted[j]].item())
                    t_class.append(predicted[j].item())


            for i in range(args.batch_size):
                label = labels[i]
                class_correct[label] += check[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the whole CIFAR-10 test images: {:.4f} %%'.format(100 * correct / total))
    for i in range(args.num_class):
        print('Accuracy of {:5s} : {:2.4f}%'.format(classes[i], 100 * class_correct[i] / class_total[i]))

    # sort the images
    arg = np.argsort(scores_class)[::-1]
    arg = arg[:3]


    # plot top 3 images
    f = 0
    fig = plt.figure(figsize=(4, 2))
    for idx in arg:
        ax = fig.add_subplot(1, 3, f + 1, xticks=[], yticks=[])
        utils.plt_imshow(img_class[idx])
        ax.set_title("{:0.02f}\n{:4s}".format(
            scores_class[idx] * 100, classes[t_class[idx]]),
            color="green")
        fig.savefig(save_dir / 'top5_scores_results.png')
        f += 1
    print('----- test is done -----')
