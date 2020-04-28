import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils import utils

# for inputs argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='./checkpoints/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('--epoch', type=int, default=10, help='epoch size for training')
parser.add_argument('--img_size', type=int, default=64, help='image size for training')
parser.add_argument('-f', '--freq', type=int, default=200, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for training')
parser.add_argument('--lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_class', type=int, default=10, help='number of classes to classify of datasets')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--model', required=True, type=str, help='beta parameters for adam optimizer')

# classes index for CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    args = parser.parse_args()

    # make folder for experiment
    today = utils.get_today()
    save_dir = Path(args.save_dir) / today
    utils.make_folder(save_dir)

    # summary writer
    writer = SummaryWriter(save_dir)

    # transform data
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load data
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)

    # data loader
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)

    # using GPU
    model = utils.check_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # load pretrained models
    if args.pretrained is not '':
        model.load_state_dict(args.pretrained)

    # set train mode
    model.train()

    # set Cross entropy Loss and optimizer for Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta))

    # training section
    for epoch in range(args.epoch):
        running_loss = 0.0
        epoch_start_time = time.time()

        # for 1 epoch
        for i, (inputs, labels) in enumerate(trainloader):
            iter_start_time = time.time()
            # zero the parameter gradients
            optimizer.zero_grad()

            # using cuda library to train
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)


            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss for tensorboard
            running_loss += loss.item()
            if i % args.freq == args.freq - 1:
                print('epoch: {}, iteration:{}/{} loss: {:0.4f}, batch_time: {:0.4f}'.format(epoch + 1,
                                                                                             i + 1,
                                                                                             len(trainloader),
                                                                                             running_loss / args.freq,
                                                                                             time.time() - iter_start_time))

                # add scalar in tensorboard
                writer.add_scalar('training loss', running_loss / args.freq, epoch * len(trainloader) + i)
                writer.add_figure('predictions vs. actuals',
                                  utils.plot_classes_preds(model, inputs, labels, batch_size=args.batch_size),
                                  global_step=epoch * len(trainloader) + i)

                running_loss = 0.0

        # print for every epoch
        print('{} epoch is end, epoch time : {:0.4f}'.format(epoch + 1, time.time() - epoch_start_time))

        # save net in every 10 epochs
        if epoch % 10 == 9:
            save_filename = save_dir / '{}_{}epochs.pth'.format(args.model, epoch + 1)
            torch.save(model.state_dict(), save_filename)

    print('----- training is done -----')
