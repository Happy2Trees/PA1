import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import utils.utils as utils

# from https://github.com/KangBK0120/CAM
# generate class activation mapping for the top1 prediction
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets', metavar='PATH', help='path of directory')
parser.add_argument('--save_dir', type=str, default='./results', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('-f', '--freq', type=int, default=1000, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for testing')
parser.add_argument('--num_class', type=int, default=10, help='number of classes to classify of datasets')
parser.add_argument('--model', required=True, type=str, help='beta parameters for adam')


def returnCAM(feature_conv, weight_softmax, class_idx, image_size):
    # from https://github.com/KangBK0120/CAM/blob/master/create_cam.py
    size_upsample = (image_size[1], image_size[0])
    b, nc, h, w = feature_conv.shape
    # weight softmax x convolution weight and normalize
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = (255 * cam / np.max(cam)).astype(np.uint8)
    output_cam = cv2.resize(cam_img, size_upsample)
    return output_cam


if __name__ == '__main__':
    args = parser.parse_args()

    # make folder
    today = utils.get_today() + '_cam'
    save_dir = Path(args.save_dir) / today
    utils.make_folder(save_dir)

    # transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #load CIFAR10 DATASETS
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    #check model and load data
    model = utils.check_model(args.model)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.pretrained), strict=False)
    model.eval()

    # get weight only from the last layer(linear)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())




    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = inputs.to(device)
            outputs = model(inputs)

            _, idx = torch.max(outputs, dim=1)
            image_size = (inputs[0].size(1), inputs[0].size(2))

            # get cam using convolution features
            CAMs = returnCAM(model.features(inputs), weight_softmax, idx[0].item(), image_size)

#            np.expand_dims(CAMs, axis=0)

            heatmap = cv2.applyColorMap(CAMs, cv2.COLORMAP_JET)


            # save the input image
            input_image = (inputs[0].numpy().transpose((1, 2, 0)) / 2 + 0.5) * 255
            input_image = Image.fromarray(input_image.astype(np.uint8))
            filename = save_dir/'test_img_{:05d}.png'.format(i + 1)
            input_image.save(filename.absolute())

            input_image = cv2.imread(str(filename))


            #Apply results and save it in the result folders
            result = 0.4 * heatmap + 0.6 * input_image
            savename = save_dir/'heatmap_{:05d}.jpg'.format(i+1)
            heatname = save_dir/'cam_{:05d}.jpg'.format(i+1)

            cv2.imwrite(str(savename), result)
            cv2.imwrite(str(heatname), heatmap)

            print('{:05d}.png is saved !'.format(i + 1))
