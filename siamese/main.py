"""
Main file. Now just runs a little test.

TODO: Build the component that compares tracks
"""

import argparse
import logging
import torch
import torch.nn.functional as F
import torchvision
import torchvision.utils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from data_loader import SiameseNetworkDataset
from helpers import (
    imshow,
    # show_plot,
    UnNormalize)
# from loss import QuadrupletLoss
from logger import configure_logger
from model import Quadruplet

logger = logging.getLogger(__name__)
logger = configure_logger(logger, debug=True)

parser = argparse.ArgumentParser()
parser.add_argument('--cpu','-cpu', action='store_true', default=False, help='Run on CPU?')
args = parser.parse_args()

if args.cpu:
    net = Quadruplet(args.cpu)
    net.load_state_dict(torch.load('weights/modelsiamesetrip2405.pt', map_location=torch.device('cpu')))
else:
    net = Quadruplet().cuda()
    net.load_state_dict(torch.load('weights/modelsiamesetrip2405.pt'))

transformation = transforms.Compose(
    [transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    ])

#remove warnings
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

un_norm = UnNormalize(mean=(0.485, 0.456, 0.406, 0.440), std=(0.225, 0.225, 0.225, 0.225))

# Locate the test dataset and load it into the SiameseNetworkDataset
folder_dataset_test = datasets.ImageFolder(root="./testimages")
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transformation)

test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)

def run_test():
    # Grab one image that we are going to test
    dataiter = iter(test_dataloader)
    x0, _, _, _ = next(dataiter)

    for i in range(20):

        # Iterate over 20 images sets and test them with the first image (x0)
        _, x1, x2, x3 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0,x1,x2,x3))

        if args.cpu:
            output1, output2, output3, output4 = net(x0,x1,x2,x3)
        else:
            output1, output2, output3, output4 = net(x0.cuda(),x1.cuda(), x2.cuda(), x3.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance2 = F.pairwise_distance(output1, output3)
        euclidean_distance3 = F.pairwise_distance(output1, output4)

        if euclidean_distance <= 2:
            verdict = "same"

        imshow(
            torchvision.utils.make_grid(un_norm(concatenated)[0:4]),
            f'Dissimilarity1: {euclidean_distance.item():.3f},      \
            Dissimilarity2: {euclidean_distance2.item():.3f},       \
            Dissimilarity2: {euclidean_distance3.item():.3f}')

    # siamese_dataset = SiameseNetworkDataset(
    #     imageFolderDataset=folder_dataset_test,
    #     transform=transformation)

print(siamese_dataset.imageFolderDataset.classes)

if __name__ == '__main__':
    run_test()

    #logger back
    logger.setLevel(old_level)

# import random
# Als de similarity boven dit getal is dan zijn ze anders
# DISSIMILARITY_THRESHOLD = 1
# CLASSES = len(siamese_dataset.imageFolderDataset.classes)

# Dit is een neppe similarity-vergelijker
# Output 1 of 2
# def random_similarity():
#   return random.randint(0,2)

# while True:
