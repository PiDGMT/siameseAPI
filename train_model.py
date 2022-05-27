import torch
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import utils

from data_loader import SiameseNetworkDataset
from helpers import show_plot, UnNormalize, imshow
from loss import QuadrupletLoss
from model import Quadruplet

folder_dataset = datasets.ImageFolder(root="/content/drive/MyDrive/Person_reID_baseline_pytorch/Market-1501-v15.09.15/pytorch/train_all")

# Resize the images and transform to tensors
transformation = transforms.Compose(
    [transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ),
    ])

# Initialize the network
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transformation)

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# Create a simple dataloader just for simple visualization
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=2,
                        batch_size=8)

# Extract one batch
example_batch = next(iter(vis_dataloader))

# Example batch is a list containing 4x8 images, first two the same and a second negative sample
concatenated = torch.cat((example_batch[0], example_batch[1], example_batch[2], example_batch[3]))

imshow(utils.make_grid(concatenated))

# Load the training dataset
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=32)

net = Quadruplet().cuda()
criterion = QuadrupletLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0005)

#summary(net, (4#, 64,128 ))

optimizer = optim.Adam(net.parameters(), lr = 0.0005)
criterion = QuadrupletLoss()

counter = []
loss_history = []
iteration_number= 0

# Iterate throught the epochs
for epoch in range(5):

    epoch_loss = []
    epoch_pos = []
    epoch_neg = []

    # Iterate over batches
    for i, (img0, img1, img2, img3) in enumerate(train_dataloader, 0):

        # Send the images to CUDA
        img0, img1, img2, img3 = img0.cuda(), img1.cuda(), img2.cuda(), img3.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the three images into the network
        output1, output2, output3, output4 = net(img0, img1, img2, img3)

        # Pass the outputs into the loss function
        loss_siamese = criterion(output1, output2, output3, output4)

        # Calculate the backpropagation
        loss_siamese.backward()

        # Optimize
        optimizer.step()

        # Every 10 batches print out the loss
        epoch_loss.append(loss_siamese.item())

        if i % 10 == 0 :
            print("epoch", epoch, "loss:", np.mean(np.array(epoch_loss)[-10:]))
            print('==============================================')
            #print(f"Epoch number {epoch}\n Current loss {loss_siamese.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_siamese.item())

show_plot(counter, loss_history)