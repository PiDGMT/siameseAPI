import torch.nn
from torchvision import models
from torchsummary import summary

class Quadruplet(torch.nn.Module):

    def __init__(self, cpu = False):
        super(Quadruplet, self).__init__()
        self.cpu = cpu

        resnet18 = models.resnet18(pretrained=True)
        modules=list(resnet18.children())[:-1]

        self.cnn1_resnet = torch.nn.Sequential(*modules)

        if self.cpu:
            summary(self.cnn1_resnet, (3, 224, 224))
            #print("cpu")
        else:
            summary(self.cnn1_resnet.cuda(), (3, 224, 224))

        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3,stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3,stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3,stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3,stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2),

            #torch.nn.Flatten()
        )

        if self.cpu:
            summary(self.cnn1, (3, 48, 48))
        else:
            summary(self.cnn1.cuda(), (3,48,48))

        # Setting up the Fully Connected Layers
        self.fc1 = torch.nn.Sequential(

            #torch.nn.Linear(512, 1024), #resnet
            torch.nn.Linear(512, 1024),
            #torch.nn.Linear(384, 1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(256,2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2, input3 = None, input4 = None):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if input3:
            output3 = self.forward_once(input3)
        else:
            output3 = None
        if input4:
            output4 = self.forward_once(input4)
        else:
            output4 = None

        return output1, output2, output3, output4