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
        else:
            summary(self.cnn1_resnet.cuda(), (3, 224, 224))
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1_resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2, input3, input4):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        output4 = self.forward_once(input4)

        return output1, output2, output3, output4