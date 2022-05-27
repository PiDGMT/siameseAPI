import torch
import torch.nn.functional as F

class QuadrupletLoss(torch.nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples. The negative examples needs not to be matching the anchor, the positive and each other.
    """
    def __init__(self, margin1 = 1, margin2 = 0.5):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):

        squared_distance_pos = (anchor - positive).pow(2).sum(1)
        squared_distance_neg = (anchor - negative1).pow(2).sum(1)
        squared_distance_neg_b = (negative1 - negative2).pow(2).sum(1)

        quadruplet_loss = \
            F.relu(self.margin1 + squared_distance_pos - squared_distance_neg) \
            + F.relu(self.margin2 + squared_distance_pos - squared_distance_neg_b)

        return quadruplet_loss.mean()