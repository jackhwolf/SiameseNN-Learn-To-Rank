import torch

# https://discuss.pytorch.org/t/is-there-standard-hinge-loss-in-pytorch/5590/5
class hingeloss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss