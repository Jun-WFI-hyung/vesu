import torch
import torch.nn as nn


class DiceLoss_BIN(nn.Module):
    def __init__(self, device):
        super(DiceLoss_BIN,self).__init__()
        self.device = device
    
    def forward(self, output:torch.Tensor, label:torch.Tensor):
        bce = nn.BCEWithLogitsLoss().to(self.device)
        bce_l = bce(output, label)

        smooth = 1.
        output = output.requires_grad_(True)
        label = label.requires_grad_(True)

        sigmoid = nn.Sigmoid()
        output = sigmoid(output)

        output = output.view(-1)
        label = label.view(-1)

        inter = (label * output).sum()
        union = output.sum() + label.sum()
        dice_l = (1-((2*inter)+smooth)/(union+smooth))
        IOU = inter / (union - inter)

        return dice_l+bce_l, IOU


# a = torch.Tensor([[0,0,0],[0,0,0],[0,0,1]])
# b = torch.Tensor([[0,0,0],[0,0,0],[0,0,1]])

# print(IOU_bin(a,b))

