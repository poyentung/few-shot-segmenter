import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import sigmoid_focal_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, lmbda:float=0.8, alpha:float=0.25, gamma:float=2.0, smooth:float=1.0, p:float=2.0, reduction='mean'):
        super().__init__()
        self.lmbda= lmbda
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.dice_loss = BinaryDiceLoss(smooth=smooth, p=p, reduction=reduction)
    
    def forward(self, predict, target):
        focal_loss = sigmoid_focal_loss(inputs=predict, 
                                        targets=target, 
                                        alpha=self.alpha, 
                                        gamma=self.gamma, 
                                        reduction=self.reduction)
        dice_loss = self.dice_loss(predict=F.sigmoid(predict), target=target)
        
        return self.lmbda*focal_loss + (1-self.lmbda)*dice_loss
    
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
