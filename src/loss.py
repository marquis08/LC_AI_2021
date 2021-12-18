import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def fetch_loss(args):
    if args.loss == "TverskyLoss":
        return TverskyLoss()
    elif args.loss == "DiceLoss":
        return DiceLoss()
    elif args.loss == "DiceBCELoss":
        return DiceBCELoss()
    elif args.loss == "IoULoss":
        return IoULoss()
    elif args.loss == "FocalLoss":
        return FocalLoss()
    elif args.loss == "BCELoss":
        return SegBCE()
    elif args.loss == "LogCosh":
        return LogCoshDiceLoss()
    elif args.loss == "BCE_DICE_Combo":
        return BCE_DICE_Combo()
    elif args.loss == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()

def mask_onehot(mask, num_class):
    mask = F.one_hot(mask, num_classes=8)
    mask = mask.permute(0, 3, 1, 2)
    return mask

class BCE_DICE_Combo(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCE_DICE_Combo, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        intersection = (inputs * targets).sum()                            
        dice = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return BCE + dice


# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
class LogCoshDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogCoshDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        x = 1 - dice
        
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)


class SegBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SegBCE, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        return F.binary_cross_entropy(inputs, targets, reduction='mean')

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


#PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


#PyTorch

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


#PyTorch

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        if len(inputs.shape) != len(targets.shape):
            targets = mask_onehot(targets, num_class=8)
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


class ComboLoss(nn.Module):
    def __init__(self, loss_1, loss_2):
        super(ComboLoss, self).__init__()
        self.loss_1 = loss_1
        self.loss_2 = loss_2
    
    def forward(self, inputs, targets):
        loss_1 = self.loss_1(inputs, targets)
        loss_2 = self.loss_2(inputs, targets)

        return (loss_1 + loss_2) / 2

#PyTorch
# ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
# CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

# class ComboLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(ComboLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=BETA, eps=1e-9):
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         #True Positives, False Positives & False Negatives
#         intersection = (inputs * targets).sum()    
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
#         inputs = torch.clamp(inputs, eps, 1.0 - eps)       
#         out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
#         weighted_ce = out.mean(-1)
#         combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
#         return combo