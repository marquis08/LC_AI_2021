import cv2
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from statistics import mean

from pytorch_lightning import Callback

def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())



def mIoU(pred_mask, mask, smooth=1e-10, n_classes=15):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas
            intersect = torch.logical_and(true_class, true_label).sum().item()
            union = torch.logical_or(true_class, true_label).sum().item()
            if union > 0:
                iou = intersect / union
                iou_per_class.append(iou)
        return mean(iou_per_class)



# Custom Callback
class ImagePredictionLogger(Callback):
    def __init__(self, val_sample_image, val_sample_mask):
        super().__init__()
        self.val_imgs = val_sample_image 
        self.val_mask = val_sample_mask
        self.category_names = ['Backgroud'] + [str(i) for i in range(1, 8)]

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_mask.to(device=pl_module.device)
        # Get model prediction
        val_preds = pl_module(val_imgs)
        # val_preds = val_preds.sigmoid() >= 0.5
        val_preds = torch.argmax(val_preds,dim=1)
        # val_preds = val_preds.int()[:,0]

        log_list = list()
        for val_img, val_pred, val_label in zip(val_imgs, val_preds, val_labels):
            # Gray image
            # show_log = self.wandb_seg_image(val_img.detach().cpu().numpy(), val_pred.detach().cpu().numpy(), val_label.detach().cpu().numpy())
            # 3ch image
            show_log = self.wandb_seg_image(val_img.detach().cpu().permute(1,2,0).numpy(), val_pred.detach().cpu().numpy(), val_label.detach().cpu().numpy())
            log_list.append(show_log)

        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":log_list
            }, commit=False)

    def wandb_seg_image(self, image, pred_mask, true_mask):

        return wandb.Image(image, masks={
        "prediction" : {"mask_data" : pred_mask, "class_labels" : self.labels()},
        "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels()}})


    def labels(self):
        l = {}
        for i, label in enumerate(self.category_names):
            l[i] = label
        return l


def dice_channel_torch(probability, truth, threshold=0.5):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                if channel_num == 1:
                    channel_dice = dice_single_channel(probability[i], truth[i], threshold)
                else:
                    channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps = 1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice



def dice_channel_numpy(probability, truth, threshold=0.5):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            if channel_num == 1:
                channel_dice = dice_single_channel(probability[i], truth[i], threshold)
            else:
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
            mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel_numpy(probability, truth, threshold, eps = 1E-9):
    p = (probability.flatten() > threshold)
    t = (truth.flatten() > 0.5)
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice



def JI_channel_numpy(probability, truth, threshold=0.5):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            if channel_num == 1:
                channel_dice = JI_single_channel_numpy(probability[i], truth[i], threshold)
            else:
                channel_dice = JI_single_channel_numpy(probability[i, j,:,:], truth[i, j, :, :], threshold)
            mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel


def JI_single_channel_numpy(probability, truth, threshold, eps = 1E-9):
    p = (probability.flatten() > threshold)
    t = (truth.flatten() > 0.5)
    ji = ((p * t).sum() + eps)/ (p.sum() + t.sum() - (p * t).sum() + eps)
    return ji