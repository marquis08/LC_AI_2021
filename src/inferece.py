import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import cv2
import numpy as np
from tqdm import tqdm

# from main import LitClassifier
from dataset import LCAIDataset
from trans import get_transforms
from utils import dice_single_channel_numpy, JI_single_channel_numpy, dice_channel_numpy
from unet_model import SegModel
from post_pro import crf

import pytorch_lightning as pl
from typing import Optional

# def load_model_pl(path):
#     model = LitClassifier()
#     return model.load_from_checkpoint(path)

class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(
        self,
        args,
        scale_list = [0.25, 0.5], # 0.125, 
        backbone: Optional[nn.Module] = None,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.args = args
        if backbone is None:
            backbone = SegModel()
        self.backbone = backbone        

    def forward(self, batch):
        output = self.backbone.model(batch) # 
        # HR OCR network 
        # output = F.interpolate(input=output[0], size=(512, 512), mode='bilinear', align_corners=True)
        return output

def inference(dataloader, model):
    output_list = list()
    with torch.no_grad():
        for img in dataloader:

            output = model(img.cuda()).cpu().detach().sigmoid().numpy()
            output_list.append(output)
    return np.concatenate(output_list)

def metric_dice_ji(pred, gt_list, img_list, th):
    th = 0.5
    dice_score_list = list()
    ji_score_list = list()
    for mask, img, out in zip(gt_list, img_list, pred):
        out = cv2.resize(out[0], mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
        out = crf(img, out)
        dice_score = dice_single_channel_numpy(out, mask, th)
        ji = JI_single_channel_numpy(out, mask, th)
        dice_score_list.append(dice_score)
        ji_score_list.append(ji)
    mean_dice_score = sum(dice_score_list) / len(dice_score_list)
    mean_ji_score = sum(ji_score_list) / len(ji_score_list)
    final = (mean_dice_score + mean_ji_score)/2
    print('validation dice score : ', mean_dice_score)
    print('validation ji score : ', mean_ji_score)
    print('mean : ',final)
    # return mean_dice_score, mean_ji_score

def load_model(path, backbone):
    # Load model
    model = LitClassifier(backbone=backbone)
    model = model.load_from_checkpoint(path)
    model = model.to(device='cuda:0')
    model = model.eval()
    return model

if __name__ == '__main__':
    # Load Data
    valid = pd.read_csv('../data/valid.csv')#[:64]
    valid_dataset = LCAIDataset(valid, base_path='../data/validation', transform=get_transforms(data='valid'))
    valid_dataload = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    # read mask
    mask_list = list()
    for i in tqdm(range(len(valid))):
        mask = valid_dataset.read_mask(valid, i, valid_dataset.base_path)
        mask_list.append(mask)

    # Load model
    path = './model/epoch=49-val_dice_score=0.9503_.ckpt'
    encoder_name = 'timm-efficientnet-b0' # tu-hrnet_w18
    backbone = SegModel(encoder_name=encoder_name)
    model = load_model(path, backbone)
    # model = LitClassifier()
    # model = model.load_from_checkpoint('./model/epoch=49-val_dice_score=0.9503_.ckpt')

    # model = model.to(device='cuda:0')
    # model = model.eval()
        
    pred = inference(valid_dataload, model)
    
    # output_list = list()
    # with torch.no_grad():
    #     for img in valid_dataload:

    #         output = model(img.cuda()).cpu().detach().sigmoid().numpy()
    #         output_list.append(output)
    
    # output = np.concatenate(output_list)#[:, 0] # it is proba pred

    # th = 0.5
    # dice_score_list = list()
    # ji_score_list = list()
    # for mask, out in zip(mask_list, pred):
    #     out = cv2.resize(out[0], mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
    #     dice_score = dice_single_channel_numpy(out, mask, th)
    #     ji = JI_single_channel_numpy(out, mask, th)
    #     dice_score_list.append(dice_score)
    #     ji_score_list.append(ji)
    # print('validation dice score : ', sum(dice_score_list) / len(dice_score_list))
    # print('validation ji score : ', sum(ji_score_list) / len(ji_score_list))
    metric_dice_ji(pred, mask_list, 0.5)