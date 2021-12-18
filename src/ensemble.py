from conf import *
from inferece import inference, metric_dice_ji
from unet_model import SegModel
from dataset import LCAITestDataset
from trans import get_transforms
from post_pro import batch_crf

import pandas as pd
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Optional
from create_df import create_test_df
from PIL import ImageColor
import cv2
import os

class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        scale_list = [0.25, 0.5], # 0.125, 
        backbone: Optional[nn.Module] = None,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        if backbone is None:
            backbone = SegModel()
        self.backbone = backbone        

    def forward(self, batch):
        output = self.backbone.model(batch) # 
        # HR OCR network 
        # output = F.interpolate(input=output[0], size=(512, 512), mode='bilinear', align_corners=True)
        return output

    

if __name__ == '__main__':
    # model = LitClassifier(backbone=SegModel(encoder_name='tu-hrnet_w18'))
    # print(model.backbone)

    # Load Data
    if args.DEBUG:
        test = create_test_df(args.test_path)[:64]
        # test = pd.read_csv('../data/test.csv')[:64]
    else:
        test = create_test_df(args.test_path)
    print(test.head())
    test_dataset = LCAITestDataset(test, 
                                    base_path=args.test_path, 
                                    transform=args.test_aug)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

   
    pred_list = list()

    for path in args.weights:
    
        # path = '../model/1214_0652_timm-efficientnet-b3_UnetPlusPlus/epoch=72-valid_sst=0.9711_0.ckpt'
        encoder_name = path.split("/")[2].split("_")[2]
        arch_name = path.split("/")[2].split("_")[-1]
        print("Inference ... ", encoder_name, arch_name)

        model = LitClassifier()
        model = model.load_from_checkpoint(path, backbone=SegModel(encoder_name=encoder_name, arch_name=arch_name)).cuda().eval()

        pred = inference(test_dataloader, model)
        pred_list.append(pred)


    pred = np.array(pred_list)
    pred = pred.mean(axis=0)

    pred_argm = np.argmax(pred, axis=1)
    
    # clr = ImageColor.getcolor("#FF00FF","RGB") # converters from CSS3-style color specifiers to RGB tuples
    color_map = {
            1: [64, 255, 0],
            2: [0, 255, 255],
            3: [0, 128, 255],
            4: [64, 0, 255],
            5: [255, 255, 0],
            6: [255, 0, 0],
            7: [255, 0, 255],
        }
    

    test_path = test_dataset.df.img_path.values
    img_list = test_dataset.imgs

    # rescale to original and save
    output_lst = []
    for i, (src_path, img, out) in tqdm(enumerate(zip(test_path, img_list, pred_argm))):
        mask = cv2.resize(out, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST) #INTER_NEAREST
        mask = np.stack((mask,)*3, axis=-1) # to 3ch
        for l in color_map:
            mask[(mask==l).all(-1)] = color_map[l] # apply assigned color
        

        new_file_name = src_path.replace('test_set_for_LCAI','Final') # '../data/test_set_for_LCAI/0028.PNG' -> ../data/submission/0001.png
        new_dir = os.path.split(new_file_name)[0] # ../data/submission/
        os.makedirs(new_dir, exist_ok=True)
        cv2.imwrite(new_file_name, mask)