import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
import psutil
import torch
import cv2

abs_path = os.path.dirname(__file__)
n_jobs = psutil.cpu_count()
ctime = datetime.today().strftime("%m%d_%H%M")

args = {
    "Competition":"LCAI_2021_Final",
    "C_fold":0,
    "device_number":[0],
    "DEBUG":False,
    "SEED":42,
    "n_folds":4,
    "epochs":60, 
    "image_size":[480, 640], # height, width
    "num_classes":8,
    "tr_bs":4,
    "val_bs":16,
    "num_workers":n_jobs,
    "arch":'UnetPlusPlus', # Unet, DeepLabV3, UnetPlusPlus
    "encoder_name":"timm-efficientnet-b0",# resnet18"mobilenet_v2", "xception" "timm-efficientnet-b0"
    "encoder_weights":'imagenet', # 'imagenet', ssl'(Semi-supervised), 'swsl'(Semi-weakly supervised)
    "loss":"CrossEntropyLoss", # DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss, BCELoss, BCE_DICE_Combo, CrossEntropyLoss
    "optimizer":"AdamW", # Adam, RAdam, AdamW, SGD, Lookahead, AdamP
    "scheduler":"ReduceLROnPlateau", # Cosine, Steplr, Lambda, Plateau, WarmupV2, CosWarm
    "combine_schedulers":["Plateau","Cosine"],
    "patience": 3,
    'es_patience': 20, # earlystop patience
    "lr":1e-4,
    "T_max":9,
    "eta_min":1e-6,
    "plateau_factor":0.9,
    "weight_decay":1e-4,
    "augment_ratio":0.5,
    "amp":False,
}

args['test_path'] = '../data/test_set_for_LCAI' #TODO change this to test
# args['test_df'] = '../data/valid.csv' #TODO change this to test
args['weights'] = [
  '../model/9999_9999_timm-efficientnet-b0_UnetPlusPlus/epoch=50-valid_sst=0.9702_1.ckpt',
  '../model/9999_9999_timm-efficientnet-b0_UnetPlusPlus/epoch=53-valid_sst=0.9693_0.ckpt',
  '../model/9999_9999_timm-efficientnet-b0_UnetPlusPlus/epoch=52-valid_sst=0.9717_2.ckpt',
  '../model/9999_9999_timm-efficientnet-b0_UnetPlusPlus/epoch=50-valid_sst=0.9714_3.ckpt',

]



args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args['tr_path'] = '../processed_640_480/train_set'
args['tr_df'] = '../processed_640_480/train.csv'
args['val_path'] = args['tr_path']
# args['val_path'] = '../data/validation'


# args['val_df'] = '../data/valid_dehaze.csv'
args['weight_path'] = './model' # for inference?
args['ckpt_path'] = f'../model/{ctime}_{args["encoder_name"]}_{args["arch"]}'

if args['DEBUG']:
    args['image_size'] = (64,128)
    args['epochs'] = 10

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

args['tr_aug'] = A.Compose([
                    # A.CenterCrop(422, 422, p=1), # 422 is min size
                    A.Resize(*args['image_size'], interpolation=cv2.INTER_LANCZOS4), 
                    # A.RandomBrightness(limit=0.1, p=0.5),
                    # A.RandomContrast(limit=[0.9, 1.1], p=0.5),
                    # A.Cutout(max_h_size=int(512* 0.375), max_w_size=int(512* 0.375), num_holes=1, p=0.8),
                    A.CLAHE(always_apply=True, p=1),
                    # A.HorizontalFlip(p=0.5),
                    # A.ShiftScaleRotate(p=0.5),
                    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                    # A.VerticalFlip(p=0.5),
                    # A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                    ])

args['val_aug'] = A.Compose([
                    A.Resize(*args['image_size']),
                    # A.CLAHE(always_apply=True, p=1),
                    ToTensorV2(),
                  ])

args['test_aug'] = A.Compose([
                    A.Resize(*args['image_size'], interpolation=cv2.INTER_LANCZOS4), 
                    # A.HorizontalFlip(),
                    # A.GridDistortion(p=0.2)
                    ToTensorV2(),
                  ])
