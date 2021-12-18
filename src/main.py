from conf import *
import pandas as pd
import numpy as np
import os
import shutil

from torch.utils.data import DataLoader
import random
import torch
import torch.nn.functional as F
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping

from sklearn.model_selection import StratifiedKFold
import pdb

# from torch.optim.lr_scheduler import CosineAnnealingLR
from optimizer import fetch_scheduler, fetch_optimizer, fetch_combined_scheduler
from typing import Optional

from build_model import build_model
from dataset import LCAIDataset
# from trans import get_transforms
from utils import ImagePredictionLogger, JI_single_channel_numpy, dice_channel_torch, mIoU, get_jaccard, JI_channel_numpy
from loss import fetch_loss

from sensitivity import Sensitivity
from sklearn.metrics import confusion_matrix

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
            backbone = build_model(args)
        self.backbone = backbone
        self.criterion = fetch_loss(args)
        self.train_sst = Sensitivity(num_classes=args.num_classes, mdmc_average="global")
        self.valid_sst = Sensitivity(num_classes=args.num_classes, mdmc_average="global")

        

    def forward(self, batch):
        output = self.backbone.model(batch) # 
        # HR OCR network 
        # output = F.interpolate(input=output[0], size=(512, 512), mode='bilinear', align_corners=True)
        return output

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        output = self.backbone(x)
        loss = self.criterion(output, y)

        try:
            self.train_sst(output, y)
            self.log("train_sst", self.train_sst, on_step= True, prog_bar=True, logger=True)
            self.log("train_loss", loss, on_step= True, prog_bar=True, logger=True)
        except:
            pass

        return {"loss": loss}#, "predictions": output.detach().cpu(), "labels": y.detach().cpu(), "sst":self.train_sst}

    def training_epoch_end(self, outputs):

        self.log('train_sst_epoch', self.train_sst)  
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.backbone(x)
        
        loss = self.criterion(output, y)

        self.valid_sst(output, y)
        self.log("valid_sst", self.valid_sst, on_step= True, prog_bar=True, logger=True)
        self.log("valid_loss", loss, on_step= True, prog_bar=True, logger=True)

        # x=7 Benign_Tumor -> 2
        # x=6 Cancer       -> 1
        # x<6 Normal       -> 0
        pred_label = []
        gt_label = []
        for out in y.cpu().detach().numpy():
            gt_label.append(np.amax(out))

        for out in output.cpu().detach().sigmoid().numpy():
            pred_label.append(np.argmax(out, axis=0).max())
        
        # change index to [0,1,2] boundary
        pred_label = [2 if x == 7 else 1 if x == 6 else 0 for x in pred_label]
        gt_label = [2 if x == 7 else 1 if x == 6 else 0 for x in gt_label]
        cnf_matrix = confusion_matrix(gt_label, pred_label)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FP_score = sum(FP)/len(gt_label)

        return {"loss": loss, "FP_score":FP_score}

    def validation_epoch_end(self, outputs):

        result = 0
        for o in outputs:
            result += o['FP_score']
        result = result/len(outputs)
        
        self.log('valid_sst_epoch', self.valid_sst) 
        self.log('valid_FP_epoch', result) 

    def configure_optimizers(self):

        param_optimizer = list(self.backbone.named_parameters()) # self.model.named_parameters()
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-6,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = fetch_optimizer(self.args, optimizer_parameters, self.hparams)
        scheduler = fetch_scheduler(self.args, optimizer)
        if self.args.scheduler == "ReduceLROnPlateau":
            return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor="valid_sst") # , lr_scheduler=scheduler_warmup lr_scheduler=scheduler[optimizer], [scheduler]
        else:
            return dict(optimizer=optimizer, lr_scheduler=scheduler)

class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        args,
        train,
        valid,
        batch_size: int = 8,
        
    ):
        super().__init__()
        self.args = args
        self.trn_dataset = LCAIDataset(train, base_path=args.tr_path, 
                                    transform=args.tr_aug)
        self.val_dataset = LCAIDataset(valid, base_path=args.val_path, 
                                    transform=args.val_aug)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.args.tr_bs, 
                            shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.val_bs, 
                            shuffle=False, num_workers=self.args.num_workers)

def cli_main(args):
    logger = WandbLogger(name=f'{args.arch} {args.loss} {args.encoder_name} {args.C_fold}', project=f'{args.Competition}')
    classifier =  LitClassifier(args=args, learning_rate=args.lr)

    if args.DEBUG:
        train = pd.read_csv(args.tr_df)[:32]
    else:
        train = pd.read_csv(args.tr_df)
    
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=args.n_folds, random_state=args.SEED, shuffle=True)
    for i, (trn_idx, val_idx) in enumerate(skf.split(X=train, y=train.type.values)):
        if i == args.C_fold:
            train_df = train.iloc[trn_idx]
            valid_df = train.iloc[val_idx]
            break

    mydatamodule = MyDataModule(args, train_df, valid_df)
    val_img, val_mask = next(iter(mydatamodule.val_dataloader()))
    trainer = pl.Trainer(
        gpus=args.device_number,
        max_epochs=args.epochs,
        # stochastic_weight_avg=True,
        callbacks=[
            ModelCheckpoint(args.ckpt_path, monitor='valid_sst', mode='max', filename='{epoch}-{valid_sst:.4f}_'+str(args.C_fold), save_top_k=2),
            ImagePredictionLogger(val_img, val_mask),
            GPUStatsMonitor(),
            EarlyStopping(monitor="valid_sst", min_delta=1e-7, patience=args.es_patience, verbose=False, mode="max")
        ],
        logger=logger
        )
    trainer.fit(classifier, datamodule=mydatamodule)
    # print(trainer.checkpoint_callbacks.best_model_score)

if __name__ == '__main__':
    os.makedirs(args.ckpt_path, exist_ok=True)
    shutil.copy2(f'./{args.experiment_name}.py',f'{args.ckpt_path}/{args.experiment_name}.py')

    seed_everything()
    cli_main(args)