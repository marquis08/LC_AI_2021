from torch.utils.data import Dataset, DataLoader
from trans import get_transforms
import torch
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm


class LCAIDataset(Dataset):

    def __init__(self, df, base_path, transform):
        super().__init__()

        self.df = df
        self.base_path = base_path
        self.trans = transform

        self.imgs = list()
        self.masks = list()
        for i in tqdm(range(len(self.df))):
            img, mask = self.read_data(self.df, i, self.base_path)
            self.imgs.append(img)
            self.masks.append(mask)
            
        else:
            print('data loaded!!')


        

    def __len__(self):
        return len(self.df)

    def get_path(self, base_path, id_, type_):
        path = os.path.join(base_path, str(type_), str(id_).zfill(4))
        return path + '.npy', path + '.png'

    def read_data(self, df, idx, base_path='../data/train'):
        label_path, img_path = self.get_path(base_path, *df.iloc[idx, -2:])
        # img = cv2.imread(img_path) # , cv2.IMREAD_GRAYSCALE
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.load(label_path)
        # print("IMG shape:", img.shape)
        # print("mask shape:", mask.shape)
        return img, mask # [..., ::-1]

    def get_label_path(self, base_path, id_, type_):
        path = os.path.join(base_path, str(type_), str(id_).zfill(4))
        return path + '.npy'

    def read_mask(self, df, idx, base_path='../data/train'):
        label_path = self.get_label_path(base_path, *df.iloc[idx, -2:])
        mask = np.load(label_path)
        return mask
        
    def __getitem__(self, idx):
        img, mask = self.imgs[idx], self.masks[idx]
        if self.trans is not None:
            transformed = self.trans(image=img, mask=mask)
            images = transformed["image"] # -> C, H, W
            masks = transformed["mask"] # -> H, W
        # print("GET ITEM")
        # print(img.shape, mask.shape)
        return images.float(), masks.long()

class LCAITestDataset(Dataset):
    def __init__(self, df, base_path, transform):
        super().__init__()

        self.df = df
        self.base_path = base_path
        self.trans = transform

        self.imgs = list()

        for i in tqdm(self.df.img_path.values):
            img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
            # img = self.read_data(self.df, i, self.base_path)
            self.imgs.append(img)
        else:
            print('data loaded!!')

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img = self.imgs[idx]
        # img, mask = self.imgs[idx], self.masks[idx]
        if self.trans is not None:
            transformed = self.trans(image=img)
            images = transformed["image"] 
        return images.float() #, torch.from_numpy(mask).float()
    

if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')
    dataset = LCAIDataset(train, base_path='../data/train', transform=get_transforms(data='train'))
    print(dataset[0])
