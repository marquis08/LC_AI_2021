import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2

import argparse
import sys
from PIL import ImageColor,Image

from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def create_df(data_path):
    df = pd.DataFrame()
    img_list = sorted(glob(os.path.join(data_path,'*','*.png')))
    mask_list = sorted(glob(os.path.join(data_path,'*','*.npy')))
    print(len(img_list), len(mask_list))
    assert len(img_list) == len(mask_list) , 'length of image and mask do not equal'
    
    df['img_path'] = img_list
    df['mask_path'] = mask_list
    df['id'] = df.img_path.apply(lambda x : os.path.split(x)[-1].split('.')[0])
    df['type'] = df.img_path.apply(lambda x : x.split('/')[-2])

    return df


def main(args):
    root_dir = '../data'
    img_width, img_height = tuple(args.image_size) #384, 256
    destination_dir = f'../processed_{img_width}_{img_height}'
    # if not os.path.exists(destination_dir):
    #     os.makedirs(destination_dir)

    label = {
        "#40ff00": 1,
        "#00FFFF": 2,
        "#0080FF": 3,
        "#4000ff": 4,
        "#FFFF00": 5,
        "#FF0000": 6,
        "#FF00FF": 7,
    }

    train_files = [x for x in sorted(glob(os.path.join(root_dir,'train_set','*','*'))) if not x.endswith('.xml')]
    train_files[-1], len(train_files)
    if args.t:
        train_files = train_files[:10]
    

    for file in tqdm(train_files):
        src_dir = os.path.split(file)[0]
        img_id = os.path.split(file)[-1].split(".")[0]
        extension = os.path.split(file)[-1].split(".")[-1]
        file_path = os.path.join(src_dir, img_id)
        category = src_dir.split('/')[-1]

        os.makedirs(destination_dir + f"/train_set/{category}", exist_ok=True)
        tree = ET.parse(f"{file_path}.xml")
        root = tree.getroot()
        root_size = root.findall("size")
        img_real = cv2.cvtColor(cv2.imread(f"{file_path}.{extension}"), cv2.COLOR_BGR2RGB)
        height, width = img_real.shape[:2]
        # depth = int(root_size[0].findtext("depth")) #TODO depth = channel?
        mask = np.zeros((height, width), dtype=np.uint8)
        shapes=root.findall("object") # annotation
        if shapes == []:
            continue # if emtpy labels exist
        for shape in shapes:
            clr=shape.findtext("clr") # find mask color
            if len(clr) == 9:
                clr = clr[0] + clr[3:] # might related color index 
                print(clr)
            points=shape.findall("points")
            data_x=points[0].findall("x")
            data_y=points[0].findall("y")
            
            r=[] # store masks

            for point_x, point_y in zip(data_x,data_y):
                r.append((int(float(point_x.text)),int(float(point_y.text))))
            
            # clr = ImageColor.getcolor(clr,"RGB") # converters from CSS3-style color specifiers to RGB tuples
            if clr in label.keys():
                new_clr = label[clr]
            else:
                print(clr)
                continue
            cv2.fillPoly(mask,[np.asarray(r)],new_clr,cv2.LINE_AA)
        
        # resize here
        small_mask     = cv2.resize(mask,(img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
        if small_mask.max() >7:
            print(file_path)
            continue
        small_mask     = np.save(file_path.replace('data',f'processed_{img_width}_{img_height}'), small_mask)    

        small_img_real = cv2.resize(np.asarray(img_real),(img_width,img_height), interpolation=cv2.INTER_LANCZOS4)
        small_img_real = Image.fromarray(small_img_real).save(file_path.replace('data',f'processed_{img_width}_{img_height}')+'.png')
        
        # print(file_path.replace('data',destination_dir)+'.png')
    
    train = create_df(f'../processed_{img_width}_{img_height}/train_set')
    print("train df: {}".format(train.shape))
    print(train.head())
    print(train.tail())
    train.to_csv(f'../processed_{img_width}_{img_height}/train.csv', index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', nargs='+', type=int, help='width, height') # 640 480
    parser.add_argument('-t', action='store_true', help='debug mode') # if use this, set debug
    args = parser.parse_args()
    main(args)
