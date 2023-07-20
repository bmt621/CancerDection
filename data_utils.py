# SKlearn
from sklearn import model_selection
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data Augmentation for Image Preprocessing
from albumentations import (VerticalFlip, HorizontalFlip, Compose, Resize,RandomBrightnessContrast,
                            HueSaturationValue,RandomResizedCrop, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50, vgg16,  vgg19, inception_v3, densenet161
import warnings
warnings.filterwarnings("ignore")
import albumentations as A

import pandas as pd
import os
import numpy as np
import cv2

from tqdm.auto import tqdm

def read_and_clean_dataset():
    my_train = df_train_clean.copy()
    concat_df = df_train_concat.copy()
    concat_df['patient_id'] = concat_df['patient_id'].fillna(0)

    #Drop unwanted columns
    drops = ['path_jpeg','path_dicom','diagnosis']
    for drop in drops:
        if drop in my_train.columns:
            my_train.drop([drop], axis =1, inplace=True)

    #Encode categorical datas
    encode = ['sex','anatom_site_general_challenge']

    encodes = []

    concat_df[encode[0]] = concat_df[encode[0]].astype(str)
    concat_df[encode[1]] = concat_df[encode[1]].astype(str)

    label_encoder = LabelEncoder()

    for column in encode:
        concat_df[column] = label_encoder.fit_transform(concat_df[column])


    concat_df.columns = my_train.columns

    #concatenate concat_df information which not available in my_train data

    get_images = my_train['dcm_name'].unique()
    get_unique_concat_data = concat_df[~concat_df['dcm_name'].isin(get_images)]

    #merge new data to my_train data

    new_df = pd.concat([my_train,get_unique_concat_data],axis=0)

    #create path columns to image folder /content/train/train
    path_train = "images/train/train/"
    new_df['path_jpg'] = path_train + new_df['dcm_name'] + '.jpg'

    #fill  age nan values with mean of age
    mean_age = new_df['age'].mean()

    new_df['age'] = new_df['age'].fillna(value=mean_age)

    #Normalize continues data
    age_normalize = preprocessing.scale(new_df['age'])
    anatomy_normalize=preprocessing.scale(new_df['anatomy'])

    new_df['age'] = age_normalize
    new_df['anatomy'] = anatomy_normalize

    #split train and test data

    train_df,test_df = model_selection.train_test_split(new_df,train_size=0.8,test_size=0.2,random_state=8)

    return train_df.iloc[range(len(train_df))].reset_index(drop=True),test_df.iloc[range(len(test_df))].reset_index(drop=True)


class DataSet(Dataset):

    def __init__(self,dataframe,vertical_flip,horizontal_flip,is_train=True):

        self.dataframe = dataframe
        self.is_train = is_train
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip

        if is_train:
            #you can try different transform values and see if there will be improvement

            self.transform = Compose([RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
                                      ShiftScaleRotate(rotate_limit=90, scale_limit = [0.7, 1]),
                                      HorizontalFlip(p = self.horizontal_flip),
                                      VerticalFlip(p = self.vertical_flip),
                                      HueSaturationValue(sat_shift_limit=[0.7, 1.3],
                                                         hue_shift_limit=[-0.1, 0.1]),
                                      RandomBrightnessContrast(brightness_limit=[0.01, 0.1],
                                                               contrast_limit= [0.01, 0.1]),
                                      A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.255],
                                                  max_pixel_value=255.0),
                                      ToTensorV2()])
        else:
            self.transform = Compose([Resize(height=224,width=224),
                                    A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.255],
                                                max_pixel_value=255.0),
                                     ToTensorV2()])


    def __len__(self):
        return(len(self.dataframe)-1)

    def __getitem__(self,index):
        image_path = self.dataframe['path_jpg'][index]

        image = cv2.imread(image_path)

        #import csv information for the specifice index
        csv_data = np.array(self.dataframe.iloc[index][['sex', 'age', 'anatomy']].values,dtype=np.float32)

        #apply the augmentation transform
        image = self.transform(image=image)['image']

        return ((image,csv_data), self.dataframe['target'][index])


