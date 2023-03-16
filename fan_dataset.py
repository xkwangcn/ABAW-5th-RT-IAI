#!/usr/bin/env python
# Adapted from the code for paper ''.
import random
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_seqlabel(frame, seq, labels):
    beginframe = int(frame - seq / 2)
    seqlabel = torch.zeros(seq)
    for i in range(0, seq):
        label = labels.iloc[beginframe+i-1, 4]
        seqlabel[i] = label
    return seqlabel


def load_image(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image)
    return image


def get_seqimg(path2file, frame, seq, hori_flip, gray, transform):
    beginframe = int(frame - seq/2)
    print(beginframe)
    seqimg = torch.zeros(seq, 3, 224, 224)
    for i in range(0, seq):
        seqimg[i] = load_image(os.path.join(path2file, str(beginframe + i).zfill(5) + '.jpg'),
                               hori_flip, gray, transform=transform)
    seqimg = seqimg.permute(1, 0, 2, 3)
    return seqimg


def get_seqDS(path2csv, frame, seq):
    beginframe = int(frame/15 - seq/2)
    allDS = pd.read_csv(path2csv)
    seqDS = torch.zeros(seq, 1024)
    for i in range(0, seq):
        seqDS[i] = torch.Tensor(allDS.iloc[beginframe-1+i, 2:].tolist())
    return seqDS


class TrainDataset(Dataset):
    def __init__(self, path, Task, Feature, Sequence):
        super(TrainDataset, self).__init__()
        self.balance = 0
        self.task = Task
        self.feature = Feature
        self.sequence = Sequence
        if Task == "AU":
            self.traincsv = os.path.join(path, "5th_ABAW_Annotations/AU_Detection_Challenge/AU_train.csv")
        elif Task == "EXPR":
            self.traincsv = os.path.join(path, "5th_ABAW_Annotations/EXPR_Classification_Challenge/Split0/EXPR_train.csv")
        elif Task == "VA":
            self.traincsv = os.path.join(path, "5th_ABAW_Annotations/VA_Estimation_Challenge/VA_train.csv")
        else:
            print("input correct task")
        self.imagefolder = os.path.join(path, "Faces")
        self.audiofolder = os.path.join(path, "Audio")

        self.allimg = pd.read_csv(self.traincsv)
        # down sample
        self.downsize = 5
        # time augmentation
        self.timeaug = random.randint(0, self.downsize - 1)
        print('timeaug:', self.timeaug)
        self.downsample = self.allimg.loc[(self.allimg['frame'] - 1 - self.timeaug) % self.downsize == 0]
        self.selectimg = np.array(self.downsample.loc[self.downsample['min seq'] >= Sequence])
        np.random.shuffle(self.selectimg)
        print(len(self.selectimg))

    def get_data(self, filename, framenum):
        # path2file = os.path.join(self.imagefolder, filename)
        imgsize = (256, 256)  # W H
        #interpolator_idx = random.randint(0, 1)
        #interpolators = [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
        #interpolator = interpolators[interpolator_idx]
        train_transform = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomResizedCrop(size=imgsize, scale=(0.9, 1), ratio=(1, 1),
                                                                           interpolation=InterpolationMode.BILINEAR),
                                              transforms.ToTensor()])
        img = Image.open(os.path.join(self.imagefolder, filename, str(framenum).zfill(5) + '.jpg'))
        img = train_transform(img)
        # img = load_image(os.path.join(path2file, str(framenum).zfill(5) + '.jpg'), transform=train_transform)
        # seqimgsize = (224, 224)  # W H
        # seqimg = get_seqimg(path2file, framenum, self.sequence, seqimgsize, hori_flip, gray, transform=train_transform)
        return img

    def get_audio(self, filename, framenum):
        DSfeats = get_seqDS(os.path.join(self.audiofolder, 'DeepSpectrum/densenet121-1-0.5-viridis-mel', filename + '.csv'),
                            framenum, int(self.sequence/15))
        return DSfeats

    def get_label(self, ix):
        #imglabel = torch.Tensor([self.selectimg[ix, 2], self.selectimg[ix, 3]])
        imglabel = torch.zeros(8)
        imglabel[self.selectimg[ix, 2]] = 1
        #imglabel = torch.Tensor([self.selectimg[ix, 2], self.selectimg[ix, 3], self.selectimg[ix, 4],
        #                         self.selectimg[ix, 5], self.selectimg[ix, 6], self.selectimg[ix, 7],
        #                         self.selectimg[ix, 8], self.selectimg[ix, 9], self.selectimg[ix, 10],
        #                         self.selectimg[ix, 11], self.selectimg[ix, 12], self.selectimg[ix, 13]])
        return imglabel

    def __getitem__(self, ix):
        filename = self.selectimg[ix, 0]
        framenum = self.selectimg[ix, 1]
        image = self.get_data(filename, framenum) #, imgseq
        #audio = self.get_audio(filename, framenum)
        label = self.get_label(ix)
        return image, label # imgseq, audio,

    def __len__(self):
        sample_pool = len(self.selectimg)
        return sample_pool


class ValDataset(Dataset):
    def __init__(self, path, Task, Feature, Sequence):
        super(ValDataset, self).__init__()
        self.task = Task
        self.feature = Feature
        self.sequence = Sequence
        if Task == "AU":
            self.validcsv = os.path.join(path, "5th_ABAW_Annotations/AU_Detection_Challenge/AU_valid.csv")
        elif Task == "EXPR":
            self.validcsv = os.path.join(path, "5th_ABAW_Annotations/EXPR_Classification_Challenge/Split0/EXPR_valid.csv")
        elif Task == "VA":
            self.validcsv = os.path.join(path, "5th_ABAW_Annotations/VA_Estimation_Challenge/VA_valid.csv")
        else:
            print("input correct task")
        self.imagefolder = os.path.join(path, "Faces")
        self.audiofolder = os.path.join(path, "Audio")

        self.allimg = pd.read_csv(self.validcsv)
        self.downsize = 5
        self.timeaug = random.randint(0, self.downsize - 1)
        print('timeaug:', self.timeaug)
        self.downsample = self.allimg.loc[(self.allimg['frame'] - 1 - self.timeaug) % self.downsize == 0]
        self.selectimg = np.array(self.downsample.loc[self.downsample['min seq'] >= Sequence])
        print(len(self.selectimg))

    def get_data(self, filename, framenum):
        #path2file = os.path.join(self.imagefolder, filename)
        imgsize = (256, 256)  # W H
        #interpolator_idx = random.randint(0, 1)
        #interpolators = [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
        #interpolator = interpolators[interpolator_idx]
        val_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomResizedCrop(size=imgsize, scale=(1, 1), ratio=(1, 1),
                                                                         interpolation=InterpolationMode.BILINEAR),
                                            transforms.ToTensor()])
        img = Image.open(os.path.join(self.imagefolder, filename, str(framenum).zfill(5) + '.jpg'))
        img = val_transform(img)
        # img = load_image(os.path.join(path2file, str(framenum).zfill(5) + '.jpg'), transform=val_transform)
        # seqimgsize = (224, 224)  # W H
        # seqimg = get_seqimg(path2file, framenum, self.sequence, seqimgsize, hori_flip, gray, transform=train_transform)
        return img

    def get_audio(self, filename, framenum):
        DSfeats = get_seqDS(os.path.join(self.audiofolder, 'DeepSpectrum/densenet121-1-0.5-viridis-mel', filename + '.csv'),
                            framenum, int(self.sequence/15))
        return DSfeats

    def get_label(self, ix):
        #imglabel = torch.Tensor([self.selectimg[ix, 2], self.selectimg[ix, 3]])
        imglabel = torch.zeros(8)
        imglabel[self.selectimg[ix, 2]] = 1
        #imglabel = torch.Tensor([self.selectimg[ix, 2], self.selectimg[ix, 3], self.selectimg[ix, 4],
        #                         self.selectimg[ix, 5], self.selectimg[ix, 6], self.selectimg[ix, 7],
        #                         self.selectimg[ix, 8], self.selectimg[ix, 9], self.selectimg[ix, 10],
        #                         self.selectimg[ix, 11], self.selectimg[ix, 12], self.selectimg[ix, 13]])
        return imglabel

    def __getitem__(self, ix):
        filename = self.selectimg[ix, 0]
        framenum = self.selectimg[ix, 1]
        image = self.get_data(filename, framenum)
        # audio = self.get_audio(filename, framenum)
        label = self.get_label(ix)
        return image, label

    def __len__(self):
        sample_pool = len(self.selectimg)
        return sample_pool


class TestDataset(Dataset):
    def __init__(self, path, Task, Feature, Sequence):
        super(TestDataset, self).__init__()
        self.task = Task
        self.feature = Feature
        self.sequence = Sequence
        if Task == "AU":
            self.testcsv = os.path.join(path, "5th_ABAW_Annotations/AU_Detection_Challenge/AU_test.csv")
        elif Task == "EXPR":
            self.testcsv = os.path.join(path, "5th_ABAW_Annotations/EXPR_Classification_Challenge/EXPR_test.csv")
        elif Task == "VA":
            self.testcsv = os.path.join(path, "5th_ABAW_Annotations/VA_Estimation_Challenge/VA_test.csv")
        else:
            print("input correct task")
        self.imagefolder = os.path.join(path, "Faces")
        self.audiofolder = os.path.join(path, "Audio")

        self.allimg = pd.read_csv(self.testcsv)
        self.downsize = 1
        self.timeaug = random.randint(0, self.downsize - 1)
        print('timeaug:', self.timeaug)
        self.downsample = self.allimg.loc[(self.allimg['frame'] - 1 - self.timeaug) % self.downsize == 0]
        self.selectimg = np.array(self.downsample.loc[self.downsample['min seq'] >= Sequence])
        print(len(self.selectimg))

    def get_data(self, filename, framenum):
        #path2file = os.path.join(self.imagefolder, filename)
        imgsize = (224, 224)  # W H
        #interpolator_idx = random.randint(0, 1)
        #interpolators = [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
        #interpolator = interpolators[interpolator_idx]
        test_transform = transforms.Compose([transforms.RandomResizedCrop(size=imgsize, scale=(1, 1), ratio=(1, 1),
                                                                          interpolation=interpolator),
                                            transforms.ToTensor()])
        img = load_image(os.path.join(self.imagefolder, filename, str(framenum).zfill(5) + '.jpg'), transform=test_transform)
        # seqimgsize = (224, 224)  # W H
        # seqimg = get_seqimg(path2file, framenum, self.sequence, seqimgsize, hori_flip, gray, transform=train_transform)
        return img

    def get_audio(self, filename, framenum):
        DSfeats = get_seqDS(os.path.join(self.audiofolder, 'DeepSpectrum/densenet121-1-0.5-viridis-mel', filename + '.csv'),
                            framenum, int(self.sequence/15))
        return DSfeats

    def get_label(self, ix):
        #imglabel = torch.Tensor([self.selectimg[ix, 2], self.selectimg[ix, 3]])
        imglabel = torch.zeros(8)
        imglabel[self.selectimg[ix, 2]] = 1
        #imglabel = torch.Tensor([self.selectimg[ix, 2], self.selectimg[ix, 3], self.selectimg[ix, 4],
        #                         self.selectimg[ix, 5], self.selectimg[ix, 6], self.selectimg[ix, 7],
        #                         self.selectimg[ix, 8], self.selectimg[ix, 9], self.selectimg[ix, 10],
        #                         self.selectimg[ix, 11], self.selectimg[ix, 12], self.selectimg[ix, 13]])
        return imglabel

    def __getitem__(self, ix):
        filename = self.selectimg[ix, 0]
        framenum = self.selectimg[ix, 1]
        image = self.get_data(filename, framenum)
        #audio = self.get_audio(filename, framenum)
        label = self.get_label(ix)
        return image, label

    def __len__(self):
        sample_pool = len(self.selectimg)
        return sample_pool

