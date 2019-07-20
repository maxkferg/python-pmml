import os
import cv2
import PIL
import time
import math
import glob
import copy
import torch
import keras
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageDraw
from torch.optim import lr_scheduler
from collections import namedtuple
from pprint import pprint


HEIGHT = 384
WIDTH = 384


def get_training_augmentation():
    """Add paddings to make image shape divisible by 32"""
    train_transform = [
        albu.PadIfNeeded(HEIGHT, WIDTH),
        albu.RandomCrop(HEIGHT, WIDTH),
        albu.HorizontalFlip(p=0.5),
        albu.IAAPerspective(p=0.1),
        albu.OneOf([
            albu.RandomBrightness(limit=0.1, p=1),
            albu.RandomGamma(p=1),
            ], p=0.9,
        ),

        albu.OneOf([
            albu.IAASharpen(p=1),
            ], p=0.9,
        ),

        albu.OneOf([
                albu.RandomContrast(limit=0.1, p=1),
            ], p=0.9,
        ),
    ]
    return albu.Compose(train_transform)



def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(224, 224, always_apply=True),
        albu.PadIfNeeded(384, 384)
    ]
    return albu.Compose(test_transform)




class GDXrayDataset(Dataset):

        # Read data from the given path
        def __init__(self, data_path, is_train=True, filter=None, image_split=9, debug=False):
            self.image_split = image_split
            self.is_train = is_train
            self.data_path = data_path
            self.range = range
            self.debug = debug

             # Store the number of each sample we have created
            self.num_clear = 0
            self.num_defect = 0

             # Load the dataset metadata {filename: [Box()...Box()...]}
            self.masks = self.preload_files(data_path)
            self.filenames = list(self.masks.keys())

            # Select the transform
            if self.is_train:
                self.transform = get_training_augmentation()
            else:
                self.transform = get_validation_augmentation()

             # Enforce the filter
            if filter is not None:
                self.filenames = [self.filenames[i] for i in filter]
                self.masks = {k:v for k,v in self.masks.items() if k in self.filenames}


        def __getitem__(self, index):
            """Return the x,y pair at the index"""
            image_name = self.filenames[index]
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self.masks[image_name]
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            print(image.shape,'o')
            print(mask.shape,'x')
            return image, mask


        def preload_files(self, data_path):
            """
            Preload the labels and image filenames
            Return a dict in the form {filename: mask}
            """
            masks = {}
            print("Loading data from:", data_path)
            if not(os.path.exists(data_path)):
                raise ValueError("Could not find directory:",data_path)
            for root, dirs, files in os.walk(data_path):
                for folder in dirs:
                    metadata_file = os.path.join(data_path, folder,"ground_truth.txt")
                    if os.path.exists(metadata_file):
                        images = glob.glob(os.path.join(data_path, folder, "*.png"))
                        for row in np.loadtxt(metadata_file):
                            row_id = int(row[0])
                            image_name = "{folder}_{id:04d}.png".format(folder=folder, id=row_id)
                            image_path = os.path.join(data_path, folder, image_name)
                            if image_path not in masks:
                                masks[image_path] = self.get_mask(image_path)
                                print("Loaded ",image_path)
            print("Found %i matching images"%len(masks))
            return masks


        def get_mask(self, image_path):
            """Return the mask for an image
            The mask is a numpy array with dimensions (h,w,1)
            """
            image_folder = os.path.dirname(image_path)
            image_name = os.path.basename(image_path)
            mask_folder = os.path.join(image_folder, "masks")
            h,w,_ = cv2.imread(image_path).shape
            mask = np.zeros((h,w,1))

            for i in range(10):
                mask_name = image_name.strip(".png") + "_{:01d}.png".format(i)
                mask_path = os.path.join(mask_folder, mask_name)
                if not os.path.exists(mask_path):
                    break
                m = cv2.imread(mask_path)
                m = np.sum(m, axis=-1, keepdims=True)
                mask += m

            mask = np.array(mask>=1, dtype=int)
            return mask


        def draw_defects(self, image, defects):
            draw = ImageDraw.Draw(image)
            for defect in defects:
                draw.rectangle((defect.x1, defect.y1, defect.x2, defect.y2), outline=(255, 0, 0))


        def __len__(self):
            """Return the length of the dataset"""
            return len(self.filenames)




class KerasDataset(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_path, batch_size=32, is_train=True, shuffle=True):
        """Initialization"""
        self.data = GDXrayDataset(data_path)
        self.loader = DataLoader(self.data, shuffle=shuffle, batch_size=batch_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.data)/self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        return next(self.iter)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.iter = iter(self.loader)

