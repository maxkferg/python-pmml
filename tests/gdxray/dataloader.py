import os
import PIL
import time
import math
import glob
import copy
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from torch.optim import lr_scheduler
from torchvision.models.resnet import Bottleneck
from torchvision import datasets, models, transforms
from collections import namedtuple
from pprint import pprint



# Transform module
transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(), # Hor, Rot >> Make conversion very slow
                    transforms.RandomVerticalFlip(),   # It seems like adding dataset would be more efficient
                    transforms.RandomApply([
                        # transforms.RandomRotation([0,90]),
                        # transforms.RandomAffine(0,translate=(0.1,0.1)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                        transforms.RandomGrayscale(),
                        # transforms.RandomCrop(224,padding=10,pad_if_needed=True),
                        # transforms.RandomResizedCrop(224,scale=(0.95, 1.0),ratio=(0.95,1.05)),
                    ], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_val = transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])




class GDXrayDataset(torch.utils.data.Dataset):

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
            self.defects = self.preload_files(data_path)
            self.filenames = list(self.defects.keys())

            # Enforce the filter
            if filter is not None:
                self.filenames = [self.filenames[i] for i in filter]
                self.defects = {k:v for k,v in self.defects.items() if k in self.filenames}


        def __getitem__(self, index):
            """Return the x,y pair at the index"""
            if self.is_train:
                transform = transform_train
            else:
                transform = transform_val
            print(self.filenames)
            filename = self.filenames[index]

            # Sample a new image
            while True:
                image = Image.open(filename).convert('RGB')
                defects = self.defects[filename]
                x, y = self.sample_image(image, defects, size=224)
                if y==0 and self.num_clear > 3*self.num_defect:
                    filename = random.choice(self.filenames)
                else:
                    break

            # Register the type of label that was seen
            if y==1:
                self.num_defect += 1
            else:
                self.num_clear +=1

            x = transform(x)
            if self.debug:
                label = "defect" if y==1 else "clean"
                filename = "debug/{0}-{1}".format(label, os.path.basename(filename))
                torchvision.utils.save_image(x, filename)
            return x, y


        def preload_files(self, data_path):
            """Preload the labels and image filenames
            Return a dict in the form {filename: [Box()...Box()...]}"""
            defects = {}
            print("Loading data from:", data_path)
            if not(os.path.exists(data_path)):
                raise ValueError("Could not find directory:",data_path)
            for root, dirs, files in os.walk(data_path):
                for folder in dirs:
                    metadata_file = os.path.join(data_path, folder,"ground_truth.txt")
                    if os.path.exists(metadata_file):
                        images = glob.glob(os.path.join(data_path, folder, "*.png"))
                        for image in images:
                            defects.setdefault(image, [])
                        for row in np.loadtxt(metadata_file):
                            row_id = int(row[0])
                            image_name = "{folder}_{id:04d}.png".format(folder=folder, id=row_id)
                            image_path = os.path.join(data_path, folder, image_name)
                            box = Box(row[1],row[3],row[2],row[4]) # (x1, y1, x2, y2)
                            defects.setdefault(image_path,[])
                            defects[image_path].append(box)
            print("Found %i matching images"%len(defects))
            return defects


        def sample_image(self, image, defects, size=224):
            """Crop a random 224x224 block from an image
            Images which contain at least one full defect will be labelled 1
            Images which do not contain any defect will be labelled zero"""
            x = None
            y = None
            while x is None:
                w,h = image.size
                if self.debug:
                    self.draw_defects(image, defects)
                if h<size or w<size:
                    pad_y = 1 + int(np.maximum(size-h, 0)/2)
                    pad_x = 1 + int(np.maximum(size-w, 0)/2)
                    padding = (pad_x, pad_y, pad_x, pad_y)
                    image = ImageOps.expand(image, padding)
                    w,h = image.size
                x1 = np.random.randint(0, high=(w-size))
                y1 = np.random.randint(0, high=(h-size))
                box = Box(x1, y1, x1+size, y1+size)

                is_clean = True
                is_defective = False
                for defect in defects:
                    is_clean = (is_clean and not is_intersect(box, defect))
                    is_defective = (is_defective or is_inside(box, defect))

                if is_defective:
                    y = 1
                    x = image.crop(((box.x1, box.y1, box.x2, box.y2)))
                elif is_clean:
                    y = 0
                    x = image.crop(((box.x1, box.y1, box.x2, box.y2)))

            return x, y


        def draw_defects(self, image, defects):
            draw = ImageDraw.Draw(image)
            for defect in defects:
                draw.rectangle((defect.x1, defect.y1, defect.x2, defect.y2), outline=(255, 0, 0))


        def __len__(self):
            """Return the length of the dataset"""
            return len(self.filenames)*self.image_split
