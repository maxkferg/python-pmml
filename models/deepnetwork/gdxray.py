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
from core.utils import is_intersect, is_inside
from PIL import Image, ImageOps, ImageDraw
from torch.optim import lr_scheduler
from torchvision.models.resnet import Bottleneck
from torchvision import datasets, models, transforms
from collections import namedtuple
from pprint import pprint



Box = namedtuple('Point', ['x1', 'y1', 'x2', 'y2'])


#-----------------
# BASIC INFORMATION
#-----------------
# Using pretrained model
use_pretrained_feature_extraction_model = False
use_pretrained_fine_tuning_model = False # Setting it true makes the feature extraction be ignored.

# Dataset dependent variables
range_total = range(1315)
range_train = random.sample(range_total, 1100)
range_val = list(set(range_total) - set(range_train))

# Data path and model filenames
data_path = os.path.expanduser('~/Data/GDXray/Castings/')
model_fe_filename = 'debug/gdxray-model.ckpt'
model_ft_filename = 'debug/gdxray-model-ft.ckpt'

#-----------------
# HYPER PARAMETERS
#-----------------
# Interation
num_epochs_feature_extraction = 0
num_epochs_fine_tuning = 10
batch_size = 32

# Optimizer
optimizer_type = 'SGD'
initial_lr = 0.02
initial_lr_finetuning = 0.005 # 0.0248364446993
weight_decay = 1e-5
momentum = 0.90 #SGD #[0.5, 0.9, 0.95, 0.99]
nesterov = False #SGD

# Learning rate scheduler
scheduler_type = 'StepLR' # ReduceLROnPlateau easily reduces LR even when training error is quite high
step_size = 15 #StepLR
gamma = 0.1  #0.140497184025 #StepLR
factor = 0.1 #ReduceLROnPlateau
patience = 5 #ReduceLROnPlateau
threshold = 5e-2 #ReduceLROnPlateau




class ResSkipNet(torchvision.models.resnet.ResNet):

        def override_layers(self):
                self.pool1 = nn.MaxPool2d(kernel_size=28, stride=14, padding=0) # Pools end of layer1
                self.pool2 = nn.MaxPool2d(kernel_size=14, stride=7, padding=0) # Pools end of layer2
                self.pool3 = nn.MaxPool2d(kernel_size=6, stride=6, padding=0) # Seongwoon
                self.pool4 = nn.AvgPool2d(7, stride=1) # ResNet 152 standard
                self.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(4096, 1),
                )


        def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x1 = self.layer1(x)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)

                f1 = self.pool1(x1)
                f2 = self.pool2(x2)
                f3 = self.pool3(x3)
                f4 = self.pool4(x4)

                f1 = f1.view(f1.size(0),-1)
                f2 = f2.view(f2.size(0),-1)
                f3 = f3.view(f3.size(0),-1)
                f4 = f4.view(f4.size(0),-1)

                x = torch.cat([f3],1)
                x = x.view(x.size(0), -1)
                x = self.fc(x)

                return x


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResSkipNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_urls = torchvision.models.resnet.model_urls
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    model.override_layers()
    return model




# Model
basemodel = resnet152(pretrained=True)


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

transform_val = transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



#-----------------
# HELPER FUNCTIONS
#-----------------
# Custom pytorch data loader
class CustomDataset(torch.utils.data.Dataset):

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
            image_index = int(index / self.image_split)
            filename = self.filenames[image_index]

            while True:
                image = Image.open(filename).convert('RGB')
                defects = self.defects[filename]
                x, y = self.sample_image(image, defects, size=224)
                if self.is_train and y==0 and self.num_clear > 3*self.num_defect:
                    filename = random.choice(self.filenames)
                else:
                    break

            if self.is_train and self.num_defect%100 == 0:
                print("TRAIN: {} clear images and {} defective images".format(self.num_clear, self.num_defect))

            if not self.is_train and self.num_defect%100 == 0:
                print("VAL: {} clear images and {} defective images".format(self.num_clear, self.num_defect))

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
                    self.num_defect += 1
                elif is_clean:
                    y = 0
                    x = image.crop(((box.x1, box.y1, box.x2, box.y2)))
                    self.num_clear +=1

            return x, y


        def draw_defects(self, image, defects):
            draw = ImageDraw.Draw(image)
            for defect in defects:
                draw.rectangle((defect.x1, defect.y1, defect.x2, defect.y2), outline=(255, 0, 0))


        def __len__(self):
            """Return the length of the dataset"""
            return len(self.filenames)*self.image_split



#-----------------
# MODEL TRAINING
#-----------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        # Manage the best model so far
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 10000000

        for epoch in range(num_epochs):
                start = time.time()
                print('Epoch {}/{}'.format(epoch+1, num_epochs))
                print('-' * 10)

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
                ### Training phase
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_corrects = 0
                incorrect = {'00':0, '01':0,'10':0, '11':0}
                # Update learning rate: StepLR
                if type(scheduler) == torch.optim.lr_scheduler.StepLR:
                        scheduler.step()
                        print("Current learning rate: ", scheduler.get_lr())
                for i, (inputs, labels) in enumerate(train_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        with torch.set_grad_enabled(True):
                            outputs = model(inputs)
                            # preds = torch.round(outputs.squeeze()).long()
                            preds = torch.zeros(outputs.squeeze().size(), dtype=torch.long).to(device)
                            for j in range(outputs.squeeze().size()[0]):
                                if outputs[j] > 0:
                                    preds[j] = 1
                            loss = criterion(outputs.squeeze(), labels.float())
                            # backward + optimize only if in training phase
                            loss.backward()
                            optimizer.step()


                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds.data == labels.data)

                        for j in range(len(labels.data)):
                                key = str(int(labels.data[j])) + str(int(preds.data[j]))
                                incorrect[key] += 1

                # Calculate accuracy
                train_loss = running_loss / len(train_dataset)
                train_acc = running_corrects.double() / len(train_dataset)
                print(incorrect)

                ### Validation phase
                model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                running_corrects2 = 0
                running_corrects3 = 0
                incorrect = {'00':0, '01':0,'10':0, '11':0}
                # Iterate over data.
                for i, (inputs, labels) in enumerate(val_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(False):
                            outputs = model(inputs)
                            # preds = torch.round(outputs.squeeze()).long()
                            preds = torch.zeros(outputs.squeeze().size(), dtype=torch.long).to(device)
                            for j in range(outputs.squeeze().size()[0]):
                                if outputs[j] > 0:
                                    preds[j] = 1
                            preds2 = torch.zeros(outputs.squeeze().size(), dtype=torch.long).to(device)
                            for j in range(outputs.squeeze().size()[0]):
                                if outputs[j] > 0.05:
                                    preds2[j] = 1
                            preds3 = torch.zeros(outputs.squeeze().size(), dtype=torch.long).to(device)
                            for j in range(outputs.squeeze().size()[0]):
                                if outputs[j] > -0.05:
                                    preds3[j] = 1
                            loss = criterion(outputs.squeeze(), labels.float())
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds.data == labels.data)
                        running_corrects2 += torch.sum(preds2.data == labels.data)
                        running_corrects3 += torch.sum(preds3.data == labels.data)

                        for j in range(len(labels.data)):
                                key = str(int(labels.data[j])) + str(int(preds.data[j]))
                                incorrect[key] += 1

                # Calculate accuracy
                val_loss = running_loss / len(val_dataset)
                val_acc = running_corrects.double() / len(val_dataset)
                val_acc2 = running_corrects2.double() / len(val_dataset)
                val_acc3 = running_corrects3.double() / len(val_dataset)
                print(incorrect)
                print('{} Loss: {:.4f} Acc: {:.4f} || {} Loss: {:.4f} Acc: {:.4f} ({:.4f},{:.4f})'.format('train', train_loss, train_acc, 'val', val_loss, val_acc, val_acc2, val_acc3))

                # deep copy the model
                if val_loss < best_loss:
                        print("** New best! **")
                        best_loss = val_loss
                        best_acc = val_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(best_model_wts, 't1_temp_model6_best.ckpt')
                else:
                        print("best so far Loss: {:.4f} Acc: {:.4f}".format(best_loss, best_acc))

                # Update learning rate: ReduceLROnPlateau
                if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                        scheduler.step(train_loss)

                end = time.time() - start
                print('elapsed time: {0} seconds'.format(int(end)))
                print()

                torch.save(model.state_dict(), 't1_temp_model6.ckpt')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


#-----------------
# SETUP MODEL AND DATA
#-----------------
# Dataset
train_dataset = CustomDataset(data_path, True, range_train)
val_dataset = CustomDataset(data_path, False, range_val)

# Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=16, batch_size=batch_size, shuffle=False)

# Enable cuda, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model
model_conv = basemodel
model_conv = model_conv.to(device)


#-----------------
# FEATURE EXTRACTION
#-----------------
if use_pretrained_feature_extraction_model:
        model_conv.load_state_dict(torch.load(model_fe_filename))

ct = 0
for child in model_conv.children():
    ct += 1
    if ct < 9:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True

criterion = nn.CrossEntropyLoss()

if optimizer_type == 'SGD':
    optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
elif optimizer_type == 'Adam':
    optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr, weight_decay=weight_decay)

if scheduler_type == 'StepLR':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)
elif scheduler_type == 'ReduceLROnPlateau':
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, patience=patience, threshold=threshold, factor=factor, verbose=True)

print()
print("START FEATURE EXTRACTION")
print("************************")
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs_feature_extraction)
torch.save(model_conv.state_dict(), model_fe_filename)


#-----------------
# FINE TUNING
#-----------------
# Fix the CNN params
# ct = 0
# for child in model_conv.children():
#       ct += 1
#       if ct < 8:
#               for param in child.parameters():
#                       param.requires_grad = False
#       else:
#               for param in child.parameters():
#                       param.requires_grad = True


block = model_conv.layer3
for i in range(len(block)):
    block[i].conv1.weight.requires_grad = True
    block[i].bn1.weight.requires_grad = True
    block[i].bn1.bias.requires_grad = True
    block[i].conv2.weight.requires_grad = True
    block[i].bn2.weight.requires_grad = True
    block[i].bn2.bias.requires_grad = True
    block[i].conv3.weight.requires_grad = True
    block[i].bn3.weight.requires_grad = True
    block[i].bn3.bias.requires_grad = True
    block[i].conv1.weight.requires_grad = True



# # Allow update for a part of layer3 of Resnet
# for i in range(0,36):
#       model_conv.layer3[i].conv1.weight.requires_grad = True
#       model_conv.layer3[i].bn1.weight.requires_grad = True
#       model_conv.layer3[i].bn1.bias.requires_grad = True
#       model_conv.layer3[i].conv2.weight.requires_grad = True
#       model_conv.layer3[i].bn2.weight.requires_grad = True
#       model_conv.layer3[i].bn2.bias.requires_grad = True
#       model_conv.layer3[i].conv3.weight.requires_grad = True
#       model_conv.layer3[i].bn3.weight.requires_grad = True
#       model_conv.layer3[i].bn3.bias.requires_grad = True
#       if i == 0:
#               model_conv.layer3[i].downsample[0].weight.requires_grad = True
#               model_conv.layer3[i].downsample[1].weight.requires_grad = True
#               model_conv.layer3[i].downsample[1].bias.requires_grad = True


# # Initialize layer 4
# for i in range(0,3):
#       torch.nn.init.xavier_uniform_(model_conv.layer4[i].conv1.weight.data)
#       model_conv.layer4[i].bn1.weight.data.uniform_()
#       model_conv.layer4[i].bn1.bias.data.zero_()
#       torch.nn.init.xavier_uniform_(model_conv.layer4[i].conv2.weight.data)
#       model_conv.layer4[i].bn2.weight.data.uniform_()
#       model_conv.layer4[i].bn2.bias.data.zero_()
#       torch.nn.init.xavier_uniform_(model_conv.layer4[i].conv3.weight.data)
#       model_conv.layer4[i].bn3.weight.data.uniform_()
#       model_conv.layer4[i].bn3.bias.data.zero_()
#       if i == 0:
#               torch.nn.init.xavier_uniform_(model_conv.layer4[i].downsample[0].weight.data)
#               model_conv.layer4[i].downsample[1].weight.data.uniform_()
#               model_conv.layer4[i].downsample[1].bias.data.zero_()


# # Allow update for a part of layer4 of Resnet
# for i in range(0,3):
#         model_conv.layer4[i].conv1.weight.requires_grad = True
#         model_conv.layer4[i].bn1.weight.requires_grad = True
#         model_conv.layer4[i].bn1.bias.requires_grad = True
#         model_conv.layer4[i].conv2.weight.requires_grad = True
#         model_conv.layer4[i].bn2.weight.requires_grad = True
#         model_conv.layer4[i].bn2.bias.requires_grad = True
#         model_conv.layer4[i].conv3.weight.requires_grad = True
#         model_conv.layer4[i].bn3.weight.requires_grad = True
#         model_conv.layer4[i].bn3.bias.requires_grad = True
#         if i == 0:
#               model_conv.layer4[i].downsample[0].weight.requires_grad = True
#               model_conv.layer4[i].downsample[1].weight.requires_grad = True
#               model_conv.layer4[i].downsample[1].bias.requires_grad = True


for i in range(0,5):

    criterion = nn.BCEWithLogitsLoss()

    if optimizer_type == 'SGD':
        optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr_finetuning, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif optimizer_type == 'Adam':
        optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=initial_lr_finetuning, weight_decay=weight_decay)


    if scheduler_type == 'StepLR':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'ReduceLROnPlateau':
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, patience=patience, threshold=threshold, verbose=True, factor=factor)

    print()
    print("START FINE TUNING " + str(int(i)))
    print("********************")
    if use_pretrained_fine_tuning_model:
            model_conv.load_state_dict(torch.load(model_ft_filename))
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs_fine_tuning)

    model_ft_filename = 't4_resnet_v2_ft8_{0}.ckpt'.format(int(i))
    torch.save(model_conv.state_dict(), model_ft_filename)
    initial_lr_finetuning = initial_lr_finetuning * (0.1)
    # initial_lr_finetuning = 0.002



