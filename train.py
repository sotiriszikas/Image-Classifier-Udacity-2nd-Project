#Imports
import torch
from torch import nn, optim, tensor
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import OrderedDict
from PIL import Image
from torch.optim import lr_scheduler
import os
import copy
import argparse
import base #imports the functions in base.py in order to use them here.

ap = argparse.ArgumentParser(description='train.py')

ap.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu")
ap.add_argument('--save_dir', dest = "save_dir", action = "store", default = "/home/workspace/ImageClassifier/checkpoint.pth")
ap.add_argument('--learning_rate', dest = "learning_rate", action = "store", default = 0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest = "epochs", action = "store", type = int, default = 6)
ap.add_argument('--arch', dest = "arch", action = "store", default = "vgg19", type = str)
ap.add_argument('--hidden', type = int, dest = "hidden", action = "store", default = 4096)

inputs = ap.parse_args()


path = inputs.save_dir
lr = inputs.learning_rate
net = inputs.arch
p = inputs.dropout
hidden = inputs.hidden
device = inputs.gpu
epochs = inputs.epochs

dataloaders,image_datasets = base.load_data()

model, criterion, optimizer, scheduler = base.build_nn(p, hidden, lr, device, net)

model = base.train_model(model, criterion, optimizer, scheduler, epochs, dataloaders, image_datasets, device = 'cuda')

base.save_checkpoint(image_datasets, optimizer, p, hidden, lr, epochs, path, model)

print("----Done! :) The model has been trained and saved!----\n")
