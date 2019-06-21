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

ap = argparse.ArgumentParser(description = 'predict.py')

ap.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu") 
ap.add_argument('--checkpoint', action = "store", default = "/home/workspace/ImageClassifier/checkpoint.pth") 
ap.add_argument('--topk', default = 5, dest = "top_k", action = "store", type = int) 
ap.add_argument('--category_names', dest = "category_names", action = "store", default = 'cat_to_name.json') 
ap.add_argument('image_path', default = '/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg', nargs='*', action = "store", type = str)


inputs = ap.parse_args()

image_path = inputs.image_path
topk = inputs.top_k
device = inputs.gpu
path = inputs.checkpoint

dataloaders,image_datasets = base.load_data()

model = base.load_checkpoint(path)

base.testdata_acc(model, dataloaders, image_datasets, 'test', True)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    
img_tensor = base.process_image(image_path)

probs = base.predict(image_path, model, topk)

print("Image Directory: ",image_path)
print("Predictions probabilities: ", probs)














