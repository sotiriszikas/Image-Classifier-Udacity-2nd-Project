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
import time 
from torch.optim import lr_scheduler
import os
import copy
import argparse

#Inputs for the nn depending on the network we are going to select.
arch = {"vgg19":25088,
        "densenet121":1024,
        "alexnet":9216}

def load_data():
    
    '''
    Arguments : None
    Returns : The dataloaders as a dictionary for train/valid/test sets. (dataloader['train'] for example.
    This function applies to the images the transformations required (crop, normalization etc) and converts the images to tensor in order to be able to be used from the nn.
    '''
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Applying the transformations. 
    data_transforms = {
    'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]),
                                                            
    'valid' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

    'test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    }
    
    
    #Loading the datasets using ImageFolder
    image_datasets = {
    'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }
    
    
    #Defining the dataloaders using the transforms and the datasets above.
    dataloaders = {
    'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle = True),
    'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64, shuffle = True)
    }
    
    return dataloaders, image_datasets



def build_nn(p, hidden, lr, device, net = 'vgg19'):
    
    '''
    Arguments: network name,dropout variable 'p', hidden layer number 'hidden', learning late 'lr' and the device
    Returns: model, criterion, optimizer, scheduler. 
    This function builds the neural network given the input arguments and returns the variables we need to use the train_model() function.
    
    
    '''
    
    if net == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif net == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif net == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
         print("Invalid model.Choose between vgg19/densert121/alexnet.")
    
    
    #Ensure that the weights are not updated.        
    for param in model.parameters():
        param.requires_grad = False
        
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(p)),  
                              ('fc1', nn.Linear(arch[net], hidden)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                            ]))
        
        model.classifier = classifier
        print(model.classifier )
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1) #Every 4 epochs decays LR by gamma.
        
        if torch.cuda.is_available() and device == 'gpu':
            model.cuda()
            
        return model, criterion, optimizer, scheduler 
    
    
    
            
def train_model(model, criterion, optimizer, scheduler, epochs, dataloaders, image_datasets, device = 'cuda'):
    
    '''
    Arguments: model, criterion optimizer abd scheduler from build_nn() function. epochs and devide.
    Returns: the trained model.
    This function takes as arguments the training parameters like epochs for example and returns the trained model.
    
    '''
    
    print("Starting the Training...\n")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training Completed in : {:.0f}m and {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model  




def testdata_acc(model, dataloaders, image_datasets, data , cuda = False):
    '''
    Arguments: model, data='str', cuda = bool
    Returns : None
    This function test the accuracy of the dataloaders['test'] dataset on our trained model.
    '''
    model.eval()
    model.to(device = 'cuda')
    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
                
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            
            if idx == 0:
                print("Predicted Flower Type:\n",predicted)
                print()
                print("Predicted Probability:\n",torch.exp(_)) 
            equals = predicted == labels.data
            if idx == 0:
                print()
                print("Model was correct(1)/Model was wrong(0):\n",equals)
                correct = equals.nonzero().size(0)
                total = len(equals)
                print()
            print(equals.float().mean())
        print("Accuracy percentage on Test Data: {:.2f} %".format((correct/total)*100))
        

         
def save_checkpoint(image_datasets, optimizer, p = 0.5, hidden = 4096, lr = 0.001, epochs = 8, path = 'checkpoint.pth', net = 'vgg19'):
    '''
    Arguments: checkpoint path and hyperparameters.
    Returns: None 
    This function saves the model at a given path.pth 
    '''
    model = net 
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu
    
    torch.save({
    'arch': net,
    'dropout': p,    
    'learning_rate': lr,
    'hidden_layers': hidden,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx
    },path)
            
      
    
    
def load_checkpoint(path = 'checkpoint.pth'):
    '''
    Arguments: Checkpoint path. 
    Returns: The NN with all the hyperparameters, biases and weights.
    '''     
    
    checkpoint = torch.load(path)
    net = checkpoint['arch']
    hidden = checkpoint['hidden_layers']
    p = checkpoint['dropout']
    lr = checkpoint['learning_rate']
    device = 'gpu'

    model,_,_,_ = build_nn(p, hidden, lr, device, net = 'vgg19')

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model     
        
def process_image(image_path):
    '''
    Arguments: Path of the Image
    Returns: The image as a tensor
    
    '''
    image = Image.open(image_path)
    
    adjust = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjust(image)
    
    return img_tensor



def predict(image_path, model, topk = 5):
    '''
    Arguments: Path of the Image, model and number of predictions
    Returns: The 'topk' choices that the model predicts.
    
    '''
    
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)



