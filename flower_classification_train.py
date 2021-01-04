# Imports here
import numpy as np
import pandas as pd
import sys
import os, random
import time, json
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models
from collections import OrderedDict
from optparse import OptionParser


import train_core as tc
    
def main():
    
    #Process script parameters
    parser = OptionParser()
    parser.add_option("-d", "--save_dir", dest="save_directory",
                        help="checkpoint save directory")
    
    parser.add_option("-a", "--arch", dest="architecture",
                        help="model's structure")

    parser.add_option("-l", "--learning_rate", dest="learning_rate",
                       help="learning_rate")
    
    parser.add_option("-u", "--hidden_units", dest="hidden_units",
                      help="hidden layers")
    
    parser.add_option("-e", "--epochs", dest="epochs",
                      help="number of epochs")

    parser.add_option("-g", "--gpu", action="store_true", dest="gpu",
                       help="use gpu for training")

    (options, args) = parser.parse_args()
    if options.save_directory != None:
        checkpoint_save_dir = options.save_directory
    else:
        checkpoint_save_dir = "no checkpoint save directory not specified"
    
    if options.architecture != None:
        model_arch = options.architecture
        if model_arch != "vgg16_bn" | model_arch != "resnet18":
            print("Model " + model_arch + " is not supported. Only vgg16_bn or resnet1010 model is supported")
            return
    else:
        model_arch = "vgg16_bn"
        
    if options.learning_rate != None:
        learning_rate = float(options.learning_rate)
    else:
        learning_rate = 0.001
        
    if options.hidden_units != None:
        hidden_units = int(options.hidden_units)
        if (hidden_units <= 0):
            print("Invalid hidden units")
            return
    else:
        hidden_units = 4096
        
    if options.epochs != None:
        epochs = int(options.epochs)
        if (epochs <= 0):
            print("Invalid number of epochs")
            return
    else:
        epochs = 3
        
    if options.gpu != None:
        if options.gpu == True:
            gpu = True
        else:
            gpu = False
    else:
        gpu = False
        
    print("the save directory is :" + checkpoint_save_dir)
    print("the architecture is :" + model_arch)
    print("the learning rate is :" + str(learning_rate))
    print("the hidden units is :" + str(hidden_units))
    print("the epochs is :" + str(epochs))
    if gpu:
        print("GPU is required")
    else:
        print("GPU is not necessary")
        
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train' :  transforms.Compose([transforms.RandomResizedCrop(224),#, scale=(0.8, 1.0)),
                                           transforms.RandomRotation(45),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

    'validation': transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

    'test': transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])     
    }                                       

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
    'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    data_loader = {
    'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=32, shuffle=True),
    'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32,shuffle=True),
    'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32, shuffle=True)}    


    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if model_arch == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
    elif model_arch == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        print("Model " + model_arch + " is not supported")
        return

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088,hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, len(cat_to_name))),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model = tc.model_train_validate(model, criterion, optimizer, data_loader, gpu, epochs)   
    tc.test_accuracy(model, data_loader)
    tc.save_checkpoint(model, optimizer, image_datasets['train_data'],
                       'vgg16_bn_flower_classification.pt', checkpoint_save_dir)      
       
    
if __name__ == "__main__":
    main()
