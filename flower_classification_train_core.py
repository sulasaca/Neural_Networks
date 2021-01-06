# Imports here
import matplotlib.pyplot as plt
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


from workspace_utils import active_session


def model_train_validate(model_param, criterion_param, optimizer_param, data_loader_param, gpu_param, epochs=3, print_every=40):

    print("model_train_validate() ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with active_session():
        if gpu_param:
            
            model_param.to(device)
        else:
            model_param.cpu()
            
        for e in range(epochs):
            train_count = 0
            valid_count = 0

            train_loss = 0
            valid_loss = 0
            train_accuracy = 0 
            valid_accuracy = 0

            total = 0
            correct = 0

            for ii, (inputs, labels) in enumerate(data_loader_param['train_loader']):
                train_count += 1

                model_param.train()
                
                if gpu_param:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                optimizer_param.zero_grad()

                # Forward and backward passes
                outputs = model_param.forward(inputs)
                loss = criterion_param(outputs, labels)
                loss.backward()
                optimizer_param.step()

                train_loss += loss.item()

                if train_count % print_every == 0:
                    print("Train count: {}".format(train_count))
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                                     "Training Loss: {:.4f}".format(train_loss/print_every))
                    train_loss = 0

            for ii, (inputs, labels) in enumerate(data_loader_param['valid_loader']):
                valid_count += 1
                model_param.eval()
                inputs, labels = inputs.to(device),labels.to(device)

                optimizer_param.zero_grad()

                outputs = model_param.forward(inputs)
                loss = criterion_param(outputs, labels)

                valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                #if valid_count % print_every == 0:
                print("Valid count: {}".format(valid_count))
                print("Valid Loss: {:.4f}".format(valid_loss/len(data_loader_param['valid_loader'])))
                valid_loss = 0 
            print('\nValidation accuracy: {:.3f}'.format(correct/total*100))
    return model_param
    
def test_accuracy(model_param, data_loader_param):  
    print("test_accuracy() ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    model_param.eval()
    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(data_loader_param['test_loader']):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model_param(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (100 * correct / total))


def save_checkpoint(model_param, optimizer_param, train_data_param, filename_param, dest_dir_param=None):
    print("save_checkpoint() ...")
    model_param.class_to_idx = train_data_param.class_to_idx

    checkpoint = {'classifier': model_param.classifier,
                   'state_dict': model_param.state_dict(), 
                   'optimizer': optimizer_param,
                   'optimizer_state_dict': optimizer_param.state_dict(),
                   'class_to_idx': model_param.class_to_idx} 
    checkpoint_path = dest_dir_param + "/" + filename_param
    print("save_checkpoint() to file : " + checkpoint_path)
    torch.save(checkpoint, checkpoint_path)


    
    
