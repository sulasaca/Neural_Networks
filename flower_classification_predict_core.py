
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
#from torch import Tensor
#import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models
from collections import OrderedDict


from workspace_utils import active_session


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.vgg16_bn(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_dix = checkpoint['class_to_idx']
    
    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    load_img = Image.open(image)
                         
    pil_img = transform(load_img).float()
    pil_img = pil_img.unsqueeze(0)                     
    return np.array(pil_img)
                     

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path_param, model_param, k_param=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    print("predict ...")
    model_param.eval()
    pil_img = process_image(image_path_param)
    pil_img = torch.from_numpy(pil_img)
   
    pil_img = pil_img.cuda()
    model_param = model_param.cuda()
        
    output = model_param(pil_img)
    
    output = nn.Softmax()(output)

    probs = torch.topk(output,k_param)[0]
    labels = torch.topk(output,k_param)[1]
    
   
    probs = probs.cpu()
    labels = labels.cpu()
    
    return pil_img, probs.detach().numpy(), labels.detach().numpy()    

    
    

