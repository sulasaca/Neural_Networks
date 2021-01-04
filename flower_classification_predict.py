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
import argparse


import predict_core as pc

def display_pred_img(image_path_param, model_param, k_param=5):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    image, probabilities, predictions = pc.predict(image_path_param, model_param, k_param)
    
    print("Diplay ...")
    
    idx_to_name = {category: cat_to_name[str(category)] for category in predictions[0]}
    
    class_names = list(idx_to_name.values())
   
    print("probability : " + str(probabilities))
    print("class : " + class_names[0])
   
    
    
def main():
    

    #TBD - script parameter processing
    #Process script parameters
    image_file = sys.argv[1]
    checkpoint_file = sys.argv[2]
    print("Image file : " + image_file)
    print("Checkpoint file : " + checkpoint_file)
    parser = OptionParser()
    parser.add_option("-t", "--top_k", dest="top_k",
                        help="top k most likely classes")
    (options, args) = parser.parse_args()
    if options.top_k != None:
        top_k = int(options.top_k)
    else:
        top_k = 5
    print("top k classes : " + str(top_k))
    
    model, optimizer = pc.load_checkpoint(checkpoint_file)
    display_pred_img(image_file, model)   
 
if __name__ == "__main__":
    main()
