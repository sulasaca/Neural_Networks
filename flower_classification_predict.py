import numpy as np
import pandas as pd
import sys
import os, random
from pathlib import Path
import os.path
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

def display_pred_img(image_path_param, model_param, catfile_param, gpu_param=True, k_param=5):
    with open(catfile_param, 'r') as f:
        cat_to_name = json.load(f)
    
    image, probabilities, predictions = pc.predict(image_path_param, model_param, gpu_param, k_param)
    
    print("Diplay ...")
    
    categories = []
    for idx in predictions[0]:
        categories.append(cat_to_name[str(idx)])        
    print("categories: " + str(categories))
    
    for i in range(k_param):
        print("top " + str(i+1) + " most likely image class and probability:")
        print("\t probability : " + str(probabilities[0][i]))
        print("\t class : \t" + categories[i])
   
    
    
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
    
    parser.add_option("-c", "--cat", dest="cat",
                        help="flower categories json file")
    
    parser.add_option("-g", "--gpu", action="store_true", dest="gpu",
                       help="use gpu for predicting")

    (options, args) = parser.parse_args()
    if options.top_k != None:
        top_k = int(options.top_k)
        if top_k <= 0:
            print("top_k must be greater or equal to 1")
            return
    else:
        top_k = 1
    print("top k classes : " + str(top_k))
    
    
    if options.cat != None:
        cat_file = options.cat
        if not os.path.isfile(cat_file):
            print("Error: category file '" + cat_file + "' does not exist")
            return
    else:
        cat_file = "cat_to_name.json"
    print("Image category mapping file " + cat_file)    
        
        
    if options.gpu != None:
        if options.gpu == True:
            gpu = True
        else:
            gpu = False
    else:
        gpu = False
    print("gpu is enabled " + str(gpu))    
    
    model, optimizer = pc.load_checkpoint(checkpoint_file)
    display_pred_img(image_file, model, cat_file, gpu, top_k)   
 
if __name__ == "__main__":
    main()
