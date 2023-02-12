import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F

from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from model_functions import *

# define command line arguments
def get_pred_arguments():
    parser = argparse.ArgumentParser(description = 'Loading and processing image data for training and prediction')
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--top_k', type=int, help='top k most likely classes', default = 1)
    parser.add_argument('--category_names', type=str, help='mapping of categories to real names', default = 'cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='use gpu for inference')
    
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    im.thumbnail((256, 256))
    
    width, height = im.size
    
    left = (width - 224)/ 2
    top = (height - 224)/ 2
    right = (width + 224)/ 2
    bottom = (height + 224)/ 2
    
    im = im.crop((left, top, right, bottom))
    
    # convert image into a np array
    np_im = np.array(im)
    
    # Divide each channel by 255 to get floats in the range 0-1
    np_im = np_im / 255
    
    # Subtract the means
    means = [0.485, 0.456, 0.406]
    np_im = np_im - means
    
    # Divide by the standard deviations
    stds = [0.229, 0.224, 0.225]
    np_im = np_im / stds
    
    np_im = np_im.transpose(2, 1, 0)
    
    # Convert the array to a PyTorch tensor
    img_tensor = torch.from_numpy(np_im)
    
    return img_tensor

def load_checkpoint(path):
    checkpoint = torch.load(path)
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(image_path, model, topk, args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_checkpoint(model)
    # define the device
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    model.eval()
    
    input_image = process_image(image_path)
    input_image = input_image.unsqueeze_(0)
    input_image = input_image.float() 
    
    input_image = input_image.to(device)
    
    top_class = []
    top_prob = []
    
    with torch.no_grad():

        logps = model.forward(input_image)
        ps = torch.exp(logps)
        top_p, top_c = ps.topk(topk, dim = 1)
        
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        idx_to_class
        
        for i in range(topk):
            x = top_c[0][i].item()
            y = top_p[0][i].item()
            cla = idx_to_class[x]
            top_class.append(cla)
            top_prob.append(y)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    classes = []

    for i in top_class:  
        y = cat_to_name[i]
        classes.append(y)
    
    return top_prob, top_class, classes