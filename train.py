# Imports here
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
from model_functions import *

# get arguments
args = get_arguments()
print(args)

# load the data
train_datasets, trainloader, valloader, testloader = load_data(args)

# define model architecture
model, criterion, optimizer = model_arch(args)

# train the model

print('Training has began ============')
model = train(model, criterion, optimizer, trainloader, testloader, args)
print('Training has ended ======================')
    
# test the model performance
accuracy = test(model, criterion, testloader, args)
print('Test Accuracy: ', accuracy)
    
# save the checkpoint
save_checkpoint(model, args.save_dir, train_datasets, accuracy, args)
    
print('Done =======================================')


