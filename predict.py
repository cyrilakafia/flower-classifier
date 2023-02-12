# loading and processing images

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
from prediction_functions import *

# get command line arguments
args = get_pred_arguments()
print(args)

# perform predictions
# predict(image_path=args.image_path, model=args.checkpoint, topk=args.top_k, args=args)
probabilities, classes, names = predict(image_path=args.image_path, model=args.checkpoint, topk=args.top_k, args = args)
print('Probabilities:', probabilities)
print('Class(s):', classes)
print('Flower name(s):', names)




