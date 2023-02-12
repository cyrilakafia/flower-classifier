# functions and classes relating to the model

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

# define command line arguments
def get_arguments():
    parser = argparse.ArgumentParser(description = 'Train a new network on a dataset with train.py')
    parser.add_argument('data_dir', type = str, help = 'Path to data directory')
    parser.add_argument('--arch', type=str, help='Choose architecture', default = 'vgg19')
    parser.add_argument('--learning_rate', type=float, help='Set hyperparameters', default=0.001)
    parser.add_argument('--epochs', type=int, help='Set hyperparameters', default = 8)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--save_dir', type=str, help='Set directory to save checkpoints', default='models')

    return parser.parse_args()

# load data
def load_data(args):
    """
    Function to load image data
    
    Arguments:
        None
    
    Returns
        train datasets
        trainloader
        valloader
        testloader
    """

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    
    val_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.255])])
    
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    val_datasets = datasets.ImageFolder(valid_dir, transform = val_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = val_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    valloader = torch.utils.data.DataLoader(val_datasets, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
    
    return train_datasets, trainloader, valloader, testloader

def model_arch(args):
    """
    Function to define model architecture for training
    
    Returns:
        model: model architecture to be trained
        criterion: the loss function
        optimizer: the optimization function to update the weights and biases
    """
    
    # Define the model
    if args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    
    for params in model.parameters():
        params.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.classifier[0].in_features, 956)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(956, 356)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.15)),
        ('fc3', nn.Linear(356, 102)),
        ('output', nn.LogSoftmax(dim = 1))]))

    # define the loss
    criterion = nn.NLLLoss()
    
    # define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    
    return model, criterion, optimizer

def train(model, criterion, optimizer, trainloader, valloader, args):
    
    """
    Function to train the network
    
    Args:
        model: the model architecture to be trained
        criterion: the loss function
        optimizer: the optimation function to update the weights and biases
        trainloader: the training dataset
        valloader: the validation dataset
        
    Returns:
        float: training loss, validation loss, validation accuracy
        model: trained model
    
    """
   
    # define the device
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
     
    epochs = args.epochs
    steps = 0
    print_every = 25
    running_loss = 0

    for epoch in range(epochs):
        model.to(device)
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            logps = model.forward(images)
        
            loss = criterion(logps, labels)
        
            loss.backward()
        
            optimizer.step()
        
            running_loss += loss.item()
        
            steps += 1
        
            if steps%print_every == 0:
                model.eval()
                validation_loss = 0
                validation_accuracy = 0
            
                with torch.no_grad():
                    for images, labels in valloader:
                        images, labels = images.to(device), labels.to(device)
                    
                        logps = model.forward(images)
                        val_loss = criterion(logps, labels)
                        validation_loss += val_loss.item() 
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                    
                        equals = top_class == labels.view(*top_class.shape)
                    
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print("Epoch: {}/{}  Training Loss: {}  Validation Loss: {}  Validation Accuracy: {}".format(epoch+1, epochs, running_loss/print_every, validation_loss/len(valloader), validation_accuracy/len(valloader)))
                running_loss = 0
                model.train()
                
    return model

def test(model, criterion, testloader, args):
    """
    Function to test model performance
    
    Arguments:
        model: model object
        criterion: loss function
        testload: test dataset
        
    Prints test loss and accuracy
    """
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    
    loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in testloader:
            model.eval()
            images, labels = images.to(device), labels.to(device)
                    
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            loss += batch_loss.item() 
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
        
            equals = top_class == labels.view(*top_class.shape)
        
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print('Loss: {:.3f}  Accuracy: {:.3f}'.format(loss/len(testloader), accuracy/len(testloader)))
    return '{:.3f}'.format(accuracy/len(testloader))

def save_checkpoint(model, directory, train_datasets, accuracy, args):

    checkpoint = {
        'arch': args.arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': train_datasets.class_to_idx
    }
    
    path = args.save_dir + '/' + args.arch + str(accuracy) + '.pth'
    
    torch.save(checkpoint, path)