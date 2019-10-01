import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, models, datasets
from torch.autograd import Variable
from torch import optim, nn

import random, os
import seaborn as sns
import torch
import time, json
import collections
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description=""" Train.py will train a new neural network on a dataset and save the model as a checkpoint. Optionally, you can also set et directory to save checkpoints: python train.py data_dir --save_dir save_directory otherwise it will be saved in CWD \n
Choose architecture: python train.py data_dir --arch "vgg16" \n
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 \n
Use GPU for training: python train.py data_dir --gpu \n
 """, epilog='\n Basic usage: python train.py data_directory \n')

    parser.add_argument('data_directory', action = 'store', default='./flowers/', help = 'Data directory of traning dataset (required)')

    parser.add_argument('--arch', type=str,
                        help='Choose architecture from torchvision.models as str')

    

    parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

    parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001')

    parser.add_argument('--dropout', type=float, default = 0.05,
                    help = 'Enter dropout for training the model, default is 0.05')

    parser.add_argument('--hidden_units',
                    type=int, default = 512,
                    help = 'Enter number of hidden units in classifier, default is 512')

    parser.add_argument('--epochs', type = int, default = 20,
                    help = 'Enter number of epochs to use during training, default is 20')

    parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Use GPU + Cuda for calculations')
    args = parser.parse_args()
    return args

# Define transforms for the training data and testing data
def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

# Load the datasets with ImageFolder
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data


# Using the image datasets and the trainforms, define the dataloaders
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader

#check GPU availbility
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


# primaryloader_model(architecture="vgg16") from torchvision

def primaryloader_model(architecture="vgg16"):
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model

# creates a classifier with the corect number of input layers
def initial_classifier(model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
        print("Number of Hidden Layers specificed as 4096.")
    
    # Find Input Layers
    input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(collections.OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# Validates training against testloader to return loss and accuracy
def validation(model, testloader, criterion, device):
    test_loss    = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy
    

# Function network_trainer represents the training of the network model
def network_trainer(model, Trainloader, validloader, Device, 
                  Epochs, print_every, count, learning_rate):
    

    if type(Epochs) == type(None):
        Epochs = 12
        print(f'Number of Epochs specificed: {Epochs}')    
 
    
    start = time.time()
    # start train Model
    print('Training started')
    
    for e in range(Epochs):
        running_loss = 0
        model.train() 
        train_mode = 0
        valid_mode = 1
        for i, (inputs, labels) in enumerate(Trainloader):
                count += 1
            
                inputs, labels = Variable(inputs.to(Device)), Variable(labels.to(Device))
                # Define loss and optimizer
                criterion = nn.NLLLoss()
                optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
                optimizer.zero_grad()
            
                # Forward and backward passes
                outputs = model.forward(inputs)
     

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()

                ps = torch.exp(outputs).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

        if i == train_mode:
            print(f'Epoch: {e+1, epochs}')
            print(f'Training Loss: {running_loss/pass_count}')
        else:
            print(f'Validation Loss: {running_loss/pass_count}')
            print(f'Accuracy: {accuracy}')

        running_loss = 0
     
    time_elapsed = time.time() - start
    print(f'Total time: {time_elapsed//60, time_elapsed % 60}')

    return model

#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def validate_model(model, testloader, device):
   # Do validation on the test set
    model.eval()
    accuracy = 0
    
    count = 0

    for data in Testloader:
        count += 1
        images, labels = data
        cuda = torch.cuda.is_available()
        if cuda == True:
            images, labels = Variable(images.to(Device)), Variable(labels.to(Device))
        else:
            images, labels = Variable(images), Variable(labels)

    output = model.forward(images)
    ps = torch.exp(output).data
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()
    print(f'Accuracy achieved by the network on test images is: {accuracy/count}')

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will be saved in cwd")
    else:
        if isdir(Save_Dir):
            # Create `class_to_idx` attribute in model
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'state_dict': model.state_dict(),
                          'classifier': model.classifier,
                          'class_to_idx': train_data.class_to_idx,
                          'opt_state': optimizer.state_dict,
                          'num_epochs': epochs}
            
            # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')

        else: 
            print("Directory not found, model will be saved in CWD.")
            torch.save(checkpoint, 'my_checkpoint.pth')


# =============================================================================
# Main Function
# =============================================================================

# Function main() is where all the above functions are called and executed 
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
   
    # Build Classifier
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu)
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: 
        learning_rate = args.learning_rate
    
    
    
    # Define deep learning method
    print_every = 30
    count = 0
    

    
    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader, validloader, 
                                  device,  args.epochs, 
                                  print_every, count, args.learning_rate)
    
    print("\nTraining process is now complete!!")
    
    # Quickly Validate the model
    validate_model(trained_model, testloader, device)
    
    # Save the model
    initial_checkpoint(trained_model, args.save_dir, train_data)


# =============================================================================
# Run Program
# =============================================================================

if __name__ == '__main__': main()
