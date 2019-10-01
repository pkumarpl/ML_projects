# Imports here
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
    # Define a parser
    parser = argparse.ArgumentParser(description="Testing deep neural network model for  classification")

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='input image(required).',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='pre-trained model path',
                        required=True)
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to real names.')


    args = parser.parse_args()
    return args

# load_checkpoint(checkpoint_path) loads  from checkpoint
def load_checkpoint(checkpoint_path):
    # Load the saved file
    checkpoint = torch.load("my_checkpoint.pth")
    
    # Load Defaults if none specified
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Function process_image(image_path) performs cropping, scaling of image for our model
def process_image(image_path):
    test_image = PIL.Image.open(image_path)

    # Get original dimensions
    orig_width, orig_height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy 
    np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
    
    np_image = normalise_std * np_image + normalise_means
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def predict(image_tensor, model,  cat_to_name, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # check top_k
    if type(top_k) == type(None):
        top_k = 5
        print("Top K not specified, assuming K=5.")
    
    # Set model to evaluate
    model.eval()

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(image_tensor, 
                                                  axis=0)).type(torch.FloatTensor)
    cuda = torch.cuda.is_available()
    if cuda:
        # Move model parameters to the GPU
        model=model.cuda()
        print("GPU version")
    else:
        model=model.cpu()
        print("CPU version")
        

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = torch.topk(pred, topk)[0].tolist()[0] 
    
    top_labels = torch.topk(pred, topk)[1].tolist()[0] 
    
    # Convert to classes
    idx_to_class = []
    for i in range(len(model.class_to_idx.items())):
        class_to_idx.append(list(model.class_to_idx.items())[i][0])

    top_labels = []
    for i in range(5):
        top_labels.append(class_to_idx[index[i]])
    
    top_flowers = []
    for i in range(5):
        top_flowers.append(cat_to_name[index[i]])

    
    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
    

# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
    

    
    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)
    
    # Probabilities
    print_probability(top_flowers, top_probs)

# =============================================================================
# Execute
# =============================================================================
if __name__ == '__main__': main()