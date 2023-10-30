#!/usr/bin/env python3
import cv2
from numpy import squeeze

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

# Create subset model from first 14 layers of pre-trained VGG19
vgg19 = models.vgg19(weights="VGG19_Weights.DEFAULT")
# vgg19_subset = list(vgg19.features.children())[:23]
# TODO check that this is a ReLu layer
vgg19_subset = list(vgg19.features.children())[:25]

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(*vgg19_subset)
model = model.to(device)
model.eval()

# Contrast limited adaptive histogram equalisation
# TODO extract hyperparameters (clip limit and grid size) to external TOML file
# clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4, 4))
clahe = cv2.createCLAHE()

def get_filters(img):
    """
    Pass an image through the NN and return the resulting convolutional filters
    """

    # Apply CLAHE
    img = clahe.apply(img)

    # Image transforms before passing through network
    transform = transforms.Compose([
        # Convert image from numpy array to tensor
        # This automatically normalises the image to the range [0, 1]
        transforms.ToTensor(),
        # Repeat the channel of the image 3 times, as the pre-trained network expects a RGB image but the shoe images are grayscale
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    # Apply image transformations
    input_tensor = transform(img)
    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move the input batch to the appropriate device
    input_batch = input_batch.to(device)

    # Pass image through network
    with torch.no_grad():
        output = model(input_batch)

    # Move to CPU
    output = output.cpu()
    # Tensor to NumPy array
    output = output.numpy()

    # Remove batch dimension and return
    return squeeze(output)
