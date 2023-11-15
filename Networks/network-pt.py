#!/usr/bin/env python3
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

"""Module used to pick a PyTorch model and then calculate convolutional filters from it."""

class Model:
    def __init__(self, model_str="VGG19", layers=23, clipLimit=2.0, tileGridSize=(8,8)) -> None:
        """Constructor for Model class.

        Args:
            model_str (str): The name of the model.
            layers (int): The number of model layers to use.
            clipLimit (float): The clip limit for the CLAHE.
            tileGridSize (tuple): The tile grid size for the CLAHE.
        """

        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        # Image transforms
        # TODO: dont think this is the full transforms done when training on imagenet
        self.transform = transforms.Compose([
            # Convert image from numpy array to tensor
            # This automatically normalises the image to the range [0, 1]
            transforms.ToTensor(),
            # Repeat the channel of the image 3 times, as the pre-trained network expects a RGB image but the shoe images are grayscale
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        # Check if a GPU is available and if not, use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Select model from model string
        if model_str == "VGG19":
            # Create model
            model = models.vgg19(weights="VGG19_Weights.DEFAULT")

        # Select n layers from model
        model = list(model.features.children())[:layers] # pyright: ignore

        # Create model
        model = nn.Sequential(*model)
        model = model.to(self.device)
        model.eval()

        self.model = model

    def get_filters(self, img) -> np.ndarray:
        """Pass an image through the model and return the resulting feature maps.

        Args:
            img (np.ndarray): Image to pass through model.

        Returns:
            np.ndarray: Array of feature maps calculated from image.
        """

        # Apply CLAHE
        img = self.clahe.apply(img)

        # Image pre-processing
        input_tensor = self.transform(img)

        # Add a batch dimension
        input_batch = input_tensor.unsqueeze(0)

        # Move the batch to the available device
        input_batch = input_batch.to(self.device)

        # Pass batch through network
        with torch.no_grad():
            output = self.model(input_batch)

        # Move ouput to CPU
        output = output.cpu()

        # Convert tensor to NumPy array
        output = output.numpy()

        # Remove batch dimension and return
        return output.squeeze()
