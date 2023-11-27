#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import timm

"""Module used to pick a PyTorch model and then calculate convolutional filters from it."""

def printmodel(model, input=(1,3,222,265)):
    from torchinfo import summary
    print(summary(model, input))

class Model:
    def __init__(
        self, model_str="VGG19", layers=23, clipLimit=2.0, tileGridSize=(8, 8)
    ) -> None:
        """Constructor for Model class.

        Args:
            model_str (str): The name of the model.
            layers (int): The number of model layers to use.
            clipLimit (float): The clip limit for the CLAHE.
            tileGridSize (tuple): The tile grid size for the CLAHE.
        """

        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        # TorchVision pre-trained network pre-processing parameters
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        torchvision_transforms = transforms.Compose(
            [
                # Convert image from numpy array to tensor
                # This automatically normalises the image to the range [0, 1]
                transforms.ToTensor(),
                # Repeat the channel of the image 3 times, as the pre-trained network expects a RGB image but the shoe images are grayscale
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                # Normalize to same values used by pytorch when training on ImageNet
                transforms.Normalize(mean, std),
            ]
        )

        # Check if a GPU is available and if not, use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Select model from model string
        if model_str == "VGG19":
            model = models.vgg19(weights='IMAGENET1K_V1')
            transform = torchvision_transforms
        elif model_str == "VGG19_BN":
            model = models.vgg19_bn(weights='IMAGENET1K_V1')
            transform = torchvision_transforms
        elif model_str == "EfficientNet_B1":
            model = models.efficientnet_b1(weights="IMAGENET1K_V2")
            transform = torchvision_transforms
        elif model_str == "EfficientNet_B2":
            model = models.efficientnet_b2(weights="IMAGENET1K_V1")
            transform = torchvision_transforms
        elif model_str == "EfficientNet_B3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            transform = torchvision_transforms
        elif model_str == "EfficientNet_B4":
            model = models.efficientnet_b4(weights="IMAGENET1K_V1")
            transform = torchvision_transforms
        else:
            raise LookupError("Model string not found")

        # elif model_str == "EfficientNet":
        #     model = timm.create_model("efficientnet_b1.ft_in1k", pretrained=True, features_only=True)
        #     data_config = timm.data.resolve_model_data_config(model)
        #     self.__timm_transform = timm.data.create_transform(**data_config, is_training=False)

        #     transform = self.__timm_transform__

        model = list(model.features.children())[:layers]  # pyright: ignore
        model = nn.Sequential(*model)

        # Create model
        model = model.to(self.device)  # pyright: ignore

        # printmodel(model)
        import ipdb; ipdb.set_trace()

        model.eval()

        self.model = model
        self.transform = transform

    def __timm_transform__(self, img):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        # Convert to PIL Image
        img = Image.fromarray(img)

        return self.__timm_transform(img)  # pyright: ignore

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
