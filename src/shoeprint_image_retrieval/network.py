#!/usr/bin/env python3
"""Module for PyTorch model creation and use."""

from typing import Tuple
import cv2

import torch
import torch.nn as nn
import torchvision.models as models  # type:ignore
import torchvision.transforms as transforms  # type:ignore

"""Module used to pick a PyTorch model and then calculate convolutional filters from it."""


def printmodel(model, input=(1, 3, 1968, 5872)):
    """Print model architecture using torchinfo."""
    from torchinfo import summary

    with torch.no_grad():
        print(summary(model, input))


def get_output_size(model, input):
    """Get dimensions of output feature maps for a model with a specified input shape."""
    input = torch.randn(input).cuda()
    with torch.no_grad():
        output = model.model(input)
    return output.shape


def get_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Compose image transformations with a specified mean and std.

    The default values are the default transforms used by PyTorch when training using ImageNet.
    This function works on grayscale images.
    """
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

    return torchvision_transforms


def get_transforms_rgb(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Like get_transforms but works on RGB images."""
    torchvision_transforms = transforms.Compose(
        [
            # Convert image from numpy array to tensor
            # This automatically normalises the image to the range [0, 1]
            transforms.ToTensor(),
            # Normalize to same values used by pytorch when training on ImageNet
            transforms.Normalize(mean, std),
        ]
    )

    return torchvision_transforms


class Model:
    """Store model and  associated functions based on the model name, number of layers to use and CLAHE clip limit and tile size."""

    def __init__(
        self, model_str="VGG19", layers=23, clipLimit=2.0, tileGridSize=(8, 8)
    ) -> None:
        """Initialise a model and associated function.

        Args:
        ----
            model_str (str): The name of the model.
            layers (int): The number of model layers to use.
            clipLimit (float): The clip limit for the CLAHE.
            tileGridSize (tuple): The tile grid size for the CLAHE.
        """
        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        # TorchVision pre-trained network pre-processing parameters

        # Check if a GPU is available and if not, use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Select model from model string
        if model_str == "VGG19":
            model = models.vgg19(weights="IMAGENET1K_V1")
            transform = get_transforms()
            transform_rgb = get_transforms_rgb()
        elif model_str == "VGG16":
            model = models.vgg16(weights="IMAGENET1K_FEATURES")
            transform = get_transforms(
                mean=[0.48235, 0.45882, 0.40784],
                std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098],
            )
            transform_rgb = get_transforms_rgb(
                mean=[0.48235, 0.45882, 0.40784],
                std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098],
            )
        elif model_str == "VGG19_BN":
            model = models.vgg19_bn(weights="IMAGENET1K_V1")
            transform = get_transforms()
        elif model_str == "EfficientNet_B1":
            model = models.efficientnet_b1(weights="IMAGENET1K_V2")
            transform = get_transforms()
        elif model_str == "EfficientNet_B2":
            model = models.efficientnet_b2(weights="IMAGENET1K_V1")
            transform = get_transforms()
        elif model_str == "EfficientNet_B3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            transform = get_transforms()
        elif model_str == "EfficientNet_B4":
            model = models.efficientnet_b4(weights="IMAGENET1K_V1")
            transform = get_transforms()
        elif model_str == "EfficientNet_B5":
            model = models.efficientnet_b5(weights="IMAGENET1K_V1")
            transform = get_transforms()
        elif model_str == "EfficientNet_B7":
            model = models.efficientnet_b7(weights="IMAGENET1K_V1")
            transform = get_transforms()
            transform_rgb = get_transforms_rgb()
        elif model_str == "EfficientNetV2_S":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
            transform_rgb = get_transforms_rgb()
            transform = get_transforms()
        elif model_str == "EfficientNetV2_M":
            model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
            transform_rgb = get_transforms_rgb()
            transform = get_transforms()
        elif model_str == "EfficientNetV2_L":
            model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
            # EfficientNetV2 L was trained on different mean and std deviation
            transform = get_transforms(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_str == "DenseNet_201":
            model = models.densenet201(weights="IMAGENET1K_V1")
            transform = get_transforms()
        else:
            raise LookupError("Model string not found")

        model = list(model.features.children())[:layers]

        model = nn.Sequential(*model)

        # Create model
        model = model.to(self.device)  # pyright: ignore

        model.eval()

        self.model = model
        self.transform = transform
        self.transform_rgb = transform_rgb

    def _clahe(self, img):
        """Convert to lab and then back for applying CLAHE to RGB image."""
        if img.ndim == 3:
            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_img)
            clahe_l_channel = self.clahe.apply(l_channel)
            clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
            img = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2RGB)
        else:
            img = self.clahe.apply(img)

        return img

    def get_feature_maps(self, img) -> Tuple:
        """Pass an image through the model and return the resulting feature maps.

        Args:
        ----
            img (np.ndarray): Image to pass through model.

        Returns
        -------
            np.ndarray: Array of feature maps calculated from image.
        """
        img = self._clahe(img)

        # Image pre-processing
        if img.ndim == 3:
            input_tensor = self.transform_rgb(img)
        else:
            input_tensor = self.transform(img)

        # Add a batch dimension
        input_batch = input_tensor.unsqueeze(0)

        # Move the batch to the available device
        input_batch = input_batch.to(self.device)

        # Pass batch through network
        with torch.no_grad():
            output = self.model(input_batch)

        # Move ouput to CPU
        # output = output.cpu()

        # Convert tensor to NumPy array
        # output = output.numpy()

        output = output.squeeze()

        return output
