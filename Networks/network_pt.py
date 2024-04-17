#!/usr/bin/env python3
from typing import Tuple
import ipdb
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import timm

"""Module used to pick a PyTorch model and then calculate convolutional filters from it."""

def printmodel(model, input=(1,3,1968,5872)):
    from torchinfo import summary
    print(summary(model, input))

def get_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
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

        # Store activations for hooks
        self.activation = {}

        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        # TorchVision pre-trained network pre-processing parameters

        # Check if a GPU is available and if not, use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Select model from model string
        if model_str == "VGG19":
            model = models.vgg19(weights='IMAGENET1K_V1')
            transform = get_transforms()
            transform_rgb = get_transforms_rgb()
        elif model_str == "VGG16":
            model = models.vgg16(weights='IMAGENET1K_FEATURES')
            transform = get_transforms(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
            transform_rgb = get_transforms_rgb(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
        elif model_str == "VGG16_Stylized":
            model = models.vgg16()
            state_dict = torch.load("./Networks/vgg16_train_60_epochs_lr0.01.pth")["state_dict"]

            # convert from old state dict
            for i in [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]:
                state_dict[f"features.{i}.weight"] = state_dict.pop(f"features.module.{i}.weight")
                state_dict[f"features.{i}.bias"] = state_dict.pop(f"features.module.{i}.bias")

            model.load_state_dict(state_dict)

            transform = get_transforms()
        elif model_str == "VGG19_BN":
            model = models.vgg19_bn(weights='IMAGENET1K_V1')
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

        # elif model_str == "EfficientNet":
        #     model = timm.create_model("efficientnet_b1.ft_in1k", pretrained=True, features_only=True)
        #     data_config = timm.data.resolve_model_data_config(model)
        #     self.__timm_transform = timm.data.create_transform(**data_config, is_training=False)

        #     transform = self.__timm_transform__
        #

        # state_dict = torch.load("../effnet-finetune/159000_1.8319744295621783.pth")

        # model.load_state_dict(state_dict)


        ipdb.set_trace()
        model = list(model.features.children())[:layers]  # pyright: ignore

        model = nn.Sequential(*model)

        # DDIS layers 1_2, 3_4, 4_4
        # h1 = model[3].register_forward_hook(self.__get_activation__('1_2'))
        # h2 = model[17].register_forward_hook(self.__get_activation__('3_4'))

        # Create model
        model = model.to(self.device)  # pyright: ignore

        # printmodel(model)

        model.eval()

        self.model = model
        self.transform = transform
        self.transform_rgb = transform_rgb

    # Handle RGB aswel as grayscale
    def _clahe(self, img):
        if img.ndim == 3:
            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_img)
            clahe_l_channel = self.clahe.apply(l_channel)
            clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
            img = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2RGB)
        else:
            img = self.clahe.apply(img)

        return img


    def __get_activation__(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def __timm_transform__(self, img):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        # Convert to PIL Image
        img = Image.fromarray(img)

        return self.__timm_transform(img)  # pyright: ignore

    def get_filters(self, img) -> Tuple:
        """Pass an image through the model and return the resulting feature maps.

        Args:
            img (np.ndarray): Image to pass through model.

        Returns:
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

        # DDIS layers 1_2, 3_4, 4_4
        # TODO make more general
        # layer_1_2 = self.activation["1_2"]
        # layer_3_4 = self.activation["3_4"]

        # layer_1_2 = layer_1_2.cpu()
        # layer_3_4 = layer_3_4.cpu()

        # layer_1_2 = layer_1_2.numpy()
        # layer_3_4 = layer_3_4.numpy()

        # layer_1_2 = layer_1_2.squeeze()
        # layer_3_4 = layer_3_4.squeeze()

        # Convert tensor to NumPy array
        # output = output.numpy()
        output = output.squeeze()

        # Remove batch dimension and return
        # return (layer_1_2, layer_3_4, output)
        return output
