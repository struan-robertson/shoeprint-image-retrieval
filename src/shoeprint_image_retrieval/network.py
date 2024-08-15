"""Module for PyTorch model creation and use."""

import cv2
import torch
from numpy.typing import NDArray
from torch import nn
from torchvision import models, transforms
from torchvision.models.vgg import cast
from torchvision.transforms.functional import Any
from tqdm import tqdm

from .customtypes import FeatureMapsArrayType, ImageArrayType


def printmodel(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 1968, 5872),
) -> None:
    """Print model architecture using torchinfo.

    Args:
        model: A PyTorch model.
        input_shape: The shape of dummy data to pass through the model.
    """
    from torchinfo import summary  # pyright: ignore[reportUnknownVariableType]

    with torch.no_grad():
        print(summary(model, input_shape))


def get_output_size(
    model: "Model",
    input_shape: tuple[int, int, int, int] | torch.Tensor,
) -> torch.Size:
    """Get dimensions of output feature maps for a model with a specified input shape.

    Args:
        model: The Model object to calculate the output dimensions of.
        input_shape: The shape of the input tensor.

    Returns:
        Size of output tensor.
    """
    input_shape = cast(torch.Tensor, torch.randn(input_shape).cuda())  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportArgumentType]
    with torch.no_grad():
        output: torch.Tensor = model.model(input_shape)
    return output.shape


def _get_transforms(
    mean: list[float] | tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Compose image transformations with a specified mean and std.

    The default values are the default transforms used by PyTorch when training using ImageNet.
    This function works on grayscale images.
    """
    return transforms.Compose(
        [
            # Convert image from numpy array to tensor
            # This automatically normalises the image to the range [0, 1]
            transforms.ToTensor(),
            # Repeat the channel of the image 3 times, as the pre-trained network expects a RGB
            # image but the shoe images are grayscale
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType, reportUnknownArgumentType]
            # Normalize to same values used by pytorch when training on ImageNet
            transforms.Normalize(mean, std),
        ],
    )


def _get_transforms_rgb(
    mean: list[float] | tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Like get_transforms but works on RGB images."""
    return transforms.Compose(
        [
            # Convert image from numpy array to tensor
            # This automatically normalises the image to the range [0, 1]
            transforms.ToTensor(),
            # Normalize to same values used by pytorch when training on ImageNet
            transforms.Normalize(mean, std),
        ],
    )


class Model:
    """Operate on model and associated functions."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        model_str: str = "VGG19",
        layers: int = 23,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        """Initialise a model and associated function.

        Args:
            model_str: The name of the model.
            layers: The number of model layers to use.
            clip_limit: The clip limit for the CLAHE.
            tile_grid_size: The tile grid size for the CLAHE.
        """
        # CLAHE
        self.clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # TorchVision pre-trained network pre-processing parameters

        # Check if a GPU is available and if not, use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Select model from model string
        if model_str == "VGG19":
            model = models.vgg19(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "VGG16":
            model = models.vgg16(weights="IMAGENET1K_FEATURES")
            transform = _get_transforms(
                mean=[0.48235, 0.45882, 0.40784],
                std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098],
            )
            transform_rgb = _get_transforms_rgb(
                mean=[0.48235, 0.45882, 0.40784],
                std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098],
            )
        elif model_str == "VGG19_BN":
            model = models.vgg19_bn(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "EfficientNet_B1":
            model = models.efficientnet_b1(weights="IMAGENET1K_V2")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "EfficientNet_B2":
            model = models.efficientnet_b2(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "EfficientNet_B3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "EfficientNet_B4":
            model = models.efficientnet_b4(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "EfficientNet_B5":
            model = models.efficientnet_b5(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "EfficientNet_B7":
            model = models.efficientnet_b7(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        elif model_str == "EfficientNetV2_S":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
            transform_rgb = _get_transforms_rgb()
            transform = _get_transforms()
        elif model_str == "EfficientNetV2_M":
            model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
            transform_rgb = _get_transforms_rgb()
            transform = _get_transforms()
        elif model_str == "EfficientNetV2_L":
            model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
            # EfficientNetV2 L was trained on different mean and std deviation
            transform = _get_transforms(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            transform_rgb = _get_transforms_rgb(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_str == "DenseNet_201":
            model = models.densenet201(weights="IMAGENET1K_V1")
            transform = _get_transforms()
            transform_rgb = _get_transforms_rgb()
        else:
            e = "Model string not found"
            raise LookupError(e)

        # Get specified model layers
        model = list(model.features.children())[:layers]
        model = nn.Sequential(*model)

        # Create model
        model = model.to(self.device)

        _ = model.eval()

        self.model = model
        self.transform = transform
        self.transform_rgb = transform_rgb

    def _clahe(self, img: NDArray[Any]) -> NDArray[Any]:
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

    def get_feature_maps(self, img: ImageArrayType) -> FeatureMapsArrayType:
        """Pass an image through the model and return the resulting feature maps.

        Args:
            img: Image to pass through model.

        Returns:
            Array of feature maps calculated from image.
        """
        img = self._clahe(img)

        # Image pre-processing
        input_tensor: torch.Tensor = cast(
            torch.Tensor,
            self.transform_rgb(img) if img.ndim == 3 else self.transform(img),
        )

        # Add a batch dimension
        input_batch = input_tensor.unsqueeze(0)

        # Move the batch to the available device
        input_batch = input_batch.to(self.device)

        # Pass batch through network
        with torch.no_grad():
            output: torch.Tensor = self.model(input_batch)

        # Move ouput to CPU
        output = output.cpu()

        # Convert tensor to NumPy array
        output_numpy: FeatureMapsArrayType = cast(FeatureMapsArrayType, output.numpy())

        # Remove batch dimension
        return output_numpy.squeeze()

    def get_multiple_feature_maps(
        self,
        images: list[NDArray[Any]],
        *,
        progress: bool = True,
    ) -> list[FeatureMapsArrayType]:
        """Pass a list of images through the model and return a list of resulting feature maps.

        Args:
            images: The list of images.
            progress: Whether to show a tqdm progress bar.

        Returns:
            The resulting list of feature maps.
        """
        image_maps: list[FeatureMapsArrayType] = []

        iterator = tqdm(images) if progress else images

        for image in iterator:
            maps = self.get_feature_maps(image)
            image_maps.append(maps)

        return image_maps
