#!/usr/bin/env python3
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import VGG19 #pyright: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input #pyright: ignore
import tensorflow.keras.models as models #pyright: ignore

"""Module used to pick a TensorFlow model and then calculate convolutional filters from it."""

class Model:
    def __init__(self, model_str="VGG19", layers=14, clipLimit=2.0, tileGridSize=(8,8)) -> None:
        """Constructor for Model class.

        Args:
            model_str (str): The name of the model.
            layers (int): The number of model layers to use.
            clipLimit (float): The clip limit for the CLAHE.
            tileGridSize (tuple): The tile grid size for the CLAHE.
        """

        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        # Check if GPU available and if not, use the CPU
        self.device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'

        # Select model from model string
        if model_str == "VGG19":
            # Create model
            model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
            model.load_weights('Networks/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        # Create model
        model = models.Model(inputs=model.inputs, outputs=model.layers[layers].output) #pyright: ignore

        # Only use model for inference
        model.trainable = False

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

        # Add batch dimension and repeat the image across channels
        img = np.repeat(img[np.newaxis, :, :, np.newaxis], 3, axis=-1)

        # Image pre-processing
        img = preprocess_input(img)

        # Pass image through model
        with tf.device(self.device): #pyright: ignore
            output = self.model.predict(img, verbose=0)

        # Remove batch dimension
        output = np.squeeze(output)

        # Transpose from [j, i, batch] to [batch, j, i] to allow for modularity with pytorch function
        output = np.transpose(output, (2, 0, 1))

        return output
