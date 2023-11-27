#!/usr/bin/env python3

import tensorflow as tf
from keras.applications import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import cv2
import numpy as np

vgg19 = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
vgg19.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Do they not realise that this returns 15 layers?
model = Model(inputs=vgg19.inputs, outputs=vgg19.layers[14].output)

# Load pre-trained VGG19, excluding classifier layers
# vgg19 = VGG19(weights='imagenet', include_top=False)

# Create a subset model from the first 14 layers of VGG19
# model = Model(inputs=vgg19.input, outputs=vgg19.layers[13].output)

# Only using model for inference
model.trainable = False

# Move device to GPU if available
device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'

# Contrast limited adaptive histogram equalisation
# TODO extract hyperparameters (clip limit and grid size) to external TOML file
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# clahe = cv2.createCLAHE()

def get_filters(img):
    """
    Pass an image through the NN and return the resulting convolutional filters
    """

    # Apply CLAHE
    img = clahe.apply(img)

    # Normalize the image to be in the range [0, 1]
    # img = img.astype('float32') / 255.0

    # Add batch dimension and repeat the image across channels
    img = np.repeat(img[np.newaxis, :, :, np.newaxis], 3, axis=-1)

    # if len(img.shape) == 2:
    #     img = np.stack((img)*3, axis=-1)

    # import ipdb; ipdb.set_trace()

    # # create batch dimension
    # img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    # Run the model and get the filter responses
    with tf.device(device):
        output = model.predict(img, verbose=0)

    # Remove batch dimension
    output = np.squeeze(output)
    # Transpose from [x, y, batch] to [batch, x, y] to allow for modularity with pytorch function
    output = np.transpose(output, (2, 0, 1))

    return output
