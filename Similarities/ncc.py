#!/usr/bin/env python3

from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import cv2

def print_images(print_filter, shoe_filter, ncc):
    plt.subplot(1, 3, 1)
    plt.imshow(print_filter)
    plt.subplot(1, 3, 2)
    plt.imshow(shoe_filter)
    plt.subplot(1, 3, 3)
    plt.imshow(ncc)
    plt.show()

@njit
def g(conv_filter):
    """
    Calculate the ratio of pixels in the convolutional filter which are larger than 0.
    Uses Numba JIT compilation even though it is a fully Numpy function, as it is very performace critical.
    """
    return np.sum(conv_filter > 0) / conv_filter.size

def get_similarity(print_, shoe):
    # Number of filters for both shoe and print
    n_filters = len(shoe)
    # Dimensions of shoe filters
    image_dims = shoe[0].shape
    # Dimensions of print filters
    template_dims = print_[0].shape

    # Amount required to pad shoe filter to allow for template matching of every pixel
    pad_y = template_dims[0] // 2
    pad_x = template_dims[1] // 2

    # Dimension sizes of shoe filter
    y = image_dims[0]
    x = image_dims[1]

    # Array to hold computed normalised cross correlation maps
    ncc_array = np.empty((n_filters, y, x), dtype=np.float32)

    # Index of ncc_array to insert new values into
    # final_index = 0
    for index in range(n_filters):


        # Print and shoe filters
        print_filter = print_[index][1:-1, 1:-1]
        shoe_filter = shoe[index][1:-1, 1:-1]

        # Pad shoe filter to allow for template matching of every pixel
        padded_target = cv2.copyMakeBorder(shoe_filter, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0,))

        # Calculate NCC map, slicing result to match the same size as the input shoe filter
        # The slicing is required in cases where the padding is an even number
        ncc_array[index] = cv2.matchTemplate(padded_target, print_filter, cv2.TM_CCORR_NORMED)[:y, :x]


    # Slice ncc_array to only include computed NCC maps
    # ncc_array = ncc_array[:final_index]

    # number of NCC maps computed
    k = ncc_array.shape[0]

    # The similarity score is equal to the maximum pixel of the sum of all NCC maps divided by the number of NCC maps
    if k != 0:
        sum_ncc = np.sum(ncc_array, axis=0)
        similarity = np.max(sum_ncc) / k
        return(similarity)
    # If no NCC maps were computed (no filters passed the threshold), similarity is 0
    else:
        similarity = 0.0
        return(similarity)
