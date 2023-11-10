#!/usr/bin/env python3

from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import fftconvolve
np.seterr(divide='ignore', invalid='ignore')

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


def normxclorr2(template, image, mode="full"):
    """
    https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template )
    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out

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
    ncc_array = np.empty((n_filters, y-2, x-2), dtype=np.float32)

    # Index of ncc_array to insert new values into
    final_index = 0
    for index in range(n_filters):

        # Remove outer pixel artefacts from prinat and shoe filter
        print_filter = print_[index][1:-1, 1:-1]
        shoe_filter = shoe[index][1:-1, 1:-1]

        # Pad shoe filter to allow for template matching of every pixel
        # padded_target = cv2.copyMakeBorder(shoe_filter, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0,))

        # Calculate NCC map, slicing result to match the same size as the input shoe filter
        # The slicing is required in cases where the padding is an even number
        # ncc_array[index] = cv2.matchTemplate(padded_target, print_filter, cv2.TM_CCORR_NORMED)[:y, :x]
        #
        T = 0.2

        if g(print_filter) > T and g(shoe_filter) > T:
            ncc_array[final_index] = normxclorr2(print_filter, shoe_filter, 'same')
            final_index += 1
        # import ipdb; ipdb.set_trace()


    # Slice ncc_array to only include computed NCC maps
    ncc_array = ncc_array[:final_index]

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
