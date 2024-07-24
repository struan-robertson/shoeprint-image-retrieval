#!/usr/bin/env python3

from sys import exit
from scipy.signal import convolve # type: ignore

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# def normxcorr(filter, image, mode="same"):
#     """
#     Calculate normalised cross-correlation between a filter and image using FFT convolutions.

#     Taken from https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py

#     Args:
#     ----
#         filter (np.ndarray): Cross-correlation filter.
#         image (np.ndarray): Search image to test correlation against.
#         mode (string): scipy.convolve mode options.
    
#     Returns
#     -------
#         np.ndarray: Cross-correlation similarity map.
#     """
#     filter = filter - np.mean(filter) 
#     image = image - np.mean(image)
#     a1 = np.ones(filter.shape)
#     # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
#     ar = np.flipud(np.fliplr(filter))

#     try:
#         out = convolve(image, ar, mode=mode) 

#         first_part = convolve(np.square(image), a1, mode=mode) 

#         second_part = np.square(convolve(image, a1, mode=mode)) 
#         third_part = (np.prod(filter.shape)) 

#         image = first_part - second_part / third_part 
#     except Exception:
#         print(f"Error {image.shape} {ar.shape}")
#         exit()


#     # Remove small machine precision errors after subtraction
#     image[np.where(image < 0)] = 0

#     filter = np.sum(np.square(filter))
#     out = out / np.sqrt(image * filter )
#     # Remove any divisions by 0 or very close to 0
#     out[np.where(np.logical_not(np.isfinite(out)))] = 0

#     return out

def normxcorr(template, image, mode="same"):
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

    template = template - np.mean(template) # -13.28 9.53
    image = image - np.mean(image) # -15.69 13.87
    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))

    try:
        out = convolve(image, ar, mode=mode) # -783.05 644

        # np.square(image) 0.0001 246
        first_part = convolve(np.square(image), a1, mode=mode) # 483 3453

        # image: -15.69 13.87
        second_part = np.square(convolve(image, a1, mode=mode)) # 0.006 67841
        third_part = (np.prod(template.shape)) # 102

        # image = convolve(np.square(image), a1, mode=mode) - \
        #         np.square(convolve(image, a1, mode=mode)) / (np.prod(template.shape))
        image = first_part - second_part / third_part # 467 3269
    except Exception as e:
        print(f"Error {image.shape} {ar.shape}")
        exit()


    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template )
    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out

def get_similarity(mark, print_):
    """
    Use normalised cross-correlation to calculate the similarity between the extracted feature maps of a shoemark and shoeprint.

    Args:
    ----
        mark (np.ndarray)   : Feature maps of shoemark.
        print_ (np.ndarray) : Feature maps of shoeprint.
    
    Returns
    -------
        float: Similarity score
    """
    # Crop arrays by 2 pixels/edge to remove edge artifacts
    mark = mark[:, 2:-2, 2:-2] 
    print_ = print_[:, 2:-2, 2:-2] 

    # Number of filters for both mark and print
    n_filters = len(mark)

    ncc_array = np.zeros(print_.shape)

    for index in range(n_filters):
        mark_map = mark[index]
        print_map = print_[index]

        ncc_array[index] = normxcorr(mark_map, print_map, "same")

    ncc_array = np.sum(ncc_array, axis=0)

    similarity = np.max(ncc_array) / n_filters

    return similarity
