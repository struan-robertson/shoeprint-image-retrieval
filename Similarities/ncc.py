#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from fft_conv_pytorch import fft_conv

import ipdb

from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import fftconvolve
np.seterr(divide='ignore', invalid='ignore')

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

# Pytorch implementation of above function
def normxclorr_pt(template, image, padding="same"):

    template = template - torch.mean(template)
    image = image - torch.mean(image)

    image = image.unsqueeze(0).unsqueeze(0)
    template = template.reshape(1, 1, template.shape[0], template.shape[1])

    template = template.cuda()
    image = image.cuda()
    a1 = torch.ones(template.shape).cuda()

    out = fft_conv(image, template.conj(), padding=padding)

    image = fft_conv(torch.square(image), a1, padding=padding) - \
            torch.square(fft_conv(image, a1, padding=padding)) / (torch.prod(torch.tensor(template.shape)))

    out = out.cpu()
    image = image.cpu()
    template = template.cpu()

    # Remove small machine precision errors after subtraction
    image[torch.where(image < 0)] = 0

    template = torch.sum(torch.square(template))

    out = out / torch.sqrt(image * template)

    out[torch.where(torch.logical_not(torch.isfinite(out)))] = 0

    return out

# Pytorch implementation of normxclorr, batching multiple kernels and images to be run simultaniously
def normxclorr_pt_many(template, image, n_filters):

    n_batches = image.shape[0]

    # Subtract mean of individual templates
    for i in range(n_filters):
        template[i] -= torch.mean(template[i])
        # image[:,i] = image[:,i] - torch.mean(image[:,i])

    for b in range(n_batches):
        for i in range(n_filters):
            image[b,i] -= torch.mean(image[b,i])


    # Only do fft convolutions on GPU
    template = template.cuda()
    image = image.cuda()
    a1 = torch.ones(template.shape).cuda()

    out = fft_conv(image, template.conj(), padding="same", groups=n_filters)

    first_part = fft_conv(torch.square(image), a1, padding="same", groups=n_filters)
    second_part = torch.square(fft_conv(image, a1, padding="same", groups=n_filters))
    third_part = (torch.prod(torch.tensor(template.shape[2:]))) # Needs to be the size of the individual filters

    first_part = first_part.cpu()
    second_part = second_part.cpu()
    third_part = third_part.cpu()
    template = template.cpu()
    out = out.cpu()

    image = first_part - second_part / third_part

    # Remove small machine precision errors after subtraction
    image[torch.where(image < 0)] = 0

    for b in range(n_batches):
        for i in range(n_filters):
            template_sum = torch.sum(torch.square(template[i]))
            out[b,i] = out[b,i] / torch.sqrt(image[b,i] * template_sum)

    out[torch.where(torch.logical_not(torch.isfinite(out)))] = 0

    return out

def get_similarity(print_, shoe, device="cpu", gpu_fix=True):

    # Crop arrays by 2 pixels/edge to remove edge artifacts
    print_ = print_[:, 2:-2, 2:-2]

    if device == "gpu":
        shoe = shoe[:, :, 2:-2, 2:-2]
    else:
        shoe = shoe[:, 2:-2, 2:-2]

    # Number of filters for both shoe and print
    n_filters = len(print_)
    # Dimensions of shoe filters
    image_dims = shoe[0].shape
    # Dimensions of print filters
    template_dims = print_[0].shape


    # if debug:
    #     # print_test = print_[0].cpu().numpy()
    #     # shoe_test = shoe[0].cpu().numpy()
    #     # test = normxclorr_pt(print_test, shoe_test)

    #     # ipdb.set_trace()
    #     ncc_array_np = np.zeros(shoe.shape)
    #     ncc_array_pt = torch.zeros(shoe.shape)
    #     for index in range(n_filters):
    #         print_filter = print_[index]
    #         shoe_filter = shoe[index]

    #         ncc_array_np[index] = normxclorr2(print_filter.numpy(), shoe_filter.numpy(), "same")
    #         ncc_array_pt[index] = normxclorr_pt(print_filter, shoe_filter, "same").squeeze()

    #     ncc_array_np_sum = np.sum(ncc_array_np, axis=0)
    #     ncc_array_pt_sum = torch.sum(ncc_array_pt, dim=0)

    #     np_sim = np.max(ncc_array_np_sum) / n_filters
    #     pt_sim = torch.max(ncc_array_pt_sum) / n_filters

    #     # test = normxclorr2(print_test, shoe_test, "same")
    #     # test2 = normxclorr_pt(print_[0], shoe[0], "same")

    #     # print(pt_sim - np_sim)

    #     if np.sqrt((pt_sim - np_sim)**2) > 0.001:
    #         # test_np = normxclorr2(print_[0].numpy(), shoe[0].numpy(), "same")
    #         test_pt = normxclorr_pt(print_[0], shoe[0], "same")
    #         print(f"{print_[0].shape} : {shoe[0].shape}")

    if device == "gpu":

        # Very ugly hack, should not be necessary on GPUs actually supported by PyTorch
        # Pad height if one of the values that results in a corrupted FFT convolution
        if gpu_fix == True:
            broken_heights = [1, 8, 9, 16, 22, 29, 30, 33, 36, 41, 48, 50, 57, 61, 63, 67, 71, 74, 78, 84, 85, 87, 96]
            new_height = print_.shape[1]
            while new_height in broken_heights:
                new_height += 1
            if new_height != print_.shape[1]:
                number_zeros = new_height - print_.shape[1]
                print_ = F.pad(print_, (0, 0, number_zeros, 0), "constant", 0)
                template_dims = print_[0].shape

        # Since we're doing depthwise convolution, out_channels = in_channels = num_channels
        # and groups = num_channels, so in_channels/groups = 1
        print_ = print_.reshape(n_filters, 1, template_dims[0], template_dims[1])

        # Calculate NCC for all given shoes
        ncc_array = normxclorr_pt_many(print_, shoe, n_filters)

        # Sum values for each shoe
        sum_ncc = torch.sum(ncc_array, dim=1)

        # Calculate similarity for each individual shoe
        n_shoes = len(shoe)
        similarity = torch.zeros(n_shoes)
        for i in range(n_shoes):
            similarity[i] = torch.max(sum_ncc[i]) / n_filters


    elif device == "cpu":

        ncc_array = np.zeros(shoe.shape)

        for index in range(n_filters):
            print_filter = print_[index]
            shoe_filter = shoe[index]

            ncc_array[index] = normxclorr2(print_filter, shoe_filter, "same")

        ncc_array = np.sum(ncc_array, axis=0)

        similarity = np.max(ncc_array) / n_filters

    else:
        raise NotImplementedError(f"Device {device} not implemented")


    return similarity
