#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from fft_conv_pytorch import fft_conv

import cupy as cp

import ipdb

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.signal import convolve

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

def normxcorr_simple(image, template, mode="same"):
    """
    Not proper NCC and slightly less accurate, however is considerably quicker to run
    """

    template = template - np.mean(template)
    image = image - np.mean(image)

    template = np.flipud(np.fliplr(template))

    out = convolve(image, template, mode=mode)

    # TODO if using this should really calculate std (and means) for all templates and images
    # before calling this function, would speed things up a lot
    out /= np.std(template) * np.std(image) * np.prod(template.shape)

    return out

# Pytorch implementation of above function
def normxcorr_pt(template, image, padding="same"):

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
def normxcorr_pt_many(template, image, n_filters):

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

def conv_fft_cupy(template, image):

    def power_of_two(n):
        if n <= 0:
            raise ValueError("Input must be a positive integer.")
        elif (n & (n - 1)) == 0:
            return n
        else:
            return 1 << (n.bit_length())

    n_filters = len(template)

    # Pad to equal size and ensure padded to power of 2
    _, template_height, template_width = template.shape
    _, image_height,  image_width  = image.shape

    # Calculate size to pad images to
    padded_height = power_of_two(image_height + template_height)
    padded_width = power_of_two(image_width + template_width)

    # extra_height = power_of_two(padded_height) - padded_height
    # extra_width = power_of_two(padded_width) - padded_width

    image_vert_padding = padded_height - image_height
    image_hor_padding  = padded_width  - image_width
    template_vert_padding = padded_height - template_height
    template_hor_padding  = padded_width  - template_width

    image_pad_top    = image_vert_padding // 2
    image_pad_bottom = image_vert_padding // 2 + image_vert_padding % 2
    image_pad_left   = image_hor_padding  // 2
    image_pad_right  = image_hor_padding  // 2 + image_hor_padding % 2

    template_pad_top    = template_vert_padding // 2
    template_pad_bottom = template_vert_padding // 2 + template_vert_padding % 2
    template_pad_left   = template_hor_padding  // 2
    template_pad_right  = template_hor_padding  // 2 + template_hor_padding % 2

    # Try reflective
    padded_template = cp.pad(template, ((0, 0), (template_pad_top, template_pad_bottom), (template_pad_left, template_pad_right)), mode='constant')
    padded_image = cp.pad(image, ((0,0), (image_pad_top, image_pad_bottom), (image_pad_left, image_pad_right)), mode='constant')

    # Pad to power of 2
    # padded_template = cp.pad(padded_template, ((0,0), (0, extra_height), (0, extra_width)), mode='constant')
    # padded_image = cp.pad(padded_image, ((0,0), (0, extra_height), (0, extra_width)), mode='constant')

    template_fft   = cp.fft.rfft2(padded_template,  axes=(1, 2))
    image_fft = cp.fft.rfft2(padded_image, axes=(1,2))

    # Multiply in the frequency domain
    product_fft = template_fft * image_fft

    # Convert back into time-domain
    result = cp.fft.irfft2(product_fft, axes=(1,2))

    result = cp.real(result)

    # Remove power of 2 padding
    # result = result[:, :-extra_height, :-extra_width]

    # Truncate padding
    # result = result[:, image_pad_top:image_height+image_pad_bottom, image_pad_right:image_width+image_pad_left]
    center_x = result.shape[2] // 2
    center_y = result.shape[1] // 2

    start_x = center_x - (image_width // 2)
    start_y = center_y - (image_height // 2)
    end_x = start_x + image_width
    end_y = start_y + image_height

    centered_slice = result[:, start_y:end_y, start_x:end_x]

    return centered_slice

def conv_many(image, template):
    # template = template.get()
    # image = image.get()

    result = np.zeros_like(image)

    n_filters = len(template)

    for i in range(n_filters):
        result[i] = convolve(image[i], template[i], mode="same")

    # return cp.array(result)
    return result

def get_similarity_cupy(print_, shoe):

    n_filters = len(print_)

    # print_ = cp.array(print_)
    # shoe = cp.array(shoe)
    # print_np = print_.get()
    # shoe_np = shoe.get()

    # ipdb.set_trace()
    # sim = get_similarity(print_np, shoe_np)

    # TODO remove edges and mean _before_ hand to speed up computation
    #
    # Crop arrays by 2 pixels/edge to remove edge artifacts
    print_ = print_[:, 2:-2, 2:-2] # -28.74 24.87
    shoe   = shoe[:, 2:-2, 2:-2] # -25.66 27.81

    # Calculate ZNCC

    # Subtract mean
    # print_ -= cp.mean(print_, axis=(1,2), keepdims=True) # -13.29 9.53
    # shoe   -= cp.mean(shoe, axis=(1,2), keepdims=True) # -15.70 13.87
    for i in range(n_filters):
        print_[i] -= cp.mean(print_[i])
        shoe[i] -= cp.mean(shoe[i])

    # ipdb.set_trace()

    a1 = cp.ones(print_.shape) # 1 1

    # Rotate as convolution is calculated but I need correlation
    ar = cp.rot90(print_, k=2, axes=(1,2)) # -12.68 9.60

    # print_ = print_.get()
    # shoe = shoe.get()
    # ar = ar.get()
    # a1 = a1.get()

    out = conv_fft_cupy(ar, shoe) # -1083.58 895.61
    # out = conv_many(shoe, ar)

    first_part  = conv_fft_cupy(a1, cp.square(shoe)) # 0 7480.11
    second_part = cp.square(conv_fft_cupy(a1, shoe)) # 0 94955.98
    # first_part = conv_many(np.square(shoe), a1) # 2120 3048
    # second_part = np.square(conv_many(shoe, a1)) # 11.43 39217
    third_part  = np.prod(print_.shape[1:]) # 102

    shoe = first_part - second_part / third_part # 0 3269.89

    # Remove machine precision errors
    shoe[cp.where(shoe < 0)] = 0 # 0 3269.89

    print_ = cp.sum(np.square(print_), axis=(1,2), keepdims=True) # 2320.22

    ncc_array = out / cp.sqrt(shoe * print_) # -0.32 0.27

    # Remove divisions by 0
    ncc_array[cp.where(cp.logical_not(cp.isfinite(ncc_array)))] = 0 # -0.405 0.46

    ncc_map = cp.sum(ncc_array, axis=0) # -4.78 4.04

    # TODO test also including negative correlation
    similarity = cp.max(ncc_map) / n_filters # 0.050

    # ipdb.set_trace()
    return similarity


def get_similarity(print_, shoe):

    # Crop arrays by 2 pixels/edge to remove edge artifacts
    print_ = print_[:, 2:-2, 2:-2] # -25.53 23.65
    shoe = shoe[:, 2:-2, 2:-2] # -25.66 27.81

    # Number of filters for both shoe and print
    n_filters = len(print_) # 80

    # Useful if using "full" padding
    # shape = (shoe.shape[0], shoe.shape[1] + print_.shape[1] -1, shoe.shape[2] + print_.shape[2] -1)
    # ncc_array = np.zeros(shape)
    ncc_array = np.zeros(shoe.shape)

    for index in range(n_filters):
        print_filter = print_[index]
        shoe_filter = shoe[index]

        ncc_array[index] = normxcorr(print_filter, shoe_filter, "same")
        # ncc_array[index] = normxcorr_simple(print_filter, shoe_filter, "same")

    # - 4.78 5.08
    ncc_array = np.sum(ncc_array, axis=0)

    similarity = np.max(ncc_array) / n_filters

    return similarity

def get_similarity_gpu(print_, shoe, gpu_fix=True):

    print_ = print_[:, 2:-2, 2:-2]

    shoe = shoe[:, :, 2:-2, 2:-2]

    # Number of filters for both shoe and print
    n_filters = len(print_)
    # Dimensions of shoe filters
    image_dims = shoe[0].shape
    # Dimensions of print filters
    template_dims = print_[0].shape

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
    ncc_array = normxcorr_pt_many(print_, shoe, n_filters)

    # Sum values for each shoe
    sum_ncc = torch.sum(ncc_array, dim=1)

    # Calculate similarity for each individual shoe
    n_shoes = len(shoe)
    similarity = torch.zeros(n_shoes)
    for i in range(n_shoes):
        similarity[i] = torch.max(sum_ncc[i]) / n_filters

    return similarity
