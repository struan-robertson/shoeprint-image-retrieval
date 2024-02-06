#!/usr/bin/env python3

import numpy as np

from Similarities.dim_activation_conv import dim_activation_conv

import ipdb

def preprocess(img):
    positive = np.where(img > 0, img, 0)

    negative = np.where(img < 0, img, 0)
    negative = negative * -1

    return np.stack((positive, negative), axis=2)

def get_similarity(print_, shoe):

    n_filters = len(shoe)

    scan_dims = shoe[0].shape
    print_dims = print_[0].shape

    # num_parts = int(np.ceil((scan_dims[0]-4) / (print_dims[0]-4)))
    num_parts = 3
    part_len = int((scan_dims[0]-4) // num_parts)

    pad = [part_len, int(scan_dims[1])]

    score = np.zeros((print_dims[0]-4, print_dims[1]-4, num_parts))

    for index in range(n_filters):
        # Remove outer pixel artefacts from prinat and shoe filter
        print_filter = print_[index][2:-2, 2:-2]
        shoe_filter = shoe[index][2:-2, 2:-2]

        # Pad shoe filter
        print_filter = np.pad(print_filter, [(pad[0], pad[0]), (pad[1], pad[1])], mode='symmetric') #pyright: ignore

        # Split shoe filter into templates
        templates = []
        for i in range(num_parts):
            start_index = i * part_len
            end_index = start_index + part_len
            if end_index > len(shoe_filter):
                continue
            # Slice the array and append to the list
            template = shoe_filter[start_index:end_index, :]
            templates.append(preprocess(template))


        print_filter = preprocess(print_filter)

        y = dim_activation_conv(templates, print_filter, [], np.array([]), 5)
        y = y[pad[0]:-pad[0], pad[1]:-pad[1], :]

        for i in range(y.shape[2]):
            score[:,:,i] += y[:,:,i]

    # Return score for highest scoring section
    # print(score)
    # ipdb.set_trace()
    return np.max(score)
