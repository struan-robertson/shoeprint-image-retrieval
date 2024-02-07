#!/usr/bin/env python3

import numpy as np
from random import randint

import torch

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

    # Scan as Iquery
    # Scan will need to be padded
    # num_parts = int(np.ceil((scan_dims[0]-4) / (print_dims[0]-4)))
    # num_parts = 3
    # part_len = int((scan_dims[0]-4) // num_parts)

    score = np.zeros((scan_dims[0]-4, scan_dims[1]-4))

    for index in range(n_filters):
        # Remove outer pixel artefacts from prinat and shoe filter
        print_filter = print_[index][2:-2, 2:-2]
        scan_filter = shoe[index][2:-2, 2:-2]

        # Pad shoe filter
        # scan_filter = np.pad(scan_filter, [(print_dims[0], print_dims[0]), (print_dims[1], print_dims[1])], mode='symmetric') #pyright: ignore

        # Split shoe filter into templates
        # templates = []
        # for i in range(num_parts):
        #     start_index = i * part_len
        #     end_index = start_index + part_len
        #     if end_index > len(shoe_filter):
        #         continue
        #     # Slice the array and append to the list
        #     template = shoe_filter[start_index:end_index, :]
        #     templates.append(preprocess(template))

        # Select 4 random filters
        selected_filters = []
        templates = []

        templates.append(preprocess(print_filter))

        for i in range(4):
            rand_index = randint(0, n_filters-1)
            while rand_index in selected_filters:
                rand_index = randint(0, n_filters-1)

            rand_filter = print_[rand_index][2:-2, 2:-2]
            selected_filters.append(rand_index)

            templates.append(preprocess(rand_filter))

        scan_filter = preprocess(scan_filter)

        y = dim_activation_conv(templates, scan_filter, [], np.array([]), 10)

        # y = y[print_dims[0]:-print_dims[0], print_dims[1]:-print_dims[1], :]

        score += y[:,:,0]

    # Return score for highest scoring section
    # print(score)
    return np.max(score)
