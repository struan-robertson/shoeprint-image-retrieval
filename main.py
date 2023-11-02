#!/usr/bin/env python3
import os
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from numba import njit

from joblib import Parallel, delayed, load, dump
import tempfile
import uuid
import gc

# from line_profiler import profile

# Uses PyTorch
# from vgg19 import get_filters

# Uses Tensorflow
from vgg19_tf import get_filters

def load_images(dir):
    """
    Load images into an array, sorted by name.
    As the image name corresponds to its ID, the index of the returned array corresponds to the image ID - 1.
    """

    # List all files in the directory
    image_files = os.listdir(dir)
    # Sort by name
    image_files.sort()

    # Load all images into list
    images = []
    for image_file in image_files:
        img_path = os.path.join(dir, image_file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        images.append(img_array)

    # Return list of images in directory
    return images

def get_all_filters(images):
    """
    Calculate the convolutional filters for each image in a list, returning a list of the corresponding filters.
    """
    image_filters = []

    # Calculate conv filter for each image
    for image in tqdm(images):
        filters = get_filters(image)
        image_filters.append(filters)

    # Return conv filters
    return image_filters


def initialise_data(data_dir):
    """
    Load all required state for testing.
    Loads:
     - Convolutional filters for print and shoe images
     - A dictionary containing the matching pairs of prints and shoes
    """

    # Directories containing images
    print_dir = os.path.join(data_dir, "tracks_cropped")
    shoe_dir = os.path.join(data_dir, "references")

    # Load images in print directory
    print_images = load_images(print_dir)
    print("Loaded ", len(print_images), " prints")

    # Load images in shoe directory
    shoe_images = load_images(shoe_dir)
    print("Loaded ", len(shoe_images), " shoes")

    # Load matching pairs from csv tabke into dictionary
    matching_pairs = {}
    with open(os.path.join(data_dir, "label_table.csv"), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matching_pairs[int(row[0])] = int(row[1])

    # Calculate conv filters for print images
    print("Calculating convolutional filters for prints")
    print_filters = get_all_filters(print_images)

    # Calculate conv filters for shoe images
    print("Calculating convolutional filters for shoes")
    shoe_filters = get_all_filters(shoe_images)

    return (print_filters, shoe_filters, matching_pairs)


@njit
def g(conv_filter):
    """
    Calculate the ratio of pixels in the convolutional filter which are larger than 0.
    Uses Numba JIT compilation even though it is a fully Numpy function, as it is very performace critical.
    """
    return np.sum(conv_filter > 0) / conv_filter.size

def print_images(print_filter, shoe_filter, ncc):
    plt.subplot(1, 3, 1)
    plt.imshow(print_filter)
    plt.subplot(1, 3, 2)
    plt.imshow(shoe_filter)
    plt.subplot(1, 3, 3)
    plt.imshow(ncc)
    plt.show()

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


# TODO Implement cv2 code in numpy and python code, so that this can be parallelised using numba
def compare(print_filters, shoe_filters, matching_pairs):
    """
    Compare each print to every shoe, generating a probability score.
    The scores are used to calculate the ranking of the true match between a print and shoe in comparison to the other shoes.
    All rankings are then returned.
    """
    rankings = []
    # Threshold TODO extract hyperparameters into toml file
    # T = 0.2

    # Progress bar to measure time taken per shoe
    pbar = tqdm(total=len(print_filters))

    # Loop through each set of print filters
    for id, print_ in enumerate(print_filters):
        # Update progressbar to reflect print being calculated
        pbar.desc = f"Print {id+1}"

        # Loop through each set of shoe filters
        # Try paralell here, each shoe and print is a numpy array and so can be passed in shared memory
        similarities = Parallel(n_jobs=10, prefer="threads")(delayed(get_similarity)(print_, shoe) for shoe in shoe_filters)

        # Sort similarities and then return the indexes in order of the sort
        # np.flip() is required as numpy sorts low -> high
        sorted = np.flip(np.argsort(similarities))

        # Find the rank of the true match within the sorted array
        # matching_pairs[id+1] because the image id is equal to index + 1
        # Add 1 to the result as the resulting index calculated will be 1 less than the true rank
        rank = np.where(sorted == (matching_pairs[id+1] -1))[0][0] +1
        rankings.append(rank)

        # Update progress bar
        pbar.update()

        # Print result
        pbar.write(f"Print {id+1} true match ranked {rank}")
        # Reset progress bar to 0

    return rankings



# print_filters, shoe_filters, matching_pairs = initialise_data("../Data/FID-300")

# rankings = compare(print_filters, shoe_filters)

# print(rankings)

# with open('results.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for rank in rankings:
#         writer.writerow([rank])
