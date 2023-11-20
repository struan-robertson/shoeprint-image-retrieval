#!/usr/bin/env python3
import os
from PIL import Image
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import csv
from tqdm import tqdm

from joblib import Parallel, delayed, load, dump
import gc

# from line_profiler import profile

# ------ Networks ------

# Uses PyTorch
# from Networks.vgg19 import get_filters

# Uses Tensorflow
# from Networks.vgg19_tf import get_filters

from Networks.network_pt import Model

# ------ Similarity Measures ------

from Similarities.ncc import get_similarity

# from Similarities.orb import get_similarity

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
        img = Image.open(img_path)  # Convert to grayscale
        img_array = np.array(img)
        images.append(img_array)

    # Return list of images in directory
    return images

def get_all_filters(images, model):
    """
    Calculate the convolutional filters for each image in a list, returning a list of the corresponding filters.
    """
    image_filters = []

    # Calculate conv filter for each image
    for image in tqdm(images):
        filters = model.get_filters(image)
        image_filters.append(filters)

    # Return conv filters
    return image_filters

def memory_map(arr, name):
    folder = "Memmaps"
    filename = os.path.join(folder, name)
    if os.path.exists(filename): os.unlink(filename)
    _ = dump(arr, filename)

    mmap = load(filename, mmap_mode='r')

    return mmap

# TODO load mmaps if already exist?
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

    # Use default values for VGG19
    model = Model()

    # Calculate conv filters for print images
    print("Calculating convolutional filters for prints")
    print_filters = get_all_filters(print_images, model)

    # Calculate conv filters for shoe images
    print("Calculating convolutional filters for shoes")
    shoe_filters = get_all_filters(shoe_images, model)

    # TODO if trim enabled
    print_filters = trim(print_filters)
    shoe_filters = trim(shoe_filters)

    print_filters = [memory_map(print_, f"print_{id}") for id, print_ in enumerate(print_filters)]
    shoe_filters = [memory_map(shoe, f"shoe_{id}") for id, shoe in enumerate(shoe_filters)]

    gc.collect()

    return (print_filters, shoe_filters, matching_pairs)

def write_csv(filename, rankings):
    with open(f'Results/{filename}', 'w', newline='') as file:
        writer = csv.writer(file)
        for rank in rankings:
            writer.writerow([rank])

def trim(feature_map_group, threshold=0.2):
    trimmed = []
    for feature_map in feature_map_group:
        trimmed.append(Model.trim_filters(feature_map, threshold=threshold))

    return trimmed

def compare(print_filters, shoe_filters, matching_pairs):
    """
    Compare each print to every shoe, generating a probability score.
    The scores are used to calculate the ranking of the true match between a print and shoe in comparison to the other shoes.
    All rankings are then returned.
    """
    rankings = []

    # Progress bar to measure time taken per shoe
    pbar = tqdm(total=len(print_filters))

    # Loop through each set of print filters
    for id, print_ in enumerate(print_filters):
        # Update progressbar to reflect print being calculated
        pbar.desc = f"Print {id+1}"

        # Loop through each set of shoe filters
        # Try paralell here, each shoe and print is a numpy array and so can be passed in shared memory
        similarities = Parallel(n_jobs=32)(delayed(get_similarity)(print_, shoe) for shoe in shoe_filters)

        # for shoe_id, shoe in tqdm(enumerate(shoe_filters)):
        #     get_similarity(print_, shoe, print_trimmed[print_id], shoe_trimmed[shoe_id])

        # Sort similarities and then return the indexes in order of the sort
        # np.flip() is required as numpy sorts low -> high
        sorted = np.flip(np.argsort(similarities)) # type: ignore

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

# ------ Example Usage ------

# print_filters, shoe_filters, matching_pairs = initialise_data("../Data/FID-300")

# rankings = compare(print_filters, shoe_filters, matching_pairs)

# write_csv('results.csv', rankings)
