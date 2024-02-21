#!/usr/bin/env python3
import os
from PIL import Image
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import csv
from tqdm import tqdm
import time
import sys

# GPU
import torch

# CPU
from multiprocessing import Array, Process, Value, Lock, Queue

import ipdb

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
# from Similarities.dim import get_similarity


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
        filters = filters.cpu()
        image_filters.append(filters)

    # Return conv filters
    return image_filters

# def memory_map(arr, name):
#     folder = "Memmaps"
#     filename = os.path.join(folder, name)
#     if os.path.exists(filename): os.unlink(filename)
#     _ = dump(arr, filename)

#     mmap = load(filename, mmap_mode='r')

#     return mmap

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

    # model = Model("EfficientNet_B5", 7)
    # model = Model("VGG19", 27)
    model = Model("EfficientNetV2_M", 4)
    # model = Model("EfficientNetV2_M", 1)
    # model = Model("VGG16", 23)

    # Calculate conv filters for print images
    print("Calculating convolutional filters for prints")
    print_filters = get_all_filters(print_images, model)

    # Calculate conv filters for shoe images
    print("Calculating convolutional filters for shoes")
    shoe_filters = get_all_filters(shoe_images, model)


    # print_filters = [memory_map(print_, f"print_{id}") for id, print_ in enumerate(print_filters)]
    # shoe_filters = [memory_map(shoe, f"shoe_{id}") for id, shoe in enumerate(shoe_filters)]


    # gc.collect()
    return (print_filters, shoe_filters, matching_pairs)

def write_csv(filename, rankings):
    with open(f'Results/{filename}', 'w', newline='') as file:
        writer = csv.writer(file)
        for rank in rankings:
            writer.writerow([rank])

def get_rank(similarities, matching_pairs, print_id):

    # Sort similarities and then return the indexes in order of the sort
    # np.flip() is required as numpy sorts low -> high
    sorted = np.flip(np.argsort(similarities)) # type: ignore
    # sorted = np.argsort(similarities)

    # Find the rank of the true match within the sorted array
    # matching_pairs[id+1] because the image id is equal to index + 1
    # Add 1 to the result as the resulting index calculated will be 1 less than the true rank
    rank = np.where(sorted == (matching_pairs[print_id+1] -1))[0][0] +1

    return rank

def worker(print_filters, shoe_filters, print_ids, matching_pairs, rankings, counter, queue):

    for i in range(len(shoe_filters)):
        shoe_filters[i] = np.frombuffer(shoe_filters[i][0].get_obj(), dtype=np.float32).reshape(shoe_filters[i][1])

    # Use the range of print_ids passed to the worker as the id of print_
    for print_, print_id in zip(print_filters, range(*print_ids)):

        similarities = []
        for shoe in shoe_filters:

            sim = get_similarity(print_, shoe, device="cpu")

            similarities.append(sim)

        rank = get_rank(similarities, matching_pairs, print_id)

        rankings[print_id] = rank

        with counter.get_lock():
            counter.value += 1

        queue.put(f"Print {print_id+1} true match ranked {rank}")


def compare(print_filters, shoe_filters, matching_pairs, device="cpu", n_processes=32, gpu_fix=True):
    """
    Compare each print to every shoe, generating a probability score.
    The scores are used to calculate the ranking of the true match between a print and shoe in comparison to the other shoes.
    All rankings are then returned.
    """

    # GPU single threaded as multithreading seemed to only add overhead
    if device == "gpu":

        rankings = []

        # Batch shoe scans by shape
        # TODO not usefull currently as does not track shoe id
        # somehow takes longer than not batching anyway so no incentive to fix
        #
        # This is effective as being scans many have the exact same dimensions
        # As all the scans must be cross correlated with the print filters, they can be
        # batched for parallelism.
        # shoes_by_shape = {}
        # for shoe in shoe_filters:
        #     shape = shoe.shape
        #     if shape not in shoes_by_shape:
        #         shoes_by_shape[shape] = [shoe]
        #     else:
        #         shoes_by_shape[shape].append(shoe)

        # # Create batches from grouped shapes
        # shoe_filters = []
        # for shape, group in shoes_by_shape.items():
        #     # if len(group) > 1:
        #     #     shoe_filters.append(torch.stack(group, dim=0))
        #     # else:
        #     #     # Add batch dimension
        #     #     shoe_filters.append(group[0].unsqueeze(0))
        #     split = [group[i:i + 40] for i in range(0, len(group), 40)]
        #     for s in split:
        #         shoe_filters.append(torch.cat(s).unsqueeze(0))


        pbar = tqdm(total=len(print_filters), file=sys.stdout)
        for print_id, print_ in enumerate(print_filters):

            similarities = []

            for shoe in shoe_filters:

                new_sims = get_similarity(print_, shoe.unsqueeze(0), device="gpu", gpu_fix=gpu_fix).numpy()
                for sim in new_sims:
                    similarities.append(sim)

            rank = get_rank(similarities, matching_pairs, print_id)
            rankings.append(rank)

            pbar.update()
            pbar.write(f"Print {print_id+1} true match ranked {rank}")

        return rankings

    elif device == "cpu":

        if type(print_filters[0]) == torch.Tensor:
            for i in range(len(print_filters)):
                print_filters[i] = print_filters[i].numpy()
        if type(shoe_filters[0]) == torch.Tensor:
            for i in range(len(shoe_filters)):
                shoe_filters[i] = shoe_filters[i].numpy()

        # Chunk prints by number of processes
        # Each process runs the computations for the prints given to it
        # the shoe scan filters in shared memory
        chunk_size = len(print_filters) // n_processes
        chunk_extra = len(print_filters) % n_processes
        chunks = []
        print_ids = []
        start = 0
        for i in range(n_processes):
            # Distribute extra items if required
            end = start + chunk_size + (1 if i < chunk_extra else 0)
            chunks.append(print_filters[start:end])
            print_ids.append((start, end)) # range of print ids contained in the chunk
            start = end


        # Shared memory variables
        counter = Value('i', 0) # Counter to contain number of prints processed
        queue = Queue() # FIFO queue so that processes can write tqdm messages

        n_prints = len(print_filters)
        rankings = Array('i', n_prints) # Rankings of each print

        # Store shoe scan filters in shared memory as these are required for every print
        shoe_shared = []
        for i in range(len(shoe_filters)):
            shape = shoe_filters[i].shape

            # Copy shoe_filters[i] into shared memory array
            shared = Array('f', shoe_filters[i].size)
            np_shared = np.frombuffer(shared.get_obj(), dtype=np.float32).reshape(shape) #pyright: ignore
            np.copyto(np_shared, shoe_filters[i])

            shoe_shared.append((shared, shape))

        # Debug
        # worker(chunks[0], shoe_shared, print_ids[0], matching_pairs, rankings, counter, queue)

        # Spawn each process
        processes = []
        for i in range(n_processes):
            p = Process(target=worker, args=(chunks[i], shoe_shared, print_ids[i], matching_pairs, rankings, counter, queue))
            processes.append(p)
            p.start()

        # Update tqdm progress bar with values in queue and counter
        with tqdm(total=n_prints, file=sys.stdout) as pbar:
            while counter.value < n_prints: #pyright: ignore
                while not queue.empty():
                    message = queue.get()
                    pbar.write(message)
                current_value = counter.value #pyright: ignore
                pbar.update(current_value - pbar.n)
                time.sleep(1)

        # Join processes to ensure they have all terminated
        for p in processes:
            p.join()

    else:
        raise NotImplementedError("Device {device} not implemented")

    return rankings

# ------ Example Usage ------

# print_filters, shoe_filters, matching_pairs = initialise_data("../Data/FID-300")

# rankings = compare(print_filters, shoe_filters, matching_pairs, device="cpu")

# write_csv('results.csv', rankings)
