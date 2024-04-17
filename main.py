#!/usr/bin/env python3
import os
import shutil
from PIL import Image
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import csv
from tqdm import tqdm
import time
import sys

from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans

# GPU
import torch

# CPU
from multiprocessing import Array, Process, Value, Queue, Manager
from ctypes import c_wchar_p

import ipdb

# from line_profiler import profile

# ------ Networks ------

# Uses PyTorch
# from Networks.vgg19 import get_filters

# Uses Tensorflow
# from Networks.vgg19_tf import get_filters

from Networks.network_pt import Model

# ------ Similarity Measures ------

from Similarities.ncc import get_similarity, get_similarity_gpu
# from Similarities.orb import get_similarity
# from Similarities.dim import get_similarity

def avg_img_size(dir):
    image_files = os.listdir(dir)
    total_width = 0
    total_height = 0
    for image_file in image_files:
        img_path = os.path.join(dir, image_file)
        with Image.open(img_path) as img:
            width, height = img.size
            total_width += width
            total_height += height
    avg_width = total_width // len(image_files)
    avg_height = total_height // len(image_files)
    return avg_width, avg_height

def smallest_img(dir):
    image_files = os.listdir(dir)

    smallest_img_size = float('inf')
    smallest_img_name = None
    smallest_img_dims = (0,0)

    for image_file in image_files:
        img_path = os.path.join(dir, image_file)

        with Image.open(img_path) as img:
            width, height = img.size
            image_size = width * height

            if image_size < smallest_img_size:
                smallest_img_name = image_file
                smallest_img_size = image_size
                smallest_img_dims = (width, height)

    return smallest_img_name, smallest_img_dims

def biggest_img(dir):
    image_files = os.listdir(dir)

    biggest_img_size = float(0)
    biggest_img_name = None
    biggest_img_dims = (0,0)

    for image_file in image_files:
        img_path = os.path.join(dir, image_file)

        with Image.open(img_path) as img:
            width, height = img.size
            image_size = width * height

            if image_size > biggest_img_size:
                biggest_img_name = image_file
                biggest_img_size = image_size
                biggest_img_dims = (width, height)

    return biggest_img_name, biggest_img_dims

def move_small_img(dir, smallest, destdir):

    image_files = os.listdir(dir)

    for image_file in image_files:
        img_path = os.path.join(dir, image_file)

        with Image.open(img_path) as img:
            width, height = img.size

            if width < smallest[0] or height < smallest[1]:
                shutil.move(img_path, destdir)
                print(f"Moved {image_file}")


def image_load_worker(image_files, scale, dir, indexes, image_list, id_list, counter, minimum_img):
    # Load all images into list
    images = []
    ids = []
    for image_file in image_files:
        img_path = os.path.join(dir, image_file)
        img = Image.open(img_path)  # Convert to grayscale

        # Resize the image
        new_width = int(img.width * scale) # 0.1
        new_height = int(img.height * scale)

        if new_height * new_width < minimum_img[0] * minimum_img[1]:
            minimum_img[0:2] = [new_width, new_height]

        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)

        images.append(img_array)
        ids.append(int(image_file.split('_')[0].split('.')[0]))

        with counter.get_lock():
            counter.value += 1


    image_list[indexes[0]:indexes[1]] = images
    id_list[indexes[0]:indexes[1]] = ids


def cluster_images_by_size(directory, n_clusters=5):
    image_sizes = []
    filenames = []

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Check if the file is an image
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Open the image using PIL
            with Image.open(filepath) as img:
                # Get the dimensions of the image
                width, height = img.size
                image_sizes.append([width, height])
                filenames.append(filename)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(image_sizes)

    # Get the cluster labels for each image
    labels = kmeans.labels_

    # Group images by cluster label
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filenames[i])

    return clusters



def load_images(dir, scale, n_processes):
    """
    Load images into an array, sorted by name.
    As the image name corresponds to its ID, the index of the returned array corresponds to the image ID - 1.
    """

    # List all files in the directory
    image_files = os.listdir(dir)
    # Sort by name
    image_files.sort()

    chunk_size = len(image_files) // n_processes
    chunk_extra = len(image_files) % n_processes
    chunks = []
    indexes = []
    start = 0
    for i in range(n_processes):
        end = start + chunk_size + (1 if 1 < chunk_extra else 0)
        chunks.append(image_files[start:end])
        indexes.append((start, end))
        start = end

    # Load all images into list
    manager = Manager()
    images = manager.list(range(len(image_files)))
    ids = manager.list(range(len(image_files)))

    counter = Value('i', 0)
    minimum_img = Array('i', [999999, 999999])

    processes = []
    for i in range(n_processes):
        p = Process(target=image_load_worker, args=(chunks[i], scale, dir, indexes[i], images, ids, counter, minimum_img))
        processes.append(p)
        p.start()

    with tqdm(total=len(image_files)) as pbar:
        while counter.value < len(image_files):
            pbar.update(counter.value - pbar.n)
            pbar.refresh()

            time.sleep(1)

        pbar.update(counter.value - pbar.n)

    for p in processes:
        p.join()

    print(f"Smallest file has a size of {minimum_img[0]}x{minimum_img[1]}")

    # images is list of image files _in order of name_
    # important that gallery images are stored in the array correctly
    # ids contains the id of said image

    # Return list of images in directory
    return images, ids, minimum_img

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

# TODO save numpy arrays to NPZ so they dont have to be calculated if restarting python
def initialise_data(data_dir, scale, n_processes):
    """
    Load all required state for testing.
    Loads:
     - Convolutional filters for print and shoe images
     - A dictionary containing the matching pairs of prints and shoes
    """

    # Directories containing images
    print_dir = os.path.join(data_dir, "Query")
    shoe_dir = os.path.join(data_dir, "Gallery")

    # Load images in print directory
    print_images, print_ids, min_print_img = load_images(print_dir, scale, n_processes)
    print("Loaded ", len(print_images), " prints")

    # Load images in shoe directory
    shoe_images, shoe_ids, _ = load_images(shoe_dir, scale, n_processes)
    print("Loaded ", len(shoe_images), " shoes")


    # Note that there is a many to one relationship between query shoemark and gallery shoeprint in WVU2019
    # Get index of corresponding shoeprint from the index of a shoemark
    matching_pairs = []
    for print_id in print_ids:
        matching_pairs.append(shoe_ids.index(print_id))

    # model = Model("EfficientNet_B5", 7)
    # model = Model("VGG19", 36)
    # model = Model()
    # model = Model("EfficientNetV2_M", 4)
    # model = Model("EfficientNetV2_S", 7)
    # model = Model("EfficientNetV2_M", 6)
    model = Model("EfficientNetV2_M", 4)
    # model = Model("EfficientNetV2_M", 1)
    # model = Model("VGG16", 23)

    # Calculate conv filters for print images
    print("Calculating convolutional filters for prints")
    print_filters = get_all_filters(print_images, model)

    # Calculate conv filters for shoe images
    print("Calculating convolutional filters for shoes")
    shoe_filters = get_all_filters(shoe_images, model)

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
    # rank = np.where(sorted == (matching_pairs[print_id+1] -1))[0][0] +1
    rank = np.where(sorted == (matching_pairs[print_id]))[0][0] + 1

    return rank

def worker(print_filters, shoe_filters, print_ids, matching_pairs, rankings, counter, queue, rotations, scales):

    for i in range(len(shoe_filters)):
        shoe_filters[i] = np.frombuffer(shoe_filters[i][0].get_obj(), dtype=np.float32).reshape(shoe_filters[i][1])

    scaled_prints_arr = [print_filters]

    # Mirror image to account for right and left feet
    # mirrored_filters = []
    # for print_ in print_filters:
    #     mirrored_filters.append(np.fliplr(print_))

    # scaled_prints_arr.append(mirrored_filters)

    for r in rotations:
        new_prints = []

        for print_ in print_filters:
            new_prints.append(ndimage.rotate(print_, r, axes=(1,2)))

        scaled_prints_arr.append(new_prints)

    for s in scales:

        new_prints = []

        for print_ in print_filters:
            new_print = []

            for filter in print_:
                filter = Image.fromarray(filter)
                filter = filter.resize((int(filter.width * s), int(filter.height * s)))

                new_print.append(filter)

            new_prints.append(np.array(new_print))

        scaled_prints_arr.append(new_prints)

    similarities_all = np.zeros((len(print_filters), len(shoe_filters)))

    for rotated_prints in scaled_prints_arr:

        for print_id, print_ in zip(range(*print_ids), rotated_prints):
            # TODO remove this to simulate IRL
            # if rankings[print_id] < 10 and rankings[print_id] != 0:
            #     with counter.get_lock():
            #         counter.value += 1
            #     continue

            similarities = []
            for shoe in shoe_filters:
                sim = get_similarity(print_, shoe)

                similarities.append(sim)

            # rank = get_rank(similarities, matching_pairs, print_id)

            # if rank < rankings[print_id] or rankings[print_id] == 0:
            #     extra = ""
            #     if rankings[print_id] != 0:
            #         extra = f", increased from previous rank {rankings[print_id]}"

            #     rankings[print_id] = rank

            #     queue.put(f"Print {print_id+1} true match ranked {rank}{extra}")

            print_index = print_id - print_ids[0]
            for shoe_id, similarity in enumerate(similarities):
                if similarity > similarities_all[print_index, shoe_id]:
                    similarities_all[print_index, shoe_id] = similarity

            with counter.get_lock():
                counter.value += 1

        # sorted =  np.flip(np.argsort(max_similarities))

        # extra = ""
        # if max_rank != 1:
        #     extra = f" (incorrectly matched shoe {sorted[0] + 1}, correct is {matching_pairs[print_id+1]})"

    for print_id, similarities in zip(range(*print_ids), similarities_all):

        rank = get_rank(similarities, matching_pairs, print_id)
        rankings[print_id] = rank
        # queue.put(f"Print {print_id+1} true match ranked {rank}, with similarity {similarities[print_id]}")
        queue.put(f"Print {print_id} true match ranked {rank}")




def compare(print_filters, shoe_filters, matching_pairs, device="cpu", n_processes=32, rotations=[-9, -3, 3, 9], scales=[0.96, 1.04], gpu_fix=True):
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

                new_sims = get_similarity_gpu(print_, shoe.unsqueeze(0), gpu_fix=gpu_fix).numpy()
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
        # worker(chunks[0], shoe_shared, print_ids[0], matching_pairs, rankings, counter, queue, rotations=rotations, scales=scales)

        # Spawn each process
        processes = []
        for i in range(n_processes):
            p = Process(target=worker, args=(chunks[i], shoe_shared, print_ids[i], matching_pairs, rankings, counter, queue, rotations, scales))
            processes.append(p)
            p.start()

        # Update tqdm progress bar with values in queue and counter
        # work = (len(rotations)+1) * (len(scales)+1) * n_prints #* 2
        work = (len(rotations) + len(scales) + 1) * n_prints #* 2
        with tqdm(total=work) as pbar:
            while counter.value < work: #pyright: ignore
                while not queue.empty():
                    message = queue.get()
                    pbar.write(message)

                pbar.update(counter.value - pbar.n ) #pyright: ignore
                pbar.refresh()

                time.sleep(1)

            pbar.update(counter.value - pbar.n) #pyright: ignore
            while not queue.empty():
                message = queue.get()
                pbar.write(message)


        # Join processes to ensure they have all terminated
        for p in processes:
            p.join()

        # No need to return a shared array
        rankings = np.frombuffer(rankings.get_obj(), dtype=np.int32) #pyright: ignore

    else:
        raise NotImplementedError("Device {device} not implemented")

    return rankings

# ------ Example Usage ------

# print_filters, shoe_filters, matching_pairs = initialise_data("../Data/FID-300")

# rankings = compare(print_filters, shoe_filters, matching_pairs, device="cpu")

# write_csv('results.csv', rankings)
