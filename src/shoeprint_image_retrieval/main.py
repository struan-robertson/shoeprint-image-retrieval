"""Main file for testing shoeprint image retrieval."""

import csv
import os
import time

# CPU
from multiprocessing import Array, Manager, Process, Queue, Value, set_start_method

import numpy as np

# GPU
import torch
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore
from tqdm import tqdm

from Networks.network_pt import Model
from parse_results import cmp_all
from Similarities.ncc import get_similarity
from utils import (
    avg_img_size,
    largest_img,
    largest_img_dir,
    move_small_img,
    smallest_img,
    smallest_img_dir,
)

np.seterr(divide="ignore", invalid="ignore")

# TODO make dataset/dataloader class
# It can store crop values, scale, numbers of images etc

def image_load_worker(
    image_files,
    scale,
    dir,
    indexes,
    image_list,
    id_list,
    counter,
    type="impress",
):
    """Worker for multi_threaded loading of images."""
    images = []
    ids = []

    for image_file in image_files:
        img_path = os.path.join(dir, image_file)

        img = Image.open(img_path)  # Convert to grayscale

        # Resize the image
        new_width = int(img.width * scale)  # 0.1
        new_height = int(img.height * scale)

        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)

        # Note that this is in all dimensions
        # FIXME this obviously affects the minimum resolution logic
        # TODO this should be defined based on the data type 
        # crop_width = int(0.15 * new_width)
        # crop_height = int(0.05 * new_height)

        # Impress
        # crop_width = int(0.2 * new_width)
        # crop_height = int(0.1 * new_height)

        # FID-300
        crop_width = 0
        crop_height = 0
        
        img_array = img_array[
            crop_height : new_height - crop_height, crop_width : new_width - crop_width
        ]
                
        images.append(img_array)

        if type == "impress":
            ids.append(int(image_file.split("_")[0].split(".")[0]))
        elif type == "WVU2019":
            ids.append(int(image_file[:3]))
        elif type == "FID-300":
            ids.append(int(image_file[:-4]))

        # with counter.get_lock():
        #     counter.value += 1

    image_list[indexes[0] : indexes[1]] = images
    id_list[indexes[0] : indexes[1]] = ids

def load_images(image_files, dir, scale, n_processes, dtype):
    """Load images in a directory, with optional scaling using multiprocessing."""
    # List all files in the directory
    # image_files = os.listdir(dir)

    # Sort by name
    image_files.sort()

    chunk_size = len(image_files) // n_processes
    chunk_extra = len(image_files) % n_processes
    chunks = []
    indexes = []
    start = 0
    for i in range(n_processes):
        end = start + chunk_size + (1 if chunk_extra > 1 else 0)
        chunks.append(image_files[start:end])
        indexes.append((start, end))
        start = end

    # Load all images into list
    manager = Manager()
    images = manager.list(range(len(image_files)))
    ids = manager.list(range(len(image_files)))

    counter = Value("i", 0)

    processes = []
    for i in range(n_processes):
        p = Process(
            target=image_load_worker,
            args=(chunks[i], scale, dir, indexes[i], images, ids, counter, dtype),
        )
        processes.append(p)
        p.start()

    # with tqdm(total=len(image_files)) as pbar:
    #     while counter.value < len(image_files): #pyright: ignore
    #         pbar.update(counter.value - pbar.n) #pyright: ignore
    #         pbar.refresh()

    #         time.sleep(1)

    #     pbar.update(counter.value - pbar.n) #pyright: ignore

    for p in processes:
        p.join()

    # images is list of image files _in order of name_
    # important that gallery images are stored in the array correctly
    # ids contains the id of said image

    # Super strange bug where final item in image array is the int 160?
    for i in images:
        try:
            len(i)
        except TypeError:
            images = images[:-1]

    # Return list of images in directory
    return images, ids


def get_all_filters(images, model):
    """
    Calculate the convolutional filters for each image in a list, returning a list of the corresponding filters.
    """
    image_filters = []

    # Calculate conv filter for each image
    for image in tqdm(images):
        filters = model.get_feature_maps(image)
        filters = filters.cpu()
        image_filters.append(filters)

    # Return conv filters
    return image_filters


def cropsize(size, factor):
    """Calculate final size of an image given a cropping factor"""
    crop_amount = int(factor * size)

    return size - crop_amount * 2


def find_best_scale(D_smallest, D_largest, D_min=300, block=6):
    """Calculate ideal scale and output block to use depending on input image resolution"""

    D_max = 800
    end_block = 4
    skip_blocks = [5]
    
    if D_smallest >= D_min and D_largest <= D_max:
        scale = 1
    elif D_smallest < D_min:
        if block > end_block:
            while True:
                block -= 1
                if block not in skip_blocks:
                    break
            D_min /= 2
            scale, block = find_best_scale(D_smallest, D_largest, D_min, block)
        else:
            scale = 1
    elif D_largest > D_max:
        scale = D_max / D_largest
        if D_smallest * scale < D_min:
            if block > end_block:
                while True:
                    block -= 1
                    if block not in skip_blocks or block == end_block:
                        break
            else:
                scale = D_min / D_smallest

    return scale, block


def cluster_images_by_size(dir, n_clusters=5):
    """
    Cluster image by size using K means clustering
    """
    image_sizes = []
    filenames = []

    # Iterate over files in the directory
    for image_file in os.listdir(dir):
        img_path = os.path.join(dir, image_file)

        # Open the image using PIL
        with Image.open(img_path) as img:
            # Get the dimensions of the image
            width, height = img.size
            # image_sizes.append([width, height])
            if width < height:
                image_sizes.append([width])
            else:
                image_sizes.append([height])

            filenames.append(image_file)

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


def minimize_clusters(clusters, print_dir, shoe_files, shoe_dir, tolerance=0.05):
    """
    Minimize the number of clusters given a tolerance
    """

    def is_within_range(num, float_list, tolerance=tolerance):
        for index, float_num in enumerate(float_list):
            if abs(num - float_num) <= tolerance:
                return True, index
        return False, None

    scales = []
    blocks = []
    minimized_groups = []

    largest_shoe = largest_img(shoe_files, shoe_dir)[1]
    smallest_shoe = smallest_img(shoe_files, shoe_dir)[1]

    for cluster in clusters.items():
        cluster = cluster[1]

        smallest_print = smallest_img(cluster, print_dir)[1]
        if min(smallest_print) < min(smallest_shoe):
            smallest = smallest_print
        else:
            smallest = smallest_shoe

        largest_print = largest_img(cluster, print_dir)[1]
        if max(largest_print) > max(largest_shoe):
            largest = largest_print
        else:
            largest = largest_shoe

        # FIXME ugly way to add cropping, guess its just for a couple datasets though
        # width_crop = 0.15
        # height_crop = 0.05

        # impress
        # width_crop = 0.2
        # height_crop = 0.1

        # FID-300
        width_crop = 0
        height_crop = 0

        smallest = (
            cropsize(smallest[0], height_crop)
            if smallest[0] < smallest[1]
            else cropsize(smallest[1], width_crop)
        )
        largest = (
            cropsize(largest[0], height_crop)
            if largest[0] > largest[1]
            else cropsize(largest[1], width_crop)
        )

        scale, block = find_best_scale(smallest, largest)

        in_range, index = is_within_range(scale, scales)

        if in_range and blocks[index] == block:
            minimized_groups[index] += cluster
        else:
            scales.append(scale)
            blocks.append(block)
            minimized_groups.append(cluster)

    return scales, blocks, minimized_groups


def orchestrate(
    data_dir, n_processes, dtype, device="cpu", rotations=[], search_scales=[]
):
    """
    Orchestrate full test
    """

    # Directories containing images
    print_dir = os.path.join(data_dir, "Query")
    shoe_dir = os.path.join(data_dir, "Gallery")

    shoe_files = os.listdir(shoe_dir)
    print_files = os.listdir(print_dir)

    print(
        f"Total of {len(shoe_files)} reference shoeprints and {len(print_files)} shoemarks"
    )

    clustered = cluster_images_by_size(print_dir, 10)
    # clustered = cluster_images_by_size(print_dir, 1)

    scales, blocks, minimized_cluster = minimize_clusters(
        clustered, print_dir, shoe_files, shoe_dir
    )

    ranks = []

    # clustered = group_floats(clustered)

    print(f"{len(minimized_cluster)} groups found")

    # Reverse as this makes it smallest first, if there is going to be a problem I want it to happen at the beginning
    for scale, block, cluster in reversed(list(zip(scales, blocks, minimized_cluster))):
        print(f"Cluster has {len(cluster)} items")

        model = Model("EfficientNetV2_M", block)

        print(f"Best cluster scale found to be {scale} on block {block}")

        # Load images in print directory
        print("Loading ", len(cluster), " prints")
        print_images, print_ids = load_images(
            cluster, print_dir, scale, n_processes, dtype
        )

        # Load images in shoe directory
        print("Loading ", len(shoe_files), " shoes")
        shoe_images, shoe_ids = load_images(
            shoe_files, shoe_dir, scale, n_processes, dtype
        )

        # Note that there is a many to one relationship between query shoemark and gallery shoeprint in WVU2019
        # Get index of corresponding shoeprint from the index of a shoemark
        matching_pairs = []
        if dtype != "FID-300":
            for print_id in print_ids:
                matching_pairs.append(shoe_ids.index(print_id))
        else:
            csv_vals = {}
            with open(os.path.join(data_dir, "label_table.csv"), "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    csv_vals[int(row[0])] = int(row[1])

            for print_id in print_ids:
                matching_pairs.append(csv_vals[print_id] - 1)

        # Calculate conv filters for print images
        print("Calculating convolutional filters for prints")
        print_filters = get_all_filters(print_images, model)

        # Calculate conv filters for shoe images
        print("Calculating convolutional filters for shoes")
        shoe_filters = get_all_filters(shoe_images, model)

        print("Calculating ranks")
        cluster_ranks = compare(
            print_filters,
            shoe_filters,
            matching_pairs,
            device=device,
            n_processes=n_processes,
            rotations=rotations,
            scales=search_scales,
        )

        ranks += cluster_ranks.tolist()
        torch.cuda.empty_cache()

    return ranks


def write_csv(filename, rankings):
    """Write results of test to CSV"""
    with open(f"Results/{filename}", "w", newline="") as file:
        writer = csv.writer(file)
        for rank in rankings:
            writer.writerow([rank])


def get_rank(similarities, matching_pairs, print_id):
    """
    Calculate rank of correctly matching shoemark
    """

    # Sort similarities and then return the indexes in order of the sort
    # np.flip() is required as numpy sorts low -> high
    sorted = np.flip(np.argsort(similarities))  # type: ignore
    # sorted = np.argsort(similarities)

    # Find the rank of the true match within the sorted array
    # matching_pairs[id+1] because the image id is equal to index + 1
    # Add 1 to the result as the resulting index calculated will be 1 less than the true rank
    # rank = np.where(sorted == (matching_pairs[print_id+1] -1))[0][0] +1
    rank = np.where(sorted == (matching_pairs[print_id]))[0][0] + 1

    return rank


def worker(
    print_filters,
    shoe_filters,
    print_ids,
    matching_pairs,
    rankings,
    counter,
    queue,
    rotations,
    scales,
    device="cpu",
):
    """
    Worker for multithreaded running of test
    """

    for i in range(len(shoe_filters)):
        if type(shoe_filters[i]) != np.ndarray:
            shoe_filters[i] = np.frombuffer(
                shoe_filters[i][0].get_obj(), dtype=np.float32
            ).reshape(shoe_filters[i][1])

    scaled_prints_arr = [print_filters]

    # Mirror image to account for right and left feet
    # mirrored_filters = []
    # for print_ in print_filters:
    #     mirrored_filters.append(np.fliplr(print_))

    # scaled_prints_arr.append(mirrored_filters)

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

    rotated_prints_arr = scaled_prints_arr.copy()

    for scaled_print in scaled_prints_arr:
        for r in rotations:
            new_prints = []

            for print_ in scaled_print:
                new_print = []

                for filter in print_:
                    filter = Image.fromarray(filter)
                    filter = filter.rotate(r)

                    new_print.append(filter)

                new_prints.append(np.array(new_print))

            rotated_prints_arr.append(new_prints)

    similarities_all = np.zeros((len(print_filters), len(shoe_filters)))

    # If using GPU
    if device == "gpu":
        for i in range(len(shoe_filters)):
            shoe_filters[i] = cp.array(shoe_filters[i])

        for i in range(len(scaled_prints_arr)):
            for j in range(len(scaled_prints_arr[i])):
                scaled_prints_arr[i][j] = cp.array(scaled_prints_arr[i][j])

    for rotated_prints in rotated_prints_arr:
        for print_id, print_ in zip(range(*print_ids), rotated_prints):
            similarities = []
            for shoe in shoe_filters:
                if device == "cpu":
                    sim = get_similarity(print_, shoe)
                else:
                    sim = cp.asnumpy(get_similarity_cupy(print_, shoe))

                similarities.append(sim)

            print_index = print_id - print_ids[0]
            for shoe_id, similarity in enumerate(similarities):
                if similarity > similarities_all[print_index, shoe_id]:
                    similarities_all[print_index, shoe_id] = similarity

            with counter.get_lock():
                counter.value += 1

    for print_id, similarities in zip(range(*print_ids), similarities_all):
        rank = get_rank(similarities, matching_pairs, print_id)
        rankings[print_id] = rank
        # queue.put(f"Print {print_id+1} true match ranked {rank}, with similarity {similarities[print_id]}")
        queue.put(f"Print {print_id} true match ranked {rank}")


def compare(
    print_filters,
    shoe_filters,
    matching_pairs,
    device="cpu",
    n_processes=32,
    rotations=[-15, -9, -3, 3, 9, 15, 180],
    scales=[1.02, 1.04, 1.08],
):
    """
    Compare each print to every shoe, generating a probability score.
    The scores are used to calculate the ranking of the true match between a print and shoe in comparison to the other shoes.
    All rankings are then returned.
    """

    # Thread spawning to allow GPU access
    if device == "gpu":
        # GPU implementation never worked
        print("GPU implementation was never finished, please use CPU")
        exit()
        set_start_method("spawn", force=True)

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
        print_ids.append((start, end))  # range of print ids contained in the chunk
        start = end

    # Shared memory variables
    counter = Value("i", 0)  # Counter to contain number of prints processed
    queue = Queue()  # FIFO queue so that processes can write tqdm messages

    n_prints = len(print_filters)
    rankings = Array("i", n_prints)  # Rankings of each print

    # Store shoe scan filters in shared memory as these are required for every print
    shoe_shared = []
    for i in range(len(shoe_filters)):
        shape = shoe_filters[i].shape

        # Copy shoe_filters[i] into shared memory array
        shared = Array("f", shoe_filters[i].size)
        np_shared = np.frombuffer(shared.get_obj(), dtype=np.float32).reshape(shape)  # pyright: ignore
        np.copyto(np_shared, shoe_filters[i])

        shoe_shared.append((shared, shape))

    # Debug
    # worker(chunks[0], shoe_shared, print_ids[0], matching_pairs, rankings, counter, queue, rotations=rotations, scales=scales, device=device)

    # Spawn each process
    processes = []
    for i in range(n_processes):
        p = Process(
            target=worker,
            args=(
                chunks[i],
                shoe_shared,
                print_ids[i],
                matching_pairs,
                rankings,
                counter,
                queue,
                rotations,
                scales,
                device,
            ),
        )
        processes.append(p)
        p.start()

    # Update tqdm progress bar with values in queue and counter
    work = (len(rotations) + 1) * (len(scales) + 1) * n_prints
    with tqdm(total=work) as pbar:
        while counter.value < work:  # pyright: ignore
            while not queue.empty():
                message = queue.get()
                pbar.write(message)

            pbar.update(counter.value - pbar.n)  # pyright: ignore
            pbar.refresh()

            time.sleep(1)

        pbar.update(counter.value - pbar.n)  # pyright: ignore
        while not queue.empty():
            message = queue.get()
            pbar.write(message)

    # Join processes to ensure they have all terminated
    for p in processes:
        p.join()

    # No need to return a shared array
    rankings = np.frombuffer(rankings.get_obj(), dtype=np.int32)  # pyright: ignore

    return rankings


# ------ Example Usage ------

# from main import *
#
# rankings = orchestrate("../Datasets/Impress/", 60, "impress", rotations=[-15, -9, -3, 3, 9, 15, 180], search_scales=[1.02, 1.04, 1.08], device="cpu")
# cmp_all(rankings, total_references=300, total_prints=2292)
