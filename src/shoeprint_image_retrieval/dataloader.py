"""Handle loading, processing and manipulation of datasets."""

import csv
import os
from multiprocessing import Manager, Process, Value
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


class Dataloader:
    """Initialise, pre-process and provide access to datasets."""

    def __init__(
        self,
        dataset_dir,
        crop,
        n_processes,
        dataset_type,
        n_clusters=10,
    ):
        """Load and pre-process dataset."""
        self.dataset_dir = Path(dataset_dir)
        self.shoeprint_dir = self.dataset_dir / "Gallery"
        self.shoemark_dir = self.dataset_dir / "Query"

        self.shoeprint_files = os.listdir(self.shoeprint_dir)
        self.shoemark_files = os.listdir(self.shoemark_dir)

        print(
            "The dataset contains: \n",
            f"    {len(self.shoeprint_files)} reference shoeprints\n",
            f"    {len(self.shoemark_files)} shoemarks",
        )

        self.crop = crop
        self.n_processes = n_processes
        self.dataset_type = dataset_type

        clustered = self._cluster_images_by_size(self.shoemark_dir, n_clusters)

        self.scales, self.blocks, self.clusters = self._minimise_clusters(
            clustered,
            0.05,  # TODO make tolerance a setting
        )

        self.num_clusters = len(self.clusters)

        self._current_cluster = 0

        # TODO take this outside of the class
        print(f"{self.num_clusters} clusters of image sizes found.")

    def _cluster_images_by_size(self, image_dir, n_clusters):
        """Cluster images by size using K means clustering.

        Args:
        ----
            image_dir (path): Directory containing the image files to cluster.
            n_clusters (int): Number of clusters to group images into.

        Returns:
        -------
            Dictionary: Keys are cluster labels and values are images clustered
        under that label.

        """
        image_sizes = []
        filenames = []

        # Iterate over files in the directory
        for image_file in os.listdir(image_dir):
            image_path = image_dir / image_file

            with Image.open(image_path) as image:
                width, height = image.size

                # We are interested in grouping by the smallest image dimension
                if width < height:
                    image_sizes.append([width])
                else:
                    image_sizes.append([height])

                filenames.append(image_file)

        # Cluster
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

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_cluster >= self.num_clusters:
            raise StopIteration

        # Going to return tuple of shoemark_images_cluster, shoeprint_images,
        # matching_pairs, block

        breakpoint()

        shoemark_images, shoemark_ids = self._load_images(
            self.clusters[self._current_cluster],
            self.shoemark_dir,
            self.scales[self._current_cluster],
        )

        shoeprint_images, shoeprint_ids = self._load_images(
            self.shoeprint_files,
            self.shoeprint_dir,
            self.scales[self._current_cluster],
        )

        # Get matching pairs
        # Note that there is a many to one relationship between query shoemark
        # and gallery shoeprint in WVU2019
        # Get index of corresponding shoeprint from the index of a shoemark
        matching_pairs = []
        if self.dataset_type != "FID-300":
            matching_pairs = [
                shoeprint_ids.index(shoemark_id) for shoemark_id in shoemark_ids
            ]
        else:
            csv_vals = {}
            with Path.open(self.dataset_dir / "label_table.csv") as file:
                reader = csv.reader(file)
                for row in reader:
                    csv_vals[int(row[0])] = int(row[1])

            matching_pairs = [csv_vals[shoemark_id] - 1 for shoemark_id in shoemark_ids]

        block = self.blocks[self._current_cluster]
            
        self._current_cluster += 1

        return shoemark_images, shoeprint_images, matching_pairs, block

    def _load_images(self, image_files, image_directory, scale):
        """Load images in a directory, using multiprocessing for post-processing."""
        # TODO add more descriptive docstring
        # Sort by name
        image_files.sort()

        chunk_size = len(image_files) // self.n_processes
        chunk_extra = len(image_files) % self.n_processes
        chunks = []
        indexes = []
        start = 0
        for _ in range(self.n_processes):
            end = start + chunk_size + (1 if chunk_extra > 1 else 0)
            chunks.append(image_files[start:end])
            indexes.append((start, end))
            start = end

        # Load all images into list
        manager = Manager()
        images = manager.list(range(len(image_files)))
        ids = manager.list(range(len(image_files)))

        processes = []
        for i in range(self.n_processes):
            p = Process(
                target=Dataloader._image_load_worker,
                args=(
                    chunks[i],
                    scale,
                    image_directory,
                    indexes[i],
                    images,
                    ids,
                    self.dataset_type,
                    self.crop,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # images is list of image files _in order of name_
        # important that gallery images are stored in the array correctly
        # ids contains the id of said image

        # Fix super strange bug where final item in image array is the int 160?
        for i in images:
            if type(i) is not np.ndarray:
                images = images[:-1]

        return images, ids

    @staticmethod
    def _image_load_worker(
        # TODO some of these should be passed in a settings dictionary
        # TODO add descriptive docstring
        # Image list is the shared memory list
        image_files,
        scale,
        image_directory,
        indexes,
        image_list,
        id_list,
        dataset_type,
        crop,
    ):
        images = []
        ids = []

        for image_file in image_files:
            image_path = image_directory / image_file

            image = Image.open(image_path)

            # Resize the image
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)

            image_resized = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

            # Crop image
            image_array = np.array(image_resized)

            image_array = image_array[
                crop[0] : new_height - crop[0],
                crop[1] : new_width - crop[1],
            ]

            images.append(image_array)

            # Parse image ID from filename
            if dataset_type == "Impress":
                ids.append(int(image_file.split("_")[0].split(".")[0]))
            elif dataset_type == "WVU2019":
                ids.append(int(image_file[:3]))
            elif dataset_type == "FID-300":
                ids.append(int(image_file[:-4]))
            else:
                error = f"Dataset type {dataset_type} not implemented."
                raise NotImplementedError(error)

        image_list[indexes[0] : indexes[1]] = images
        id_list[indexes[0] : indexes[1]] = ids

    def _minimise_clusters(self, clusters, tolerance):
        """Minimise the number of clusters given a tolerance.

        This stops a dataset of identically sized images from producing >1 clusters.

        Args:
        ----
            clusters (dictionary): Clustered images (from `_cluster_images_by_size').
            tolerance (float): TODO figure out exactly what this does.

        Returns:
        -------
            TODO what does it return?

        """

        def is_within_range(num, float_list):
            for index, float_num in enumerate(float_list):
                if abs(num - float_num) <= tolerance:
                    return True, index
            return False, None

        scales = []
        blocks = []
        minimised_groups = []

        # Shoeprints are not clustered and so this only need calculated once
        largest_shoeprint, smallest_shoeprint = self._image_extremes(
            self.shoeprint_files,
            self.shoeprint_dir,
        )

        for cluster in clusters.items():
            cluster_files = cluster[1]

            largest_shoemark, smallest_shoemark = self._image_extremes(
                cluster_files,
                self.shoemark_dir,
            )

            if min(smallest_shoemark[1]) < min(smallest_shoeprint[1]):
                smallest_dims = smallest_shoemark[1]
            else:
                smallest_dims = smallest_shoeprint[1]

            if max(largest_shoemark[1]) > max(largest_shoeprint[1]):
                largest_dims = largest_shoemark[1]
            else:
                largest_dims = largest_shoeprint[1]

            smallest_dim = (
                self._cropsize(smallest_dims[0], self.crop[0])
                if smallest_dims[0] < smallest_dims[1]
                else self._cropsize(smallest_dims[1], self.crop[1])
            )
            largest_dim = (
                self._cropsize(largest_dims[0], self.crop[0])
                if largest_dims[0] > largest_dims[1]
                else self._cropsize(largest_dims[1], self.crop[1])
            )

            # TODO get minimum_dim from settings
            scale, block = self._find_best_scale(
                smallest_dim,
                largest_dim,
                minimum_dim=300,
            )

            # TODO figure out exactly what this does and document it
            in_range, index = is_within_range(scale, scales)

            if in_range and blocks[index] == block:
                minimised_groups[index] += cluster_files
            else:
                scales.append(scale)
                blocks.append(block)
                minimised_groups.append(cluster_files)

        return scales, blocks, minimised_groups

    def _find_best_scale(self, smallest_dim, largest_dim, minimum_dim, block=6):
        """Calculate ideal input image scale and network output block.

        Note that this is a recursive algorithm, described in Algorithm 1 of the paper.

        Args:
        ----
            smallest_dim (int): Smallest dimension of an image within a set.
            largest_dim (int): Largest dimension of an image within a set.
            minimum_dim (int): Dimension at which generated feature maps will become
            overly compressed.
            block (int): Network block to calculate on.

        Returns:
        -------
            (float, int): Ideal image scale and network block to use.

        """
        # TODO make settings
        maximumum_dim = 800
        end_block = 4
        skip_blocks = [5]

        if smallest_dim >= minimum_dim and largest_dim <= maximumum_dim:
            scale = 1
        elif smallest_dim < minimum_dim:
            if block > end_block:
                while True:
                    block -= 1
                    if block not in skip_blocks:
                        break
                minimum_dim /= 2
                scale, block = self._find_best_scale(
                    smallest_dim,
                    largest_dim,
                    minimum_dim,
                    block,
                )
            else:
                scale = 1
        elif largest_dim > maximumum_dim:
            scale = maximumum_dim / largest_dim
            if smallest_dim * scale < minimum_dim:
                if block > end_block:
                    while True:
                        block -= 1
                        if block not in skip_blocks or block == end_block:
                            break
                else:
                    scale = minimum_dim / smallest_dim

        return scale, block

    def _cropsize(self, size, factor):
        """Calculate the cropped size of an image dimension given a cropping factor."""
        crop_amount = int(factor * size)

        return size - crop_amount * 2

    def _image_extremes(
        self,
        image_files,
        image_directory,
    ):
        """Return the largest image given a list of files in a directory.

        Args:
        ----
            image_files (list): Names of image files.
            image_directory (Path): Path to the directory image_files are contained in.

        Returns:
        -------
            (largest_image_name, (width, height)), (smallest_image_name (width,height))

        """
        largest_image_size = float(0)
        largest_image_name = None
        largest_image_dims = (0, 0)

        smallest_image_size = float("inf")
        smallest_image_name = None
        smallest_image_dims = (0, 0)

        for image_file in image_files:
            image_path = image_directory / image_file

            # TODO does it make sense to measure area rather than largest dimension?
            with Image.open(image_path) as image:
                width, height = image.size
                image_size = width * height

                if image_size > largest_image_size:
                    largest_image_name = image_file
                    largest_image_size = image_size
                    largest_image_dims = (width, height)

                elif image_size < smallest_image_size:
                    smallest_image_name = image_file
                    smallest_image_size = image_size
                    smallest_image_dims = (width, height)

        return (largest_image_name, largest_image_dims), (
            smallest_image_name,
            smallest_image_dims,
        )
