"""Handle loading, processing and manipulation of datasets."""

from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.managers import ListProxy

    from .config import Config
    from .customtypes import DatasetTypeType

import csv
import os
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.cluster import KMeans


class Dataloader:
    """Initialise, pre-process and provide access to datasets."""

    def __init__(self, config: Config) -> None:
        """Load and pre-process dataset.

        Args:
            config: Config for system.

        """
        self.config = config

        self.dataset_dir = Path(config["dataset"]["dir"])
        self.shoeprint_dir = self.dataset_dir / "Gallery"
        self.shoemark_dir = self.dataset_dir / "Query"

        self.shoeprint_files = os.listdir(self.shoeprint_dir)
        self.shoemark_files = os.listdir(self.shoemark_dir)

        print(
            "The dataset contains: \n",
            f"    {len(self.shoeprint_files)} reference shoeprints\n",
            f"    {len(self.shoemark_files)} shoemarks",
        )

        clustered: dict[int, list[Any]] = self._cluster_images_by_size(
            self.shoemark_dir,
            config["dataset"]["n_clusters"],
        )

        self.scales, self.blocks, self.clusters = self._minimise_clusters(
            clustered,
        )

        self.num_clusters = len(self.clusters)

        self._current_cluster = 0

    def __iter__(self):
        """Make class an iterator."""
        return self

    def __next__(self) -> tuple[list[NDArray[Any]], list[NDArray[Any]], list[int], int]:
        """Calculate what to return each iteration.

        Returns:
            Cluster of shoemark images
            All shoeprint images
            Matching pair IDs
            Network block

        """
        if self._current_cluster >= self.num_clusters:
            raise StopIteration

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
        matching_pairs: list[int] = []
        if self.config["dataset"]["type"] != "FID-300":
            matching_pairs = [shoeprint_ids.index(shoemark_id) for shoemark_id in shoemark_ids]
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

    def _load_images(
        self,
        image_files: list[str],
        image_directory: Path,
        scale: float,
    ) -> tuple[list[NDArray[Any]], list[int]]:
        """Load images in a directory, using multiprocessing for post-processing.

        Args:
            image_files: List of image filenames to load.
            image_directory: Directory which contains image_files.
            scale: The scale to apply to each image.

        Returns:
            list of loaded images, list of image IDs.

        """
        # Sort by name
        image_files.sort()

        n_processes = self.config["dataset"]["n_processes"]

        chunk_size = len(image_files) // n_processes
        chunk_extra = len(image_files) % n_processes
        chunks: list[list[str]] = []
        indexes: list[tuple[int, int]] = []
        start = 0
        for _ in range(n_processes):
            end = start + chunk_size + (1 if chunk_extra > 1 else 0)
            chunks.append(image_files[start:end])
            indexes.append((start, end))
            start = end

        # Load all images into list
        manager = Manager()
        shared_image_list = manager.list([np.empty(0)] * len(image_files))
        shared_ids_list = manager.list([0] * len(image_files))

        processes: list[Process] = []
        for i in range(n_processes):
            p = Process(
                target=Dataloader._image_load_worker,
                args=(
                    chunks[i],
                    scale,
                    image_directory,
                    indexes[i],
                    shared_image_list,
                    shared_ids_list,
                    self.config["dataset"]["type"],
                    self.config["dataset"]["crop"],
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
        for i in shared_image_list:
            if type(i) is not np.ndarray:
                shared_image_list = shared_image_list[:-1]

        return list(shared_image_list), list(shared_ids_list)

    @staticmethod
    def _image_load_worker(  # noqa: PLR0913
        image_files: list[str],
        scale: float,
        image_directory: Path,
        indexes: tuple[int, int],
        shared_image_list: ListProxy[NDArray[Any]],
        shared_id_list: ListProxy[int],
        dataset_type: DatasetTypeType,
        crop: tuple[float, float],
    ) -> None:
        """Static method to load images from different processes.

        Args:
            image_files: List of image file names to load.
            scale: Scale to apply to each image file.
            image_directory: Directory containing image_files.
            indexes: Indexes of shared_image_list to load image files into.
            shared_image_list: List shared by all processes in which to load image files into.
            shared_id_list: List shared by all processes in which to load image ID into.
            dataset_type: Type of dataset, used to determine image ID.
            crop: Cropping to apply to each image.

        """
        images: list[NDArray[Any]] = []
        ids: list[int] = []

        for image_file in image_files:
            image_path = image_directory / image_file

            image = Image.open(image_path)

            # Crop the image
            crop_height = floor(image.height * crop[0])
            crop_width = floor(image.width * crop[1])

            crop_box = (
                crop_width,
                crop_height,
                image.width - crop_width,
                image.height - crop_height,
            )

            image = image.crop(crop_box)

            # Resize the image
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)

            image_resized = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

            # Crop image
            image_array = np.array(image_resized)

            images.append(image_array)

            # Parse image ID from filename
            if dataset_type == "Impress":
                ids.append(int(image_file.split("_")[0].split(".")[0]))
            elif dataset_type == "WVU2019":
                ids.append(int(image_file[:3]))
            elif dataset_type == "FID-300":
                ids.append(int(image_file[:-4]))

        shared_image_list[indexes[0] : indexes[1]] = images
        shared_id_list[indexes[0] : indexes[1]] = ids

    def _cluster_images_by_size(self, image_dir: Path, n_clusters: int) -> dict[int, list[str]]:
        """Cluster images by size using K means clustering.

        Args:
            image_dir: Directory containing the image files to cluster.
            n_clusters: Number of clusters to group images into.

        Returns:
            Keys are cluster labels and values are images clustered under that label.

        """
        image_sizes: list[list[int]] = []
        filenames: list[str] = []

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
        _ = kmeans.fit(image_sizes)  # pyright: ignore[reportUnknownMemberType]

        # Get the cluster labels for each image
        labels: NDArray[np.int_] = cast(NDArray[np.int_], kmeans.labels_)

        # Group images by cluster label
        clusters: dict[int, list[str]] = {}
        for i, label in enumerate(labels):
            label: int
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(filenames[i])

        return clusters

    def _minimise_clusters(self, clusters: dict[int, list[str]]):
        """Minimise the number of clusters given a tolerance.

        This stops a dataset of identically sized images from producing >1 clusters.

        Args:
            clusters: Clustered images (from `_cluster_images_by_size').
            tolerance: Tolerance between scales, if less than which to merge clusters.

        Returns:
            lists of scales, blocks and minimised groups

        """

        def is_within_range(scale: float, scale_list: list[float]):
            """Checks if the scale is within the tolerance of other selected scales."""
            for index, scale_num in enumerate(scale_list):
                if abs(scale - scale_num) <= self.config["dataset"]["cluster_minimise_tolerance"]:
                    return True, index
            return False, 2**31 - 1  # Max 32 bit int

        scales: list[float] = []
        blocks: list[int] = []
        minimised_groups: list[list[str]] = []

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

            if smallest_shoemark[1] < smallest_shoeprint[1]:
                smallest_dim = smallest_shoemark[1]
            else:
                smallest_dim = smallest_shoeprint[1]

            if largest_shoemark[1] > largest_shoeprint[1]:
                largest_dim = largest_shoemark[1]
            else:
                largest_dim = largest_shoeprint[1]

            scale, block = self._find_best_scale(
                smallest_dim,
                largest_dim,
                minimum_dim=self.config["model"]["minimum_dim"],
                block=self.config["model"]["start_block"],
            )

            in_range, index = is_within_range(scale, scales)
            if in_range and blocks[index] == block:
                minimised_groups[index] += cluster_files
            else:
                scales.append(scale)
                blocks.append(block)
                minimised_groups.append(cluster_files)

        return scales, blocks, minimised_groups

    def _find_best_scale(
        self,
        smallest_dim: int,
        largest_dim: int,
        minimum_dim: int,
        block: int,
    ) -> tuple[float, int]:
        """Calculate ideal input image scale and network output block.

        Note that this is a recursive algorithm, described in Algorithm 1 of the paper.

        Args:
            smallest_dim: Smallest dimension of an image within a set.
            largest_dim: Largest dimension of an image within a set.
            minimum_dim: Dimension at which generated feature maps will become
            overly compressed.
            block: Network block to calculate on.

        Returns:
            Ideal image scale and network block to use.

        """
        maximumum_dim = self.config["model"]["maximum_dim"]
        end_block = self.config["model"]["end_block"]
        skip_blocks = self.config["model"]["skip_blocks"]
        scale = 1

        if smallest_dim < minimum_dim:
            if block > end_block:
                while True:
                    block -= 1
                    if block not in skip_blocks:
                        break
                minimum_dim = int(minimum_dim / 2)
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

    def _image_extremes(
        self,
        image_files: list[str],
        image_directory: Path,
    ) -> tuple[tuple[str, int], tuple[str, int]]:
        """Return the largest image given a list of files in a directory.

        Args:
            image_files: Names of image files.
            image_directory: Path to the directory image_files are contained in.

        Returns:
            (largest_image_name, (width, height)), (smallest_image_name (width,height))

        """
        largest_image_dim = 0
        largest_image_name = ""

        smallest_image_dim = 2**31 - 1  # Max 32 bit int
        smallest_image_name = ""

        for image_file in image_files:
            image_path = image_directory / image_file

            with Image.open(image_path) as image:
                height, width = image.size

                # Take into account cropping
                crop_height = floor(height * self.config["dataset"]["crop"][0] * 2)
                crop_width = floor(width * self.config["dataset"]["crop"][1] * 2)

                height -= crop_height
                width -= crop_width

                largest_dim = max(width, height)
                smallest_dim = min(width, height)

                if largest_dim > largest_image_dim:
                    largest_image_name = image_file
                    largest_image_dim = largest_dim

                elif smallest_dim < smallest_image_dim:
                    smallest_image_name = image_file
                    smallest_image_dim = smallest_dim

        return (largest_image_name, largest_image_dim), (
            smallest_image_name,
            smallest_image_dim,
        )
