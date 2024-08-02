"""Handle loading, processing and manipulation of datasets."""

import os
from pathlib import Path

from PIL import Image
from sklearn.cluster import KMeans


class Dataloader:
    """Initialise, pre-process and provide access to datasets."""

    def __init__(
        self,
        dataset_dir,
        dataset_type,
        n_processes,
        crop,
        n_clusters=10,
    ):
        """Load and pre-process dataset."""
        self.shoeprint_dir = dataset_dir / "Gallery"
        self.shoemark_dir = dataset_dir / "Query"

        self.shoeprint_files = os.listdir(self.shoeprint_dir)
        self.shoemark_files = os.listdir(self.shoemark_dir)

        print(
            "The dataset contains: \n",
            f"    {len(self.shoeprint_files)} reference shoeprints\n",
            f"    {len(self.shoemark_files)} shoemarks",
        )

        self.crop = crop

        clustered = self._cluster_images_by_size(self.shoemark_dir, n_clusters)

        scales, blocks, minimised_clusters = self._minimise_clusters(
            clustered,
            self.shoemark_dir,
            self.shoeprint_files,
            self.shoeprint_dir,
        )

        print(f"{len(minimised_clusters)} clusters of image sizes found.")

        ranks = []

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
                minimised_groups[index] += cluster
            else:
                scales.append(scale)
                blocks.append(block)
                minimised_groups.append(cluster)

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
                    smallest_dim, largest_dim, minimum_dim, block
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
        """Return the largest image given a list of files in a directory."""
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
