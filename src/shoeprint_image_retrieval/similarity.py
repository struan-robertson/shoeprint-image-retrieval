"""Module for comparing the similarity between shoemark and shoeprint feature maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized, SynchronizedArray

    from .config import Config

import time
from multiprocessing import Array, Process, Queue, Value
from typing import Any, Literal, cast

import numpy as np
from PIL import Image
from scipy.signal import convolve  # pyright: ignore[reportUnknownVariableType]
from tqdm import tqdm

from .customtypes import FeatureMapsArrayType, ImageArrayType

_ = np.seterr(divide="ignore", invalid="ignore")


def normxcorr(
    # Would be nice to use jaxtyping here but I think my numpy version is too low
    template: ImageArrayType,
    image: ImageArrayType,
    mode: Literal["full", "valid", "same"] = "same",
) -> ImageArrayType:
    """Calculate normalised cross correlation between two images using FFT convolutions.

    Args:
    ----
        template: Template image.
        image: Search image.
        mode: Scipy.signal convolve cropping mode to use.

    Returns:
    -------
         Correlation map

    Credit to:
    https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py

    """
    template = template - np.mean(template)
    image = image - np.mean(image)
    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve
    # instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))

    out: ImageArrayType = convolve(image, ar, mode=mode)  # pyright: ignore[reportAssignmentType]

    first_part: ImageArrayType = convolve(np.square(image), a1, mode=mode)  # pyright: ignore[reportAssignmentType]

    second_part: ImageArrayType = np.square(convolve(image, a1, mode=mode))  # pyright: ignore[reportUnknownArgumentType]
    third_part = np.prod(template.shape)

    image = first_part - second_part / third_part

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)
    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


def get_similarity(
    shoemark: FeatureMapsArrayType,
    shoeprint: FeatureMapsArrayType,
) -> np.floating[Any]:
    """Use `normxcorr` to calculate the similarity between shoemark and shoeprint.

    Args:
    ----
        shoemark: Feature maps of shoemark.
        shoeprint: Feature maps of shoeprint.

    Returns:
    -------
        Similarity score

    """
    # Crop arrays by 2 pixels/edge to remove edge artifacts
    shoemark = shoemark[:, 2:-2, 2:-2]
    shoeprint = shoeprint[:, 2:-2, 2:-2]

    # Number of maps for both shoemark and shoeprint
    n_maps = len(shoemark)

    ncc_array = np.zeros(shoeprint.shape)

    for index in range(n_maps):
        mark_map: ImageArrayType = shoemark[index]
        print_map: ImageArrayType = shoeprint[index]

        ncc_array[index] = normxcorr(mark_map, print_map, "same")

    ncc_array: ImageArrayType = np.sum(ncc_array, axis=0)

    return np.max(ncc_array) / n_maps


class MultiProcessingTrackers:
    """Store variables which are shared among all processes.

    counter (int):  Overall count of processesed shoemarks.
    queue (int):    Message queue for printing to tqdm bar.
    rankings (int): Calculated ranks for all shoemarks.
    """

    def __init__(
        self,
        rankings_len: int,
    ):
        """Initialise trackers."""
        self.counter: Synchronized[int] = Value("i", 0)
        self.queue: Queue[str] = Queue()
        self.rankings: SynchronizedArray[int] = Array("i", rankings_len)


def compare_maps(
    shoemark_maps: list[FeatureMapsArrayType],
    shoeprint_maps: list[FeatureMapsArrayType],
    matching_pairs: list[int],
    config: Config,
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    """Compare a number of shoemarks and shoeprints with features extracted using NCC.

    Args:
        shoemark_maps: List of different shoemark feature maps.
        shoeprint_maps: List of differetn shoeprint feature_maps.
        matching_pairs: List of matching shoemark-shooeprint pairs, where
        index == shoemark ID and object == shoeprint ID.
        config: System config.

    """
    # Chunk prints by number of processes.
    n_processes = config["comparison"]["n_processes"]
    chunk_size = len(shoemark_maps) // n_processes
    chunk_extra = len(shoemark_maps) % n_processes
    chunks: list[list[FeatureMapsArrayType]] = []
    shoemark_ids: list[tuple[int, int]] = []
    start = 0

    for i in range(n_processes):
        end = start + chunk_size + (1 if i < chunk_extra else 0)
        chunks.append(shoemark_maps[start:end])
        shoemark_ids.append((start, end))  # Range of shoemark ids contained in chunk.
        start = end

    # Shared memory trackers
    trackers = MultiProcessingTrackers(len(shoemark_maps))

    # As shoeprint feature maps are shared among all chunks, they are
    # stored in shared memory.
    shoeprint_shared: list[tuple[SynchronizedArray[Any], tuple[int, int, int]]] = []
    for i in range(len(shoeprint_maps)):
        shape = cast(tuple[int, int, int], shoeprint_maps[i].shape)

        # Copy shoeprint maps into shared memory array
        shared = Array("f", shoeprint_maps[i].size)
        np_shared: ImageArrayType = cast(
            ImageArrayType,
            np.frombuffer(shared.get_obj(), dtype=np.float32).reshape(shape),  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportArgumentType]
        )
        np.copyto(np_shared, shoeprint_maps[i])

        shoeprint_shared.append((shared, shape))

    rotations = config["comparison"]["rotations"]
    scales = config["comparison"]["scales"]

    # Spawn a process for each chunk
    processes: list[Process] = []
    for i in range(n_processes):
        p = Process(
            target=_comparison_worker,
            args=(
                chunks[i],
                shoeprint_shared,
                shoemark_ids[i],
                matching_pairs,
                trackers,
                rotations,
                scales,
            ),
        )
        processes.append(p)
        p.start()

    # Update tqdm progress bar with values in queue and counter
    rotations_work = len(rotations) + 1 if rotations is not None else 1
    scales_work = len(scales) + 1 if scales is not None else 1
    work = rotations_work * scales_work * len(shoemark_maps)
    with tqdm(total=work) as pbar:
        while trackers.counter.value < work:
            while not trackers.queue.empty():
                message = trackers.queue.get()
                pbar.write(message)

            _ = pbar.update(trackers.counter.value - cast(int, pbar.n))
            pbar.refresh()

            time.sleep(1)

        _ = pbar.update(trackers.counter.value - cast(int, pbar.n))
        while not trackers.queue.empty():
            message = trackers.queue.get()
            pbar.write(message)

    # Join processes to enssure they have all terminated
    for p in processes:
        p.join()

    # No need to return a shared array
    return cast(
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.frombuffer(trackers.rankings.get_obj(), dtype=np.int32),  # pyright: ignore[reportCallIssue, reportArgumentType]
    )


# TODO test this actually works lol
def _apply_transformations(
    all_shoemark_maps: list[list[FeatureMapsArrayType]],
    transformations: list[float],
    original_shoemark_maps: list[FeatureMapsArrayType],
    transformation_type: Literal["rotate", "scale"],
) -> list[list[FeatureMapsArrayType]]:
    """Apply all of a set of different scale or rotation transformations to shoemark maps.

    Args:
        all_shoemark_maps: List containing n copies of the original list of shoemark maps, with each
        copy being a transformation (or if no transformations applied [original_shoemark_maps]).
        transformations: Set of scale or rotation transformations to apply.
        original_shoemark_maps: List containing original set of shoemark maps.
        transformation_type: Either rotate or scale transformation.

    Returns:
        List of all transformed shoemark maps with each scale transformation
        applied.

    """
    # List containing original shoemark maps at each scale
    scaled_shoemark_maps: list[list[FeatureMapsArrayType]] = []

    # A list of shoemark maps either at a number of transformations or 1 item long containing the
    # original list
    for transformed_shoemark_maps in all_shoemark_maps:
        for transformation in transformations:
            # A list of shoemark maps at a specific transformation
            new_transformed_shoemark_maps: list[FeatureMapsArrayType] = []
            # The individually transformed shoemarks
            for shoemark in transformed_shoemark_maps:
                new_shoemark: list[Image.Image] = []
                for feature_map in shoemark:
                    feature_map: ImageArrayType
                    feature_map_img = Image.fromarray(feature_map)

                    if transformation_type == "rotate":
                        feature_map_img = feature_map_img.rotate(transformation)
                    elif transformation_type == "scale":
                        feature_map_img = feature_map_img.resize(
                            (
                                int(feature_map_img.width * transformation),
                                int(feature_map_img.height * transformation),
                            ),
                        )

                    new_shoemark.append(feature_map_img)

                new_transformed_shoemark_maps.append(np.array(new_shoemark))

            scaled_shoemark_maps.append(new_transformed_shoemark_maps)

    scaled_shoemark_maps.insert(0, original_shoemark_maps)

    return scaled_shoemark_maps


def _comparison_worker(  # noqa: C901, PLR0913
    shoemark_maps: list[FeatureMapsArrayType],
    shoeprint_maps_shared: list[tuple[SynchronizedArray[Any], tuple[int, int, int]]],
    shoemark_ids: tuple[int, int],
    matching_pairs: list[int],
    trackers: MultiProcessingTrackers,
    rotations: list[float] | None,
    scales: list[float] | None,
) -> None:
    """Multiprocessing worker function to compare feature maps and rank results.

    Args:
        shoemark_maps: Feature maps of shoemarks to compare.
        shoeprint_maps_shared: Shoeprint feature maps (shared across all processes).
        shoemark_ids: IDs of shoemarks in shoemark_maps.
        matching_pairs: List of matching pairs where the index == shoemark_id and the
        object == shoeprint_id.
        trackers: Trackers for monitoring multiprocessing progress.
        rotations: Rotations to apply to each shoemark.
        scales: Scales to apply to each shoemark.

    """
    shoeprint_maps: list[FeatureMapsArrayType] = [np.empty(0)] * len(shoeprint_maps_shared)

    # Convert shared array into Numpy array (no data copying required
    for i in range(len(shoeprint_maps_shared)):
        if type(shoeprint_maps_shared[i]) is not np.ndarray:
            shoeprint_maps[i] = np.frombuffer(  # pyright: ignore[reportCallIssue, reportUnknownMemberType]
                shoeprint_maps_shared[i][0].get_obj(),  # pyright: ignore[reportArgumentType]
                dtype=np.float32,
            ).reshape(shoeprint_maps_shared[i][1])

    # Apply optional scaling or rotations

    if rotations is not None and scales is not None:
        rotated_shoemark_maps = _apply_transformations(
            [shoemark_maps.copy()],
            rotations,
            shoemark_maps.copy(),
            "rotate",
        )

        transformed_shoemark_maps = _apply_transformations(
            rotated_shoemark_maps,
            scales,
            shoemark_maps.copy(),
            "scale",
        )

    elif rotations is not None and scales is None:
        transformed_shoemark_maps = _apply_transformations(
            [shoemark_maps.copy()],
            rotations,
            shoemark_maps.copy(),
            "rotate",
        )

    elif rotations is None and scales is not None:
        transformed_shoemark_maps = _apply_transformations(
            [shoemark_maps.copy()],
            scales,
            shoemark_maps.copy(),
            "scale",
        )

    else:
        transformed_shoemark_maps = [shoemark_maps.copy()]

    similarities_all = np.zeros((len(shoemark_maps), len(shoeprint_maps)), dtype=np.float32)

    for shoemarks in transformed_shoemark_maps:
        for shoemark_id, shoemark in zip(range(*shoemark_ids), shoemarks, strict=True):
            similarities: list[np.floating[Any]] = []
            for shoeprint in shoeprint_maps:
                sim = get_similarity(shoemark, shoeprint)
                similarities.append(sim)

            shoemark_index = shoemark_id - shoemark_ids[0]
            for shoeprint_id, similarity in enumerate(similarities):
                if similarity > similarities_all[shoemark_index, shoeprint_id]:
                    similarities_all[shoemark_index, shoeprint_id] = similarity

            with trackers.counter.get_lock():
                trackers.counter.value += 1

    for shoemark_id, similarities in zip(range(*shoemark_ids), similarities_all, strict=True):
        rank = _get_rank(similarities, matching_pairs, shoemark_id)
        trackers.rankings[shoemark_id] = rank
        trackers.queue.put(f"Print {shoemark_id} true match ranked {rank}")


def _get_rank(similarities: list[np.float32], matching_pairs: list[int], print_id: int) -> int:
    """Sort similarities and then return the indexes in order of the sort."""
    # np.flip() is required as numpy sorts low -> high
    sorted_sims = np.flip(np.argsort(similarities))

    # Find the rank of the true match within the sorted array
    # matching_pairs[id+1] because the image id is equal to index + 1
    # Add 1 to the result as the resulting index calculated will be 1 less than the true rank
    return cast(int, np.where(sorted_sims == (matching_pairs[print_id]))[0][0] + 1)  # pyright: ignore[reportAny]
