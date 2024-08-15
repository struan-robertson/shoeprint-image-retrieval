"""Custom types."""

from typing import Any, Literal, TypeAlias

import numpy as np

ImageArrayType: TypeAlias = np.ndarray[
    tuple[int, int],
    np.dtype[np.floating[Any]],
]  # 2D array of floats
FeatureMapsArrayType: TypeAlias = np.ndarray[
    tuple[int, int, int],
    np.dtype[np.floating[Any]],
]  # 3D array of floats

DatasetTypeType: TypeAlias = Literal["FID-300", "Impress", "WVU2019"]
