#!/usr/bin/env python3
"""Run the shoeprint image retrieval module."""

from src.shoeprint_image_retrieval.config import load_config
from src.shoeprint_image_retrieval.dataloader import Dataloader
from src.shoeprint_image_retrieval.network import Model
from src.shoeprint_image_retrieval.parse_results import cmp_all
from src.shoeprint_image_retrieval.similarity import compare_maps

# Load run.toml
config = load_config("run.toml")

dataloader = Dataloader(config)

print(f"{dataloader.num_clusters} clusters of image sizes found.")

for shoemark_images, shoeprint_images, matching_shoeprint_ids, block in dataloader:
    print(f"Cluster has {len(shoemark_images)} items.")

    model = Model(config, block)

    # Calculate feature maps for shoemark and shoeprint images
    shoemark_features = model.get_multiple_feature_maps(shoemark_images)
    shoeprint_features = model.get_multiple_feature_maps(shoeprint_images)

    print("Calculating ranks:")

    ranks = compare_maps(shoemark_features, shoeprint_features, matching_shoeprint_ids, config)

    cmp_all(
        list(ranks),
        total_shoeprints=len(dataloader.shoeprint_files),
        total_shoemarks=len(dataloader.shoemark_files),
    )
