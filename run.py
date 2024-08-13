"""Run the shoeprint image retrieval module."""

from src.shoeprint_image_retrieval.dataloader import Dataloader

dataloader = Dataloader("/home/srobertson/Datasets/WVU2019/Dataset Uncropped", (0,0), 16, "WVU2019")

for shoemark_images, shoeprint_images, matching_shoeprint_ids, block in dataloader:
    breakpoint()
