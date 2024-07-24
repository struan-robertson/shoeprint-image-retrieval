#!/usr/bin/env python3

import os
import shutil
from PIL import Image

def avg_img_size(dir):
    """
    Get average width and height of images in a directory
    """
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

def smallest_img_dir(dir):
    """
    Get smallest image in a directory
    """
    image_files = os.listdir(dir)

    return smallest_img(image_files, dir)

def smallest_img(image_files, dir):
    """
    Get smallest image given a list of file names and a directory
    """
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

def largest_img_dir(dir):
    """
    Get largest image in a directory
    """
    image_files = os.listdir(dir)

    return largest_img(image_files, dir)

def largest_img(image_files, dir):
    """
    Get largest image given a list of file names and a directory
    """
    largest_img_size = float(0)
    largest_img_name = None
    largest_img_dims = (0,0)

    for image_file in image_files:
        img_path = os.path.join(dir, image_file)

        with Image.open(img_path) as img:
            width, height = img.size
            image_size = width * height

            if image_size > largest_img_size:
                largest_img_name = image_file
                largest_img_size = image_size
                largest_img_dims = (width, height)

    return largest_img_name, largest_img_dims

def move_small_img(dir, smallest, destdir):
    """
    Move images under a specified size to a new directory
    """
    image_files = os.listdir(dir)

    for image_file in image_files:
        img_path = os.path.join(dir, image_file)

        with Image.open(img_path) as img:
            width, height = img.size

            if width < smallest[0] or height < smallest[1]:
                shutil.move(img_path, destdir)
                print(f"Moved {image_file}")
