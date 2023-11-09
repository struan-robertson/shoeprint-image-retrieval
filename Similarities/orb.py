#!/usr/bin/env python3

import numpy as np
import cv2

def normalise_filters(print_, shoe):
    # Find global minimum and maximum across both filters
    global_min = min(np.min(print_), np.min(shoe))
    global_max = max(np.max(print_), np.max(shoe))

    normalised_print = 255 * (print_ - global_min) / (global_max - global_min)
    normalised_shoe = 255 * (print_ - global_min) / (global_max - global_min)

    normalised_print = np.uint8(normalised_print)
    normalised_shoe = np.uint8(normalised_shoe)

    return normalised_print, normalised_shoe

def get_similarity(print_, shoe):
    # Number of filters for both shoe and print
    n_filters = len(shoe)

    # detector = cv2.ORB_create() # type: ignore
    # detector = cv2.BRISK_create() # type: ignore
    detector = cv2.AKAZE_create() # type: ignore

    for index in range(n_filters):


        # Print and shoe filters
        print_filter = print_[index][1:-1, 1:-1]
        shoe_filter = shoe[index][1:-1, 1:-1]

        # print_filter, shoe_filter = normalise_filters(print_filter, shoe_filter)
        print_filter = cv2.normalize(print_filter, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=None) # type: ignore
        shoe_filter = cv2.normalize(shoe_filter, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=None) # type: ignore


        # Find the keypoints and descriptors with DETECTOR
        kp1, des1 = detector.detectAndCompute(print_filter, None)
        kp2, des2 = detector.detectAndCompute(shoe_filter, None)

        # Create BFMatcher (BRUTEFORCE_HAMMING) object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in ascending order of distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate similarity score based on matches
        if len(kp1) > 0 or len(kp2) > 0:
            similarity_score = len(matches) / (len(kp1) + len(kp2))
            print(f"{similarity_score} {len(matches)} {len(kp1)} {len(kp2)}")
        else:
            similarity_score = 0

    return 0

def compare_pictures(print_, shoe):

    detector = cv2.ORB_create() # type: ignore

    # Find the keypoints and descriptors with DETECTOR
    kp1, des1 = detector.detectAndCompute(print_, None)
    kp2, des2 = detector.detectAndCompute(shoe, None)

    # Create BFMatcher (BRUTEFORCE_HAMMING) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity score based on matches
    if len(kp1) > 0 or len(kp2) > 0:
        similarity_score = len(matches) / (len(kp1) + len(kp2))
    else:
        similarity_score = 0

    return similarity_score
