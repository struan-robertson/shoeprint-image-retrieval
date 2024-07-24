#!/usr/bin/env python3

import numpy as np
import cv2
from timm.models import convmixer

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
    # detector = cv2.AKAZE_create() # type: ignore
    detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    total_similarity = 0

    for index in range(n_filters):

        # Print and shoe filters
        print_filter = print_[index]
        shoe_filter = shoe[index]

        # print_filter, shoe_filter = normalise_filters(print_filter, shoe_filter)
        print_filter = cv2.normalize(print_filter, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=None) # type: ignore
        shoe_filter = cv2.normalize(shoe_filter, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=None) # type: ignore

        # Find the keypoints and descriptors with DETECTOR
        kp1, des1 = detector.detectAndCompute(print_filter, None)
        kp2, des2 = detector.detectAndCompute(shoe_filter, None)

        if type(des1) != np.ndarray or type(des2) != np.ndarray :
            continue

        # Create BFMatcher (BRUTEFORCE_HAMMING) object
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        # matches = bf.match(des1, des2, k=2)
        matches = bf.knnMatch(des1, des2, k=2)

        # Sort them in ascending order of distance
        # matches = sorted(matches, key=lambda x: x.distance)

        try:
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
        except ValueError:
            continue

        # TODO currently counting the number of matches, not the _quality_
        # import ipdb; ipdb.set_trace()

        # Calculate similarity score based on matches
        if len(kp1) > 0 or len(kp2) > 0:
            # total_similarity += len(matches) / (len(kp1) + len(kp2))
            total_similarity += len(good) / (len(kp1) + len(kp2))
        # for match in matches[:5]:
            # total_similarity += match.distance


            # print(f"{similarity_score} {len(matches)} {len(kp1)} {len(kp2)}")
        # else:
        #     similarity_score = 0

    return total_similarity

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
