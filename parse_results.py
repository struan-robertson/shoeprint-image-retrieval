#!/usr/bin/env python3

import csv
import ipdb

total_references = 87
total_prints = 208
# total_prints = 51

def load_results(results_file):
    rankings = []
    with open(results_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            rankings.append(int(row[0]))

    return rankings

def cmp(rankings, p):

    total_sum = 0

    for rank in rankings:
        z_i = rank

        z = (p * total_references) / 100 # calculate p percent of number gallery images

        if z_i <= z:
            total_sum += 1

    return total_sum / total_prints

def cmp_all(rankings):

    S1 = cmp(rankings, 1) * 100
    S5 = cmp(rankings, 5) * 100
    S10 = cmp(rankings, 10) * 100
    S15 = cmp(rankings, 15) * 100
    S20 = cmp(rankings, 20) * 100

    print(f"S1:{S1:.2f} S5:{S5:.2f} S10:{S10:.2f} S15:{S15:.2f} S20:{S20:.2f}")
