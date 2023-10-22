#!/usr/bin/env python3

import csv

total_references = 1175
total_shoes = 300

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

        z = (p * total_references) / 100

        if z_i <= z:
            total_sum += 1

    return total_sum / total_shoes
