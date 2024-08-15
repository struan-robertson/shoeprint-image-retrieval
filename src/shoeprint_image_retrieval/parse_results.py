"""Module to parse results into S scores."""


def cmp(rankings: list[int], p: int, total_shoeprints: int, total_shoemarks: int) -> float:
    """Calculate an S score for all shoemarks.

    Args:
        rankings: Rankings of correct shoeprint for each shoemark.
        p: Percentage for S score.
        total_shoeprints: Number of shoeprints in the dataset.
        total_shoemarks: Number of shoemarks in the dataset.

    """
    total_sum = 0

    for rank in rankings:
        z_i = rank

        z = (p * total_shoeprints) / 100  # calculate p percent of number gallery images

        if z_i <= z:
            total_sum += 1

    return total_sum / total_shoemarks


def cmp_all(rankings: list[int], total_shoeprints: int, total_shoemarks: int) -> None:
    """Print S1, S5, S10, S15, and S20 scores for all ranked shoemarks."""
    s1 = cmp(rankings, 1, total_shoeprints, total_shoemarks) * 100
    s5 = cmp(rankings, 5, total_shoeprints, total_shoemarks) * 100
    s10 = cmp(rankings, 10, total_shoeprints, total_shoemarks) * 100
    s15 = cmp(rankings, 15, total_shoeprints, total_shoemarks) * 100
    s20 = cmp(rankings, 20, total_shoeprints, total_shoemarks) * 100

    print(f"S1:{s1:.2f} S5:{s5:.2f} S10:{s10:.2f} S15:{s15:.2f} S20:{s20:.2f}")
