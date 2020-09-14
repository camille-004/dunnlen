import numpy as np


def heaviside_step(fet: object, threshold: object) -> object:
    return np.where(fet >= threshold, 1, 0)
