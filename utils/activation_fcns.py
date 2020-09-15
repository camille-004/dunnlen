import numpy as np


# ----------Forward Propagation Activation Functions---------- #

def heaviside_step(fet: object, threshold: object) -> object:
    return np.where(fet >= threshold, 1, 0)


def sigmoid(fet: object) -> object:
    """Logistic activation function"""
    return 1 / (1 + np.exp(-fet))


def reLu(fet: object) -> object:
    """Rectified linear unit"""
    return np.where(fet > 0, fet, 0)

# ---------Backward Propagation Activation Functions---------- #

def sigmoid_deriv(d_act, fet):
    return (sigmoid(fet) * (1 - sigmoid(fet))) * d_act

def reLu_deriv(d_act, fet):
    return np.where(fet <= 0, 0, 1)