import numpy as np


def cross_entropy(y_hat, y):
    """Log loss for probability output"""
    if y == 1:
        return -np.log(y_hat)
    else:
        return -np.log(1 - y_hat)


def hinge(y_hat, y):
    """Hinge loss for classification"""
    return max(0, 1 - y_hat * y)


def huber(y_hat, y, delta=1):
    """Huber loss for regression, less sensitive to outliers than MSE"""
    return np.where(
        np.abs(y - y_hat) < delta, .5 * (
                y - y_hat
        ) ** 2, delta * (np.abs(y - y_hat) - 0.5 * delta))


def kullback_leibler_divergence(y_hat, y):
    """Kullback-Leibler Divergence"""
    return np.sum(y_hat * np.log((y_hat / y)))


def mae(y_hat, y):
    """Mean absolute error"""
    return np.sum(np.abs(y_hat - y))


def mse(y_hat, y):
    """Mean squared error"""
    return np.sum((y_hat - y) ** 2) / y.size


def mse_deriv(y_hat, y):
    """MSE derivative"""
    return y_hat - y
